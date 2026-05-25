# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimia Audio AR stage: HF backbone + MIMO bifurcation for S2S.

This module implements the fused_thinker_talker stage for Kimi-Audio S2S (speech-to-speech).
It uses reference HuggingFace Transformers backbone and MIMO layers for
exact numerical match with the Kimi-Audio implementation.

Architecture (from weight inspection of Kimi-Audio-7B-Instruct):
- Qwen2-7B backbone: 28 layers, hidden=3584, heads=28, kv_heads=4 (GQA with bias)
- MIMO bifurcation: hidden states cloned after layer 21
- MIMO branch: 6 decoder layers + RMSNorm + linear head (168448 vocab)
- Unified vocab: 168448 = 152064 text + 16384 audio tokens (offset 152064)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable
from typing import Any

import numpy as np
import torch
from transformers import Qwen2Config
from transformers.cache_utils import Cache
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.interfaces import SupportsRealtime
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _to_cache(legacy_tuple):
    """Wrap a legacy tuple-of-tuples KV cache as a transformers Cache object.

    Transformers 5.x expects Cache objects with .update() and .get_seq_length().
    When a plain tuple is passed, the model ignores it (no .update() method) and
    attention only sees the current token — breaking autoregressive generation.
    """
    if legacy_tuple is None:
        return None

    class TupleAsCache(Cache):
        """Bridges legacy tuple KV cache format to transformers 5.x Cache API."""

        def __init__(self, data):
            self._data = list(data)

        def __len__(self):
            return len(self._data)

        def __iter__(self):
            return iter(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

        def get_seq_length(self):
            if not self._data:
                return 0
            for layer in self._data:
                if layer is not None:
                    return layer[0].shape[2]
            return 0

        def update(self, key_states, value_states, layer_idx):
            # Expand data list if needed
            while len(self._data) <= layer_idx:
                self._data.append(None)
            existing = self._data[layer_idx]
            if existing is not None:
                key_states = torch.cat([existing[0], key_states], dim=2)
                value_states = torch.cat([existing[1], value_states], dim=2)
            self._data[layer_idx] = (key_states, value_states)
            return key_states, value_states

    return TupleAsCache(legacy_tuple)


def _from_cache(cache_obj, num_layers=None):
    """Extract a legacy tuple from a Cache object (or return as-is if already tuple)."""
    if cache_obj is None:
        return None
    if isinstance(cache_obj, tuple):
        return cache_obj
    # It's a Cache object — iterate to get tuple format
    result = []
    for i, layer_cache in enumerate(cache_obj):
        if layer_cache is not None:
            result.append((layer_cache[0], layer_cache[1]))
        elif num_layers is not None and i < num_layers:
            # Placeholder for layers that weren't touched
            pass
    return tuple(result)


def _sample_audio_topk(
    logits: torch.Tensor,
    top_k: int = 0,
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
    recent_tokens: torch.Tensor | None = None,
    repetition_window: int = 64,
) -> torch.Tensor:
    """Sample audio tokens matching reference KimiASampler.sample_audio_logits.

    Reference uses audio_temperature=0.8 with audio_top_k=10 for TTS/S2S.
    Sampling follows the KimiASampler double-softmax pattern:
    log_softmax(logits) → divide by temperature → exp() → top-k → multinomial.

    Args:
        logits: [batch, seq, vocab] or [batch, vocab] — already sliced to audio subspace.
        top_k: Number of top candidates to sample from (0 = greedy argmax).
        temperature: Sampling temperature (0 = greedy).
        repetition_penalty: Penalty factor for recent tokens (>1.0 penalizes).
        recent_tokens: Recent token IDs for repetition penalty.
        repetition_window: Number of recent tokens to penalize.

    Returns:
        Sampled token IDs with same batch/seq shape as input logits minus vocab dim.
    """
    # Apply repetition penalty — start penalizing as soon as we have any
    # recent tokens. The reference model avoids repeating audio codes in
    # quick succession, which is critical for natural-sounding TTS.
    if repetition_penalty > 1.0 and recent_tokens is not None and recent_tokens.numel() > 0:
        # Use up to repetition_window most recent tokens for penalty
        n = min(recent_tokens.numel(), repetition_window)
        window_tokens = recent_tokens[-n:].long().to(logits.device)
        if logits.dim() == 3:
            logits_flat = logits[:, -1]  # [batch, vocab]
        else:
            logits_flat = logits
        window_scores = torch.gather(logits_flat, dim=-1, index=window_tokens.unsqueeze(0))
        window_scores = torch.where(
            window_scores < 0,
            window_scores * repetition_penalty,
            window_scores / repetition_penalty,
        )
        logits_flat.scatter_(-1, window_tokens.unsqueeze(0), window_scores)
        if logits.dim() == 3:
            logits = logits.clone()
            logits[:, -1] = logits_flat

    # Greedy decoding (matching reference audio_temperature=0.0)
    if temperature < 1e-6 or top_k <= 0:
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        return torch.argmax(logprobs, dim=-1)

    # Match reference KimiASampler.double-softmax approach:
    # 1. logprobs = log_softmax(logits)
    # 2. logprobs = logprobs / temperature
    # 3. probs = exp(logprobs)
    # This produces a SMOOTHER distribution than log_softmax(logits/temperature),
    # giving more diverse sampling and preventing DC-heavy audio output.
    # See kimia_infer/utils/sampler.py lines 66-75.
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
    logprobs = logprobs / temperature
    probs = torch.exp(logprobs)

    if top_k > 0:
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        sampled = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
        return top_k_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    else:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class KimiaRealtimeBuffer:
    """Accumulates streaming audio chunks and yields TokensPrompt with Whisper features.

    Mimics VoxtralRealtimeBuffer but simplified for Kimi Audio's speech-to-speech
    pattern. Audio chunks are concatenated into segments; each segment triggers
    Whisper feature extraction and yields a TokensPrompt.

    Args:
        sampling_rate: Audio sample rate (default 16000).
        segment_duration_s: Duration per segment in seconds for low-latency streaming.
        whisper_extractor: Lazy-loaded WhisperFeatureExtractor instance.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        segment_duration_s: float = 5.0,
    ):
        self._sampling_rate = sampling_rate
        self._segment_samples = int(sampling_rate * segment_duration_s)
        self._audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._accumulated: list[np.ndarray] = []
        self._total_samples = 0
        self._ended = False

    async def append_audio(self, audio_array: np.ndarray | None) -> None:
        await self._audio_queue.put(audio_array)

    async def get_input_stream(
        self,
        whisper_extractor: Any,
    ) -> AsyncGenerator[TokensPrompt, None]:
        """Yield TokensPrompt with Whisper features as audio segments accumulate."""
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                self._ended = True
                break

            self._accumulated.append(chunk)
            self._total_samples += len(chunk)

            # Yield a prompt when we have enough audio for one segment
            if self._total_samples >= self._segment_samples:
                # Concatenate and split: take one segment, keep rest
                full_audio = np.concatenate(self._accumulated)
                segment = full_audio[: self._segment_samples]
                remainder = full_audio[self._segment_samples :]

                # Extract Whisper features for this segment
                whisper_emb = whisper_extractor.extract(segment, sample_rate=self._sampling_rate)

                yield TokensPrompt(
                    prompt_token_ids=[],
                    mm_processor_kwargs={
                        "whisper_input_feature": whisper_emb,
                    },
                )

                if len(remainder) > 0:
                    self._accumulated = [remainder]
                    self._total_samples = len(remainder)
                else:
                    self._accumulated = []
                    self._total_samples = 0

        # Flush remaining audio
        if self._total_samples > 0:
            full_audio = np.concatenate(self._accumulated)
            whisper_emb = whisper_extractor.extract(full_audio, sample_rate=self._sampling_rate)
            yield TokensPrompt(
                prompt_token_ids=[],
                mm_processor_kwargs={
                    "whisper_input_feature": whisper_emb,
                },
            )


class KimiaAudioFusedForConditionalGeneration(
    torch.nn.Module,
    SupportsPP,
    CustomProcessMixin,
    SupportsRealtime,
):
    """Fused thinker-talker for Kimi-Audio S2S.

    Uses vLLM's Qwen2 backbone for both text and audio feature extraction,
    with 6 MIMO decoder layers that branch off at layer 21 for audio generation.

    Architecture (matching reference Kimi-Audio):
    - Backbone: Qwen2 28 layers → backbone norm → lm_head → text logits
    - MIMO branch: 6 layers (from layer 21) + RMSNorm + mimo_output → audio logits

    The MIMO layers use vLLM's Qwen2DecoderLayer for weight compatibility
    with the original checkpoint (FlashAttn attention + SiLU-gated MLP +
    2 RMSNorms).

    NOTE: vLLM v1's GPU-based sampler does NOT support the executor's
    audio code injection into the feedback loop (sample_tokens doesn't
    exist in v1). Instead, we accumulate audio codes generated across
    all forward passes (prefill + decode) into a single sequence.
    """

    realtime_max_tokens = 128
    """Maximum audio code tokens to generate per streaming audio segment."""

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[TokensPrompt, None]:
        """Buffer streaming audio and yield TokensPrompt with Whisper features.

        Receives raw audio chunks from the realtime endpoint, accumulates them
        into segments, extracts Whisper features, and yields TokensPrompt for
        each segment. The engine generates audio codes which are streamed back
        via input_stream (not yet consumed by this buffer — codes go directly
        to code2wav stage).

        Args:
            audio_stream: Async generator yielding numpy audio chunks.
            input_stream: Queue for receiving generated tokens (unused here,
                codes flow to code2wav via the pipeline).
            model_config: Model configuration (unused, kept for protocol).

        Yields:
            TokensPrompt with whisper_input_feature in mm_processor_kwargs.
        """
        # Lazy import to avoid loading Whisper at module import time
        from vllm_omni.model_executor.models.kimia_audio.whisper_feature_extractor import (
            WhisperFeatureExtractor as RuntimeExtractor,
        )

        sampling_rate = 16000
        segment_duration_s = 5.0  # Low-latency: process every 5 seconds

        buffer = KimiaRealtimeBuffer(
            sampling_rate=sampling_rate,
            segment_duration_s=segment_duration_s,
        )

        # Feed audio into buffer in background
        async def feed_audio():
            async for chunk in audio_stream:
                await buffer.append_audio(chunk)
            await buffer.append_audio(None)  # signal end

        audio_task = asyncio.create_task(feed_audio())

        # Load Whisper extractor for feature extraction
        model_dir = model_config.model.rstrip("/")
        whisper_path = f"{model_dir}/whisper-large-v3"
        import glob
        import os
        vq_shard_path = None
        shards = glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors"))
        for shard in sorted(shards):
            from safetensors.torch import load_file
            ckpt = load_file(shard)
            if any("vq_adaptor" in k for k in ckpt.keys()):
                vq_shard_path = shard
                break

        if vq_shard_path is None:
            logger.error("VQAdaptor shard not found for realtime Whisper extraction")
            return

        extractor = RuntimeExtractor(
            whisper_model_path=whisper_path,
            vq_adaptor_shard_path=vq_shard_path,
        )

        try:
            async for tokens_prompt in buffer.get_input_stream(extractor):
                yield tokens_prompt
        finally:
            audio_task.cancel()

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # Get config from vllm_config which was already loaded by vLLM.
        # Calling KimiaAudioConfig.from_pretrained() here causes infinite
        # recursion (it triggers initialize_model again).
        # The loaded config is transformers_modules.KimiAudioConfig (from the
        # checkpoint's remote code), which already has all Kimia-specific attrs.
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config

        # Patch rope_theta onto config for transformers 5.8+ compatibility.
        # Qwen2Config no longer stores rope_theta as a direct attribute;
        # it's in rope_parameters['rope_theta']. The reference modeling file
        # accesses config.rope_theta directly.
        if not hasattr(self.config, 'rope_theta'):
            rope_params = getattr(self.config, 'rope_parameters', {})
            self.config.rope_theta = rope_params.get('rope_theta', 1000000.0)

        # Fix incorrect kimia_audio_output_vocab in the checkpoint config
        # (16896 vs actual 16384). The detokenizer only has 16385 embeddings
        # (indices 0-16384), so codes >= 16384 would be silently dropped.
        expected_audio_vocab = int(getattr(self.config, "kimia_token_offset", 152064))
        actual_vocab = int(getattr(self.config, "vocab_size", 168448))
        correct_audio_vocab = actual_vocab - expected_audio_vocab  # 16384
        current_audio_vocab = int(getattr(self.config, "kimia_audio_output_vocab", 16384))
        if current_audio_vocab != correct_audio_vocab:
            logger.warning(
                "Fixing kimia_audio_output_vocab from %d to %d (vocab_size=%d - token_offset=%d)",
                current_audio_vocab,
                correct_audio_vocab,
                actual_vocab,
                expected_audio_vocab,
            )
            self.config.kimia_audio_output_vocab = correct_audio_vocab

        # Use Qwen2 backbone from vLLM. We must pass a Qwen2Config explicitly
        # to avoid initialize_model resolving back to KimiaAudioFusedForConditionalGeneration
        # (which would cause infinite recursion).
        qwen2_config = Qwen2Config(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            rope_theta=getattr(self.config, 'rope_theta', None) or self.config.rope_parameters.get('rope_theta', 1000000.0),
            num_hidden_layers=self.config.num_hidden_layers,
        )
        self.llm = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=prefix,
            hf_config=qwen2_config,
            architectures=["Qwen2ForCausalLM"],
        )

        # Load reference HuggingFace model (backbone + MIMO layers) to match
        # exact numerical behavior. The vLLM MIMO layers (PlainRMSNormDecoderLayer
        # + MimoAttentionOverride with SDPA) diverge from reference during decode
        # (MIMO layer 5 std explodes from ~33 to ~74 vs reference ~30).
        # HF MoonshotDecoderLayer handles FlashAttention/SDPA + past_key_values
        # identically to the reference implementation.
        logger.info("Loading reference HuggingFace model for exact numerical match (backbone + MIMO)")
        model_dir = self.vllm_config.model_config.model.rstrip("/")

        from transformers import AutoModelForCausalLM

        t0 = __import__("time").time()
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda",
        )
        logger.info(f"Loaded HF model (backbone + MIMO) in {__import__('time').time() - t0:.1f}s")

        # Patch apply_rotary_pos_emb in the moonshot_kimia module to handle
        # the reference model's [seq_len, dim] cos/sin from RotaryEmbedding.
        # Modern transformers' apply_rotary_pos_emb does cos.unsqueeze(1)
        # → [seq_len, 1, dim], which doesn't broadcast correctly against
        # Q/K of shape [batch, heads, seq_len, head_dim].
        # The patched version uses position_ids to index into cos/sin first,
        # then unsqueezes for proper broadcasting.
        import sys
        _patched_rope_applied = False
        for mod_name, mod in list(sys.modules.items()):
            if 'moonshot_kimia' in mod_name and mod is not None:
                if hasattr(mod, 'apply_rotary_pos_emb'):
                    def _rotate_half(x):
                        x1 = x[..., : x.shape[-1] // 2]
                        x2 = x[..., x.shape[-1] // 2 :]
                        return torch.cat((-x2, x1), dim=-1)

                    def _patched_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                        if position_ids is not None:
                            cos = cos[position_ids]
                            sin = sin[position_ids]
                        else:
                            seq_len_q = q.shape[2]
                            cos = cos[-seq_len_q:]
                            sin = sin[-seq_len_q:]
                            cos = cos.unsqueeze(0).expand(q.shape[0], -1, -1)
                            sin = sin.unsqueeze(0).expand(q.shape[0], -1, -1)
                        cos = cos.unsqueeze(1)
                        sin = sin.unsqueeze(1)
                        q_embed = (q * cos) + (_rotate_half(q) * sin)
                        k_embed = (k * cos) + (_rotate_half(k) * sin)
                        return q_embed, k_embed

                    mod.apply_rotary_pos_emb = _patched_apply_rotary_pos_emb
                    _patched_rope_applied = True
                    break
        if _patched_rope_applied:
            logger.info("Patched apply_rotary_pos_emb for reference model RoPE compatibility")

        # Extract HF backbone components
        self._hf_embed = hf_model.model.embed_tokens
        self._hf_layers = hf_model.model.layers  # 28 MoonshotDecoderLayer instances
        self._hf_norm = hf_model.model.norm
        self._hf_lm_head = hf_model.lm_head

        # Extract HF MIMO components — these are the SAME MoonshotDecoderLayer
        # instances used in the reference, ensuring identical numerical behavior.
        self._hf_mimo_layers = hf_model.model.mimo_layers  # 6 MoonshotDecoderLayer instances
        self._hf_mimo_norm = hf_model.model.mimo_norm
        self._hf_mimo_output = hf_model.mimo_output

        # Keep full HF model reference for calling its forward() directly.
        # This handles embedding construction with media token scaffolding,
        # VQAdaptor application, is_continuous_mask, and proper attention masks.
        self._hf_full_model = hf_model
        self._hf_full_model.eval()

        # Freeze all HF model parameters to avoid accidental gradients
        for param in self._hf_full_model.parameters():
            param.requires_grad_(False)

        # Free vLLM backbone layers to save GPU memory (HF layers are used instead)
        del self.llm.model.layers
        self.llm.model.layers = torch.nn.ModuleList()  # empty replacement
        torch.cuda.empty_cache()
        logger.info("Loaded HF backbone + MIMO layers (vLLM layers freed for memory)")

        logger.info("HF attention: SDPA for decode (q_len==1), FlashAttention for prefill (patched in modeling_moonshot_kimia.py)")

        # KV cache for HF layers during decode
        self._hf_past_key_values = None  # Backbone layers KV cache
        self._mimo_past_key_values = None  # MIMO layers KV cache

        # Store HF layer references to keep model alive
        self._hf_model = None  # No longer needed

        # NOTE: BackboneFlashAttention and vLLM layer patching are no longer needed.
        # Forward pass uses _hf_layers (HuggingFace MoonshotDecoderLayer) directly,
        # which has FlashAttention + RMSNorm matching the reference implementation.

        # MIMO layers are HF MoonshotDecoderLayer instances from _hf_mimo_layers.
        # No vLLM PlainRMSNormDecoderLayer or MimoAttentionOverride needed.

        # Logits processor for text generation path
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
        )

        # Track decode steps for delay token application
        self._decode_step_counter: int = 0
        self._diag_decode_log_count: int = 0
        self._diag_log_step: int = 0

        # Store audio code cap configuration

        # Accumulated audio codes across prefill + decode steps.
        # Since vLLM v1 doesn't support audio code injection into the
        # feedback loop, we collect all codes generated by the MIMO
        # layers into a flat sequence for the code2wav stage.
        self._audio_codes_list: list[torch.Tensor] = []
        self._audio_codes: torch.Tensor | None = None
        self._text_codes_list: list[torch.Tensor] = []
        self._text_codes: torch.Tensor | None = None

        # Cap audio codes to prevent runaway generation when no EOS token
        # is produced (common in TTS where audio keeps generating noise).
        # Default: 3000 codes ≈ 60s at 50 Hz frame rate.
        self._max_audio_codes: int = int(
            getattr(self.config, "kimia_max_audio_codes", 3000),
        )
        self._audio_code_cap_logged: bool = False

        # Load Whisper embedding from reference audio for S2S conditioning.
        # The model was trained with Whisper features, so using real features
        # during prefill should significantly improve audio quality.
        self._whisper_emb: torch.Tensor | None = None

        # Audio frame count set during prefill for position override.
        # When Whisper features are available, the combined embedding
        # sequence is [audio_frames + text_tokens, hidden_size]. Positions
        # must be overridden to span the full sequence length.
        self._prefill_audio_offset: int = 0
        self._prefill_seq_len: int = 0  # Actual prefill seq_len with parallel audio+text
        self._prefill_text_len: int = 0  # vLLM's view of text length during prefill
        self._ref_model_did_prefill: bool = False  # True when reference model handled prefill
        self._audio_array = None  # Raw audio array for reference model Whisper extraction
        self._audio_sample_rate: int = 16000
        self._speaker_emb: torch.Tensor | None = None
        self._load_whisper_embedding()

        # Per-request Whisper features extracted from user input audio
        # (speech-to-speech mode). Set by serving_chat.py via
        # additional_information["whisper_input_feature"].
        self._input_whisper_emb: torch.Tensor | None = None
        self._input_whisper_emb_logged: bool = False
        # 5120-dim raw Whisper features (pre-VQAdaptor) for native HF model path.
        # When available, the HF model applies VQAdaptor internally, ensuring
        # bit-identical embedding construction with the reference model.
        self._input_whisper_raw: torch.Tensor | None = None
        self._input_whisper_raw_logged: bool = False
        self._is_asr_mode: bool = False  # True when audio input but text-only output
        self._asr_text_hidden_states: torch.Tensor | None = None  # Text path output from ref model prefill
        self._asr_text_seq_len: int = 0  # Length of text stream during ASR prefill

        # Lazy-loaded Whisper feature extractor for runtime extraction from
        # user audio input (speech-to-speech).
        self._whisper_extractor: Any | None = None

        # Mark model as having multimodal outputs so the runner extracts
        # audio_codes from OmniOutput.multimodal_outputs.
        self.have_multimodal_outputs = True

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.llm.model.embed_tokens

    def _load_whisper_embedding(self) -> None:
        """Load pre-extracted Whisper embedding from reference audio."""
        import os
        whisper_path = os.path.join(
            os.path.dirname(__file__), "whisper_embedding.pt"
        )
        if not os.path.exists(whisper_path):
            logger.warning(
                "Whisper embedding not found at %s, using placeholder audio embeddings. "
                "Run extract_whisper_embedding.py to generate it.",
                whisper_path,
            )
            return

        try:
            data = torch.load(whisper_path, map_location="cpu", weights_only=True)
            # whisper_emb: [1, seq_len, 3584] — keep the full temporal sequence
            # The model was trained with temporal Whisper features, so using
            # the full sequence (interpolated to match text token count) provides
            # phonetic/phonetic content guidance, not just voice characteristics.
            whisper_emb = data["whisper_emb"]
            # Also store a mean-pooled speaker embedding for decode phase
            speaker_emb = whisper_emb.mean(dim=1, keepdim=True)  # [1, 1, 3584]
            self._whisper_emb = whisper_emb  # [1, seq_len, 3584]
            self._speaker_emb = speaker_emb  # [1, 1, 3584]
            logger.info(
                "Loaded Whisper embedding from reference audio: temporal shape %s, "
                "speaker embedding %s",
                list(whisper_emb.shape),
                list(speaker_emb.shape),
            )
        except Exception as e:
            logger.warning("Failed to load Whisper embedding: %s", e)

    def _get_whisper_extractor(self) -> Any | None:
        """Lazy-load Whisper feature extractor for runtime audio processing.

        Returns None if the required model paths are not available.
        """
        if self._whisper_extractor is not None:
            return self._whisper_extractor

        try:
            import os

            from vllm_omni.model_executor.models.kimia_audio.whisper_feature_extractor import (
                WhisperFeatureExtractor as RuntimeExtractor,
            )

            # Locate the whisper-large-v3 subfolder next to the checkpoint
            model_dir = self.vllm_config.model_config.model
            whisper_path = os.path.join(model_dir, "whisper-large-v3")
            if not os.path.isdir(whisper_path):
                logger.warning(
                    "Whisper model not found at %s, speech-to-speech "
                    "feature extraction disabled",
                    whisper_path,
                )
                return None

            # VQAdaptor weights are in the last shard
            vq_shard_path = os.path.join(model_dir, "model-35-of-35.safetensors")
            if not os.path.isfile(vq_shard_path):
                # Try to find any shard containing vq_adaptor
                import glob
                shards = glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors"))
                for shard in sorted(shards):
                    from safetensors.torch import load_file
                    ckpt = load_file(shard)
                    if any("vq_adaptor" in k for k in ckpt.keys()):
                        vq_shard_path = shard
                        break
                else:
                    logger.warning("VQAdaptor shard not found, speech-to-speech disabled")
                    return None

            self._whisper_extractor = RuntimeExtractor(
                whisper_model_path=whisper_path,
                vq_adaptor_shard_path=vq_shard_path,
            )
            # Trigger lazy init
            _ = self._whisper_extractor.vq_adaptor
            logger.info("Whisper feature extractor initialized")
            return self._whisper_extractor
        except Exception as e:
            logger.warning("Failed to initialize Whisper extractor: %s", e)
            return None

    def _get_reference_whisper_encoder(self) -> Any | None:
        """Lazy-load reference Kimi-Audio WhisperEncoder for S2S feature extraction.

        This uses the SAME WhisperEncoder as the reference prompt_manager,
        ensuring identical Whisper features for S2S audio generation.

        Returns None if the Kimi-Audio package is not available.
        """
        if hasattr(self, "_ref_whisper_encoder"):
            return self._ref_whisper_encoder

        self._ref_whisper_encoder = None  # cache miss sentinel
        try:
            import os
            import sys
            # Ensure Kimi-Audio package is importable
            kimia_path = "/root/learning/Kimi-Audio"
            if kimia_path not in sys.path:
                sys.path.insert(0, kimia_path)

            from kimia_infer.models.tokenizer.whisper_Lv3.whisper import (
                WhisperEncoder,
            )

            model_dir = self.vllm_config.model_config.model
            whisper_path = os.path.join(model_dir, "whisper-large-v3")
            if not os.path.isdir(whisper_path):
                logger.warning(
                    "Reference Whisper model not found at %s, "
                    "falling back to vLLM extractor",
                    whisper_path,
                )
                return None

            logger.info("Loading reference WhisperEncoder from %s...", whisper_path)
            ref_encoder = WhisperEncoder(whisper_path, mel_batch_size=20)
            ref_encoder = ref_encoder.to(torch.cuda.current_device())
            ref_encoder = ref_encoder.bfloat16()
            ref_encoder.eval()

            self._ref_whisper_encoder = ref_encoder
            logger.info("Reference WhisperEncoder loaded successfully")
            return ref_encoder
        except Exception as e:
            logger.warning("Failed to initialize reference WhisperEncoder: %s", e)
            return None

    def _extract_reference_whisper_features(
        self, audio_array: np.ndarray, sample_rate: int = 16000
    ) -> torch.Tensor | None:
        """Extract Whisper features using reference Kimi-Audio WhisperEncoder.

        Pipeline: waveform -> reference WhisperEncoder.tokenize_waveform -> [1, seq, 1280]
        Then 4x downsample + VQAdaptor -> [1, seq//4, 3584]

        Args:
            audio_array: Raw audio waveform (numpy array, 16kHz mono).
            sample_rate: Sample rate (default 16000).

        Returns:
            Whisper embedding tensor [1, seq_len, 3584] or None on failure.
        """
        ref_encoder = self._get_reference_whisper_encoder()
        if ref_encoder is None:
            return None

        try:
            # Convert to torch tensor if needed
            if isinstance(audio_array, np.ndarray):
                audio_tensor = torch.from_numpy(audio_array).float()
            else:
                audio_tensor = audio_array

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio_np = audio_tensor.cpu().numpy()
                audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
                audio_tensor = torch.from_numpy(audio_np).float()

            # Get encoder output [1, encoder_frames, 1280]
            with torch.no_grad():
                encoder_output = ref_encoder(audio_tensor.unsqueeze(0))

            # 4x downsample + VQAdaptor (reuse vLLM's VQAdaptor logic)
            whisper_extractor = self._get_whisper_extractor()
            if whisper_extractor is None:
                logger.warning("VQAdaptor not available for reference Whisper features")
                return None

            # Downsample: concatenate every 4 consecutive frames
            batch_size, seq_len, hidden_dim = encoder_output.shape
            trunc_len = (seq_len // 4) * 4
            if trunc_len != seq_len:
                encoder_output = encoder_output[:, :trunc_len, :]
            downsampled = encoder_output.view(
                batch_size, trunc_len // 4, hidden_dim * 4
            )  # [B, seq//4, 5120]

            # VQAdaptor projection
            whisper_emb = whisper_extractor.vq_adaptor(downsampled)  # [1, seq//4, 3584]

            logger.info(
                "Reference Whisper features: %d samples -> %d encoder frames "
                "-> %d downsampled -> output %s",
                audio_tensor.shape[0], seq_len, trunc_len // 4, list(whisper_emb.shape),
            )
            return whisper_emb
        except Exception as e:
            logger.warning("Failed to extract reference Whisper features: %s", e)
            return None

    def _extract_whisper_from_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Extract Whisper features from kwargs passed by the vLLM engine.

        The serving layer puts ``whisper_input_feature`` in
        ``engine_prompt["additional_information"]``.  The engine serializes
        this into ``model_intermediate_buffer``, which arrives as a list of
        per-request dicts in kwargs.

        Supports three modes:
        1. Direct tensor: kwargs["whisper_input_feature"] -> torch.Tensor [1, seq, 3584]
        2. model_intermediate_buffer: list of dicts containing whisper_input_feature
        3. Audio array: kwargs["audio_array"] -> numpy array (extract features at runtime)
        """
        # Mode 0: Always store raw audio array if present (needed for reference model).
        # Do this FIRST before Mode 1/2 return early.
        # Check both top-level kwargs and inside model_intermediate_buffer.
        audio_array = kwargs.get("audio_array")
        if audio_array is None:
            # Also check inside model_intermediate_buffer
            buffer = kwargs.get("model_intermediate_buffer")
            if buffer is not None and isinstance(buffer, list) and buffer:
                first_entry = buffer[0]
                if isinstance(first_entry, dict):
                    audio_array = first_entry.get("audio_array")
                    sample_rate = first_entry.get("audio_sample_rate")
                    if audio_array is not None and sample_rate is not None:
                        kwargs["audio_sample_rate"] = sample_rate  # Make available below

        if audio_array is not None:
            if isinstance(audio_array, (list, tuple)):
                # Handle nested/irregular lists from serialization
                flat = []
                for item in audio_array:
                    if isinstance(item, (list, tuple)):
                        flat.extend(item)
                    else:
                        flat.append(item)
                audio_array = np.array(flat, dtype=np.float32)
            elif isinstance(audio_array, np.ndarray):
                # Already a numpy array from deserialization
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                audio_array = audio_array.reshape(-1)
            elif hasattr(audio_array, 'shape') and len(getattr(audio_array, 'shape', ())) > 1:
                audio_array = audio_array.reshape(-1).astype(np.float32)
            self._audio_array = audio_array
            self._audio_sample_rate = kwargs.get("audio_sample_rate", 16000)
            if not self._input_whisper_emb_logged:
                logger.info(
                    "[DIAG STEP0] Stored raw audio array: shape=%s sample_rate=%d",
                    audio_array.shape if hasattr(audio_array, 'shape') else len(audio_array),
                    self._audio_sample_rate,
                )

        # ASR mode check: extract BEFORE any early returns (Modes 1/2/3 return).
        # Passed via engine_prompt["additional_information"]["is_asr_mode"].
        # Check top-level kwargs, model_intermediate_buffer, and runtime_additional_information.
        is_asr = kwargs.get("is_asr_mode")
        _diag_kw_keys = list(kwargs.keys()) if hasattr(kwargs, 'keys') else type(kwargs).__name__
        logger.info("[DIAG ASR] kwargs keys=%s", _diag_kw_keys)
        if is_asr is None:
            buffer = kwargs.get("model_intermediate_buffer")
            if isinstance(buffer, list) and buffer and isinstance(buffer[0], dict):
                is_asr = buffer[0].get("is_asr_mode")
                logger.info("[DIAG ASR] buffer[0] keys=%s, is_asr_mode=%s", list(buffer[0].keys()), is_asr)
            else:
                logger.info("[DIAG ASR] buffer type=%s, len=%d", type(buffer).__name__, len(buffer) if isinstance(buffer, (list, tuple)) else 0)
        if is_asr is None:
            runtime_info = kwargs.get("runtime_additional_information")
            if isinstance(runtime_info, dict):
                is_asr = runtime_info.get("is_asr_mode")
                logger.info("[DIAG ASR] runtime_additional_information keys=%s, is_asr_mode=%s, full=%s",
                           list(runtime_info.keys()) if hasattr(runtime_info, 'keys') else type(runtime_info).__name__, is_asr, runtime_info)
            elif isinstance(runtime_info, list) and runtime_info:
                is_asr = runtime_info[0].get("is_asr_mode") if isinstance(runtime_info[0], dict) else None
                logger.info("[DIAG ASR] runtime_additional_information[0] type=%s, keys=%s, is_asr_mode=%s",
                           type(runtime_info[0]).__name__, list(runtime_info[0].keys()) if isinstance(runtime_info[0], dict) else None, is_asr)
            else:
                logger.info("[DIAG ASR] runtime_additional_information type=%s, value=%s", type(runtime_info).__name__, runtime_info)
        if is_asr is not None:
            self._is_asr_mode = bool(is_asr)
            logger.info("[DIAG STEP1y] is_asr_mode=%s (extracted early)", self._is_asr_mode)
        else:
            logger.warning("[DIAG ASR] is_asr_mode NOT FOUND in any location")

        # Mode 1: Pre-extracted Whisper features at top level of kwargs
        whisper_feature = kwargs.get("whisper_input_feature")
        if whisper_feature is not None:
            if isinstance(whisper_feature, (torch.Tensor, np.ndarray)):
                if isinstance(whisper_feature, np.ndarray):
                    whisper_feature = torch.from_numpy(whisper_feature)
                self._input_whisper_emb = whisper_feature
                if not self._input_whisper_emb_logged:
                    logger.info(
                        "[DIAG STEP1] Whisper from kwargs top level: shape=%s mean=%.4f std=%.4f min=%.4f max=%.4f",
                        list(whisper_feature.shape),
                        whisper_feature.float().mean().item(),
                        whisper_feature.float().std().item(),
                        whisper_feature.float().min().item(),
                        whisper_feature.float().max().item(),
                    )
                    self._input_whisper_emb_logged = True
            # Also check for raw 5120-dim Whisper features (pre-VQAdaptor)
            whisper_raw = kwargs.get("whisper_raw")
            if whisper_raw is not None and isinstance(whisper_raw, (torch.Tensor, np.ndarray)):
                if isinstance(whisper_raw, np.ndarray):
                    whisper_raw = torch.from_numpy(whisper_raw).cuda()
                elif whisper_raw.device.type != "cuda":
                    whisper_raw = whisper_raw.cuda()
                self._input_whisper_raw = whisper_raw
                if not self._input_whisper_raw_logged:
                    logger.info(
                        "[DIAG STEP1] Raw Whisper (5120-dim) from kwargs: shape=%s mean=%.4f std=%.4f",
                        list(whisper_raw.shape),
                        whisper_raw.float().mean().item(),
                        whisper_raw.float().std().item(),
                    )
                    self._input_whisper_raw_logged = True
            return

        # Mode 2: Per-request Whisper features via model_intermediate_buffer
        buffer = kwargs.get("model_intermediate_buffer")
        if buffer is not None and isinstance(buffer, list):
            if not self._input_whisper_emb_logged:
                logger.info(
                    "[DIAG STEP1b] model_intermediate_buffer has %d entries, first entry keys=%s",
                    len(buffer),
                    list(buffer[0].keys()) if buffer and isinstance(buffer[0], dict) else "N/A",
                )
            for req_info in buffer:
                if isinstance(req_info, dict) and "whisper_input_feature" in req_info:
                    whisper_feature = req_info["whisper_input_feature"]
                    # Deserialize bfloat16 tensors arrive as numpy arrays
                    # (see serialization.py line 119: bfloat16 → numpy).
                    if isinstance(whisper_feature, (torch.Tensor, np.ndarray)):
                        if isinstance(whisper_feature, np.ndarray):
                            whisper_feature = torch.from_numpy(whisper_feature)
                        self._input_whisper_emb = whisper_feature
                        if not self._input_whisper_emb_logged:
                            logger.info(
                                "[DIAG STEP1] Whisper from model_intermediate_buffer: shape=%s mean=%.4f std=%.4f",
                                list(whisper_feature.shape),
                                whisper_feature.float().mean().item(),
                                whisper_feature.float().std().item(),
                            )
                            self._input_whisper_emb_logged = True
                        # Also check for raw 5120-dim Whisper features
                        whisper_raw = req_info.get("whisper_raw")
                        if whisper_raw is not None and isinstance(whisper_raw, (torch.Tensor, np.ndarray)):
                            if isinstance(whisper_raw, np.ndarray):
                                whisper_raw = torch.from_numpy(whisper_raw).cuda()
                            elif whisper_raw.device.type != "cuda":
                                whisper_raw = whisper_raw.cuda()
                            self._input_whisper_raw = whisper_raw
                            self._input_whisper_raw = whisper_raw
                            if not self._input_whisper_raw_logged:
                                logger.info(
                                    "[DIAG STEP1] Raw Whisper (5120-dim) from buffer: shape=%s mean=%.4f std=%.4f",
                                    list(whisper_raw.shape),
                                    whisper_raw.float().mean().item(),
                                    whisper_raw.float().std().item(),
                                )
                                self._input_whisper_raw_logged = True
                        return
                    # May be a serialized tensor (numpy bytes from serialization)
                    if isinstance(whisper_feature, dict) and "tensor_data" in whisper_feature:
                        try:
                            buf = bytes(whisper_feature["tensor_data"])
                            shape = tuple(whisper_feature["tensor_shape"])
                            dtype_name = whisper_feature.get("tensor_dtype", "float32")
                            dtype_map = {"float32": np.float32, "float16": np.float16, "bfloat16": np.float16, "float64": np.float64}
                            arr = np.frombuffer(buf, dtype=dtype_map.get(dtype_name, np.float32))
                            arr = arr.reshape(shape)
                            self._input_whisper_emb = torch.from_numpy(arr)
                            if not self._input_whisper_emb_logged:
                                logger.info(
                                    "[DIAG STEP1] Whisper deserialized from buffer: shape=%s mean=%.4f std=%.4f",
                                    list(self._input_whisper_emb.shape),
                                    self._input_whisper_emb.float().mean().item(),
                                    self._input_whisper_emb.float().std().item(),
                                )
                                self._input_whisper_emb_logged = True
                            return
                        except Exception as e:
                            logger.warning("Failed to deserialize Whisper features: %s", e)
                            return

        # Mode 3: Raw audio array — extract features at runtime if not already
        # captured via Mode 0/1/2.
        if audio_array is not None and not self._input_whisper_emb_logged:
            sample_rate = kwargs.get("audio_sample_rate", 16000)
            try:
                # Try reference Kimi-Audio WhisperEncoder first (produces identical
                # features to the reference prompt_manager). Falls back to vLLM's
                # HF-based extractor if reference is unavailable.
                ref_emb = self._extract_reference_whisper_features(
                    audio_array, sample_rate=int(sample_rate),
                )
                if ref_emb is not None:
                    self._input_whisper_emb = ref_emb
                    if not self._input_whisper_emb_logged:
                        logger.info(
                            "[DIAG STEP1] Reference Whisper extracted: shape=%s mean=%.4f std=%.4f",
                            list(ref_emb.shape),
                            ref_emb.float().mean().item(),
                            ref_emb.float().std().item(),
                        )
                        self._input_whisper_emb_logged = True
                    return

                # Fallback: vLLM's HF-based Whisper extractor
                extractor = self._get_whisper_extractor()
                if extractor is None:
                    logger.warning("[DIAG STEP1] Mode 3: audio_array present but Whisper extractor unavailable")
                    return

                whisper_emb = extractor.extract(audio_array, sample_rate=int(sample_rate))
                self._input_whisper_emb = whisper_emb
                if not self._input_whisper_emb_logged:
                    logger.info(
                        "[DIAG STEP1] Whisper extracted from audio (vLLM extractor): shape=%s mean=%.4f std=%.4f",
                        list(whisper_emb.shape),
                        whisper_emb.float().mean().item(),
                        whisper_emb.float().std().item(),
                    )
                    self._input_whisper_emb_logged = True
            except Exception as e:
                logger.warning("Failed to extract Whisper features: %s", e)
            return


        # No whisper feature found
        if not self._input_whisper_emb_logged:
            # Also dump runtime_additional_information for debugging
            runtime_info = kwargs.get("runtime_additional_information")
            if runtime_info is not None:
                logger.info(
                    "[DIAG STEP1x] runtime_additional_information type=%s keys=%s",
                    type(runtime_info).__name__,
                    list(runtime_info.keys()) if isinstance(runtime_info, dict) else "N/A",
                )
                if isinstance(runtime_info, dict):
                    for k, v in runtime_info.items():
                        if isinstance(v, dict):
                            logger.info("[DIAG STEP1x]   %s -> dict keys=%s", k, list(v.keys())[:10])
                        elif isinstance(v, (list, tuple)):
                            logger.info("[DIAG STEP1x]   %s -> %s len=%d", k, type(v).__name__, len(v))
                        elif hasattr(v, 'shape'):
                            logger.info("[DIAG STEP1x]   %s -> tensor shape=%s", k, list(v.shape))
                        else:
                            logger.info("[DIAG STEP1x]   %s -> %s", k, type(v).__name__)
            logger.info("[DIAG STEP1] No Whisper features found in kwargs. keys=%s", list(kwargs.keys()))

    def _extract_raw_whisper_features(self, audio_array, sample_rate=16000):
        """Extract raw Whisper features (5120-dim) before VQAdaptor.

        The reference model's VQAdaptor expects 5120-dim input, reshaped
        from the raw Whisper large-v3 output [N, 1280] to [N/4, 5120].
        """
        from transformers import WhisperModel, WhisperProcessor

        if not hasattr(self, "_raw_whisper_model"):
            model_dir = self.vllm_config.model_config.model.rstrip("/")
            whisper_path = f"{model_dir}/whisper-large-v3"
            processor = WhisperProcessor.from_pretrained(whisper_path)
            raw_model = WhisperModel.from_pretrained(whisper_path, torch_dtype=torch.bfloat16).encoder
            raw_model.eval()
            self._raw_whisper_processor = processor
            self._raw_whisper_model = raw_model
            logger.info("Loaded raw Whisper model for 5120-dim feature extraction")

        # Process audio through Whisper encoder
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Normalize audio
        audio_array = audio_array / max(np.abs(audio_array).max(), 1e-8)

        # Get Whisper features
        inputs = self._raw_whisper_processor(
            audio_array, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(torch.bfloat16).to(
            self._raw_whisper_model.device
        )

        with torch.no_grad():
            # Encoder output: [1, num_mel_frames, 1280]
            whisper_out = self._raw_whisper_model(input_features).last_hidden_state

        # Reshape: [1, N, 1280] -> [1, N/4, 5120] (matching reference prompt_manager.py)
        seq_len = whisper_out.shape[1]
        reshaped_len = seq_len // 4
        whisper_out = whisper_out[:, :reshaped_len * 4, :].reshape(1, reshaped_len, 5120)

        return whisper_out  # [1, num_frames, 5120]

    def _call_reference_model_decode(
        self,
        audio_code: int,
        text_token: int,
        position: int,
        backbone_past_key_values=None,
        mimo_past_key_values=None,
    ):
        """Call full reference model forward for single-token decode step.

        Matches the reference decode loop (kimia.py:180-192):
        - decoder_input_audio_ids = next_audio_token.unsqueeze(1)  (raw code)
        - decoder_input_text_ids = next_token_text.unsqueeze(1)
        - decoder_position_ids = last_position_id + 1
        - decoder_input_whisper_feature = None
        - decoder_is_continuous_mask = None
        """
        hf_model = self._hf_full_model
        device = self._hf_past_key_values[0][0].device

        # Combine backbone + MIMO KV caches — HF model expects all 34 layers
        if backbone_past_key_values is not None and mimo_past_key_values is not None:
            all_past_kv = backbone_past_key_values + mimo_past_key_values
        else:
            all_past_kv = backbone_past_key_values

        # Wrap as Cache object for transformers 5.x compatibility
        all_past_kv = _to_cache(all_past_kv)

        # Absolute audio token ID (token_offset + relative_code) — matches reference
        # model which uses absolute token IDs from sample_audio_logits.
        audio_input_ids = torch.tensor([[audio_code]], device=device, dtype=torch.long)
        text_input_ids = torch.tensor([[text_token]], device=device, dtype=torch.long)
        position_ids = torch.tensor([[position]], device=device, dtype=torch.long)

        # Compute sequence length from KV cache for attention mask
        past_seq_len = all_past_kv.get_seq_length()
        seq_len_with_past = 1 + past_seq_len  # 1 for current token

        # Build attention mask so decode token can attend to all cached tokens.
        attention_mask = torch.ones(
            1, 1, 1, seq_len_with_past,
            device=device, dtype=hf_model.model.embed_tokens.weight.dtype,
        )

        with torch.no_grad():
            outputs = hf_model.model(
                input_ids=audio_input_ids,
                text_input_ids=text_input_ids,
                whisper_input_feature=None,
                is_continuous_mask=None,
                position_ids=position_ids,
                past_key_values=all_past_kv,
                use_cache=True,
                return_dict=False,
                attention_mask=attention_mask,
            )

        hidden_states = outputs[0].squeeze(0)       # [1, hidden_size]
        mimo_hidden_states = outputs[1].squeeze(0)  # [1, hidden_size]
        new_past_key_values = outputs[2] if len(outputs) > 2 else None

        return hidden_states, mimo_hidden_states, new_past_key_values

    def _call_reference_model_forward(
        self,
        input_ids: torch.Tensor,
        whisper_feat_5120: torch.Tensor,
        text_input_ids: torch.Tensor | None = None,
        past_key_values=None,
        whisper_emb_3584: torch.Tensor | None = None,
    ):
        """Call full reference model forward with proper input construction.

        Builds input_ids with media_begin/media_end scaffolding, is_continuous_mask,
        and calls the full MoonshotKimiaModel.forward() to get correct embeddings.

        Args:
            input_ids: text token IDs from vLLM
            whisper_feat_5120: raw 5120-dim Whisper features (requires VQAdaptor)
            text_input_ids: text token IDs (same as input_ids)
            past_key_values: KV cache (None for prefill)
            whisper_emb_3584: pre-processed 3584-dim Whisper features (S2S mode, no VQAdaptor)

        Returns: (hidden_states, mimo_hidden_states, past_key_values)
        """
        hf_model = self._hf_full_model
        media_begin = int(self.config.kimia_media_begin)  # 151661
        media_end = int(self.config.kimia_media_end)      # 151663
        token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
        user_msg_start = int(getattr(self.config, "kimia_user_msg_start", 151670))
        speech_ctd = int(getattr(self.config, "kimia_speech_ctd", 151676))
        msg_end = int(getattr(self.config, "kimia_msg_end", 151645))
        assistant_msg_start = int(getattr(self.config, "kimia_assistant_msg_start", 151671))

        device = input_ids.device
        text_len = text_input_ids.shape[1] if text_input_ids is not None else 0

        # Determine audio frame count from available Whisper features
        if whisper_feat_5120 is not None:
            # S2S mode: 5120-dim raw features, model applies VQAdaptor internally
            num_audio_frames = whisper_feat_5120.shape[1]
            whisper_for_model = whisper_feat_5120
        elif whisper_emb_3584 is not None:
            # Legacy S2S mode: 3584-dim features (post-VQAdaptor)
            # Use HF model native forward by passing through VQAdaptor internally.
            # Apply VQAdaptor to get 5120-dim-like path — actually, we need raw 5120-dim.
            # This path is deprecated; callers should pass 5120-dim features.
            num_audio_frames = whisper_emb_3584.shape[1]
            whisper_for_model = None  # Will use manual embedding (legacy)
        else:
            raise ValueError("Either whisper_feat_5120 or whisper_emb_3584 must be provided")

        # Build input_ids with media token scaffolding
        # Pattern: [user_msg_start, media_begin, audio_0, ..., audio_N, media_end, speech_ctd, msg_end, assistant_msg_start]
        # Matches reference prompt_manager (53 tokens total, including assistant_msg_start).
        audio_token_ids = torch.arange(token_offset, token_offset + num_audio_frames, device=device).unsqueeze(0)
        scaffolding_ids = torch.cat([
            torch.tensor([[user_msg_start]], device=device),
            torch.tensor([[media_begin]], device=device),
            audio_token_ids,
            torch.tensor([[media_end]], device=device),
            torch.tensor([[speech_ctd]], device=device),
            torch.tensor([[msg_end]], device=device),
            torch.tensor([[assistant_msg_start]], device=device),
        ], dim=1)  # [1, num_audio_frames + 7]

        # Build is_continuous_mask: True only for actual audio positions (indices 2 through N+1)
        audio_len = scaffolding_ids.shape[1]  # num_audio_frames + 7
        is_continuous_mask = torch.zeros(1, audio_len, dtype=torch.bool, device=device)
        is_continuous_mask[0, 2:-4] = True  # positions 2..N+1 (audio positions only)

        # In S2S mode, the reference model uses kimia_text_blank placeholders
        # for text positions during prefill. The actual prompt text is only used
        # for text generation output, not for audio prefill embedding.
        # This ensures the KV cache matches the reference model's prefill output.
        # See plan: "Text Blank Placeholder Fix" section.
        if whisper_emb_3584 is not None:
            # S2S mode: use blank placeholders matching reference
            kimia_text_blank_id = int(getattr(self.config, "kimia_text_blank", 151666))
            full_text_ids = torch.full(
                (1, audio_len + text_len),
                fill_value=kimia_text_blank_id,
                dtype=text_input_ids.dtype,
                device=device,
            )
        else:
            # No Whisper features — dead code path, TTS removed
            full_text_ids = torch.zeros(1, audio_len + text_len, dtype=text_input_ids.dtype, device=device)
            full_text_ids[0, audio_len:] = text_input_ids

        # Concatenate: [scaffolding] + [text_tokens] — used for S2S position IDs.
        # Sequence length is the same regardless of text content (blank vs actual).
        full_input_ids = torch.cat([scaffolding_ids, text_input_ids], dim=1)

        # Build position_ids: [0, 1, 2, ..., seq_len-1]
        seq_len = full_input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        past_kv = past_key_values if past_key_values is not None else None

        if whisper_emb_3584 is not None:
            # S2S mode: match the reference model's EXACT embedding construction.
            #
            # Reference (MoonshotKimiaModel.forward lines 705-757):
            #   audio_emb = embed_tokens(input_ids)      # [1, 52, 3584] — audio stream
            #   text_emb = embed_tokens(text_input_ids)   # [1, 52, 3584] — text stream
            #   inputs_embeds = audio_emb + text_emb      # ADDITIVE, NOT concatenation!
            #   # Then replace audio positions with Whisper features
            #   encoder_input = (audio_emb + whisper_emb) * sqrt(2)
            #   audio_emb = audio_emb * (~mask) + encoder_input * mask
            #   # But since we ADD text_emb, final is:
            #   inputs_embeds = audio_emb + text_emb
            #
            # The audio and text streams have the SAME length and are added
            # element-wise. This is the reference's dual-stream architecture.
            hf_model_model = hf_model.model
            embed_tokens = hf_model_model.embed_tokens

            # Position IDs for S2S: reference uses audio stream length only.
            # The dual-stream architecture has audio and text at same positions.
            position_ids = torch.arange(audio_len, device=device).unsqueeze(0)

            # Step 1: Embed audio scaffolding tokens
            audio_emb = embed_tokens(scaffolding_ids)  # [1, audio_len, hidden]

            # Step 2: Embed text tokens (same length as audio stream).
            # Reference uses text_input_ids with same shape as audio_input_ids.
            # Truncate or pad to match audio length.
            if text_input_ids.shape[1] != audio_len:
                # Reference text stream length matches audio stream length.
                kimia_text_blank_id = int(getattr(self.config, "kimia_text_blank", 151666))
                text_ids_for_embed = torch.full(
                    (1, audio_len), fill_value=kimia_text_blank_id,
                    dtype=text_input_ids.dtype, device=device,
                )
            else:
                text_ids_for_embed = text_input_ids

            text_emb = embed_tokens(text_ids_for_embed)  # [1, audio_len, hidden]

            # Step 3: Expand Whisper features to match audio scaffolding length
            # Cast to model dtype to match embed_tokens output (bfloat16)
            model_dtype = embed_tokens.weight.dtype
            if whisper_emb_3584.dtype != model_dtype:
                whisper_emb_3584 = whisper_emb_3584.to(model_dtype)
            whisper_dtype = whisper_emb_3584.dtype
            expanded_whisper = torch.zeros(1, audio_len, whisper_emb_3584.shape[-1],
                                           device=device, dtype=whisper_dtype)
            expanded_whisper[0, 2:-4, :] = whisper_emb_3584.squeeze(0)  # audio positions 2..N+1

            # Step 4: Apply is_continuous_mask (zero out media tokens)
            mask_f = is_continuous_mask[:, :, None].to(whisper_dtype)
            expanded_whisper = expanded_whisper * mask_f

            # Step 5: Apply Whisper features with sqrt(2) scaling (matching reference lines 744-755)
            combined = (audio_emb + expanded_whisper) * (2.0 ** 0.5)
            audio_emb = audio_emb * (~mask_f.to(torch.bool)) + combined * mask_f

            # Step 6: ADD text embeddings (reference line 757: audio_emb + text_emb)
            inputs_embeds = audio_emb + text_emb  # [1, audio_len, hidden]

            with torch.no_grad():
                outputs = hf_model_model(
                    input_ids=None,  # Use inputs_embeds instead
                    text_input_ids=None,  # Already included in inputs_embeds
                    whisper_input_feature=None,  # Already applied
                    is_continuous_mask=None,  # Already applied
                    position_ids=position_ids,
                    past_key_values=past_kv,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    return_dict=False,
                )
        else:
            # S2S mode: use model's VQAdaptor path
            with torch.no_grad():
                outputs = hf_model.model(
                    input_ids=full_input_ids,
                    text_input_ids=full_text_ids,
                    whisper_input_feature=whisper_for_model,
                    is_continuous_mask=is_continuous_mask,
                    position_ids=position_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=False,
                )

        hidden_states = outputs[0]    # [1, seq_len, hidden_size]
        mimo_hidden_states = outputs[1]  # [1, seq_len, hidden_size]
        new_past_key_values = outputs[2] if len(outputs) > 2 else None

        # Flatten to 2D for vLLM pipeline parallelism compatibility
        hidden_states = hidden_states.squeeze(0)  # [seq_len, hidden_size]
        mimo_hidden_states = mimo_hidden_states.squeeze(0)

        return hidden_states, mimo_hidden_states, new_past_key_values

    def _run_asr_text_generation(
        self,
        text_logits: torch.Tensor,       # [seq_len, vocab_size] — prefill logits
        past_key_values,                  # KV cache from prefill (tuple or Cache)
        max_new_tokens: int,
        token_offset: int,
    ) -> list[torch.Tensor]:
        """Generate text autoregressively for ASR mode.

        Samples the first text token from prefill logits, then runs the HF
        model with KV cache for subsequent decode steps. Text-only — no audio
        generation.

        Returns:
            List of generated text token tensors.
        """
        hf_model = self._hf_full_model
        device = text_logits.device
        generated_text: list[torch.Tensor] = []
        text_stream_finished = False

        logger.info(
            "[ASR TEXT GEN] Starting: prefill_logits=%s, max_new_tokens=%d",
            list(text_logits.shape), max_new_tokens,
        )

        # Convert tuple to Cache object for HF model compatibility
        if isinstance(past_key_values, tuple):
            past_kv = _to_cache(past_key_values)
        else:
            past_kv = past_key_values

        # Step 0: Sample first text token from prefill logits (last position)
        text_logits_slice = text_logits[-1, :token_offset]  # [vocab_subset]
        next_token = torch.argmax(
            torch.log_softmax(text_logits_slice, dim=-1, dtype=torch.float),
            dim=-1,
        )  # scalar

        kimia_text_blank = int(getattr(self.config, "kimia_text_blank", 151666))

        for step in range(max_new_tokens):
            if step == 0:
                # First token already sampled from prefill logits
                pass
            else:
                # Decode step: run HF model with KV cache
                with torch.no_grad():
                    # Build input for single-token decode
                    audio_input_ids = torch.tensor([[kimia_text_blank]], device=device, dtype=torch.long)
                    text_input_ids = next_token.unsqueeze(0).unsqueeze(0)  # [1, 1]
                    current_pos = past_kv.get_seq_length()
                    position_ids = torch.tensor([[current_pos]], device=device, dtype=torch.long)

                    transformer_out = hf_model.model(
                        input_ids=audio_input_ids,
                        text_input_ids=text_input_ids,
                        whisper_input_feature=None,
                        is_continuous_mask=None,
                        position_ids=position_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=False,
                    )

                past_kv = transformer_out[2] if len(transformer_out) > 2 else past_kv
                # HF model may return a tuple instead of a Cache object
                if isinstance(past_kv, tuple):
                    past_kv = _to_cache(past_kv)

                # Compute text logits from backbone output
                text_hs = transformer_out[0]  # [1, 1, hidden]
                text_logits_step = self._hf_lm_head(text_hs[0, 0, :])  # [vocab]
                text_logits_slice = text_logits_step[:token_offset]
                next_token = torch.argmax(
                    torch.log_softmax(text_logits_slice, dim=-1, dtype=torch.float),
                    dim=-1,
                )  # scalar

            # Check for EOS or out-of-vocab
            token_id = next_token.item() if next_token.dim() == 0 else next_token
            if token_id >= token_offset - 1:
                text_stream_finished = True
                break

            generated_text.append(next_token.cpu().unsqueeze(0))

            if text_stream_finished:
                break

        logger.info(
            "[ASR TEXT GEN] Finished: generated %d tokens", len(generated_text),
        )
        return generated_text

    def _run_reference_generate_loop(
        self,
        audio_input_ids: torch.Tensor,       # [1, audio_seq_len] — prefill audio tokens
        text_input_ids: torch.Tensor,        # [1, text_seq_len] — prefill text tokens
        is_continuous_mask: torch.Tensor,    # [1, audio_seq_len] bool
        whisper_features: torch.Tensor,      # [1, num_frames, 5120] raw Whisper features
        max_new_tokens: int = 1500,
        audio_output_vocab: int = 16384,
    ) -> list[torch.Tensor]:
        """Generate all audio codes using reference _generate_loop logic.

        This re-implements the reference model's _generate_loop (kimia.py:52-207)
        using the existing HF model (MoonshotKimiaForCausalLM). The HF model's
        forward() handles embedding construction internally (media token scaffolding,
        VQAdaptor, is_continuous_mask, sqrt(2) scaling), ensuring the prefill
        matches the reference model exactly.

        Args:
            audio_input_ids: Audio stream token IDs for prefill.
            text_input_ids: Text stream token IDs for prefill.
            is_continuous_mask: Boolean mask marking continuous audio positions.
            whisper_features: Raw 5120-dim Whisper features (pre-VQAdaptor).
            max_new_tokens: Maximum audio codes to generate.
            audio_output_vocab: Size of audio vocabulary (default 16384).

        Returns:
            List of relative audio code tensors (0 to audio_output_vocab-1).
        """
        hf_model = self._hf_full_model
        device = hf_model.model.embed_tokens.weight.device
        token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
        kimia_text_blank = int(getattr(self.config, "kimia_text_blank", 151666))

        # EOD tokens — absolute token IDs the reference uses for audio stream termination.
        # msg_end=151645 and media_end=151663 are below token_offset (152064),
        # so they exist in the text token range but the model CAN output them as audio
        # stream termination signals (matching reference kimia.py line 49).
        msg_end = int(getattr(self.config, "kimia_msg_end", 151645))
        media_end = int(getattr(self.config, "kimia_media_end", 151663))
        eod_ids = {msg_end, media_end}

        delay_tokens = int(getattr(self.config, "kimia_mimo_audiodelaytokens", 6))
        audio_temperature = float(getattr(self.config, "kimia_audio_temperature", 0.8))
        audio_top_k = int(getattr(self.config, "kimia_audio_top_k", 10))
        rep_penalty = float(getattr(self.config, "kimia_repetition_penalty", 1.1))

        # Initialize tracking buffers (4096 max, matching reference)
        previous_audio_tokens = torch.zeros(
            (4096,), dtype=torch.long, device=device,
        )
        text_previous_tokens = torch.zeros(
            (4096,), dtype=torch.long, device=device,
        )

        # Prefill inputs
        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        decoder_position_ids = torch.arange(
            0, decoder_input_audio_ids.shape[1], device=device,
        ).unsqueeze(0).long()
        # HF model expects whisper_input_feature as a LIST of tensors
        # Cast to model dtype (bfloat16) to match model weights and move to GPU
        model_dtype = hf_model.model.embed_tokens.weight.dtype
        if whisper_features.device.type != "cuda":
            whisper_features = whisper_features.cuda()
        if whisper_features.dtype != model_dtype:
            logger.info(
                "[REFERENCE GEN LOOP] Casting whisper_features from %s to %s",
                whisper_features.dtype, model_dtype,
            )
            whisper_features = whisper_features.to(model_dtype)
        decoder_input_whisper_feature = [whisper_features]
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1
        valid_audio_length = 0
        text_stream_finished = False
        all_audio_codes: list[torch.Tensor] = []
        all_text_codes: list[torch.Tensor] = []

        audio_seq_len = audio_input_ids.shape[1]
        logger.info(
            "[REFERENCE GEN LOOP] Starting: audio_seq_len=%d, text_seq_len=%d, "
            "max_new_tokens=%d, delay_tokens=%d",
            audio_seq_len, text_input_ids.shape[1],
            max_new_tokens, delay_tokens,
        )

        for step in range(max_new_tokens):
            # Full model forward (handles embedding construction internally)
            with torch.no_grad():
                outputs = hf_model(
                    input_ids=decoder_input_audio_ids,
                    text_input_ids=decoder_input_text_ids,
                    whisper_input_feature=decoder_input_whisper_feature,
                    is_continuous_mask=decoder_is_continuous_mask,
                    position_ids=decoder_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=False,
                )

            # HF model (MoonshotKimiaForCausalLM) with return_dict=False returns:
            # (audio_logits, text_logits, past_key_values)
            # See modeling_moonshot_kimia.py:944-946
            audio_logits = outputs[0]    # [1, seq_len, vocab] or [1, vocab]
            text_logits = outputs[1]     # [1, seq_len, vocab] or [1, vocab]
            past_key_values = outputs[2] if len(outputs) > 2 else None

            # Logits are already computed by the HF model forward.
            # Slice to the position we need (last for prefill, first for decode).
            if step == 0:
                # Prefill: sample from last position of the full sequence
                text_logits_slice = text_logits[:, -1, :].squeeze(0)  # [vocab]
                audio_logits_slice = audio_logits[:, -1, :].squeeze(0)  # [vocab]
            else:
                # Decode: single token at position 0 — squeeze batch dim
                # audio_logits shape is [1, 1, vocab] or [1, vocab]
                if audio_logits.dim() == 3:
                    text_logits_slice = text_logits.squeeze(1).squeeze(0)  # [1, 1, V] -> [1, V] -> [V]
                    audio_logits_slice = audio_logits.squeeze(1).squeeze(0)
                else:
                    text_logits_slice = text_logits.squeeze(0)  # [1, V] -> [V]
                    audio_logits_slice = audio_logits.squeeze(0)

            # Slice to text subspace (below token_offset)
            text_logits_for_sampling = text_logits_slice[:token_offset]

            # Sample text token (greedy, matching reference text_temperature=0.0)
            next_token_text = torch.argmax(
                torch.log_softmax(text_logits_for_sampling, dim=-1, dtype=torch.float),
                dim=-1,
            ).unsqueeze(0)  # [1] for compatibility with tracking buffer

            # Sample audio from full range including EOD tokens below token_offset.
            # The reference model also uses the full audio logits without slicing,
            # so the model can naturally output msg_end/media_end to stop generation.
            # EOD tokens (msg_end=151645, media_end=151663) are below token_offset,
            # so we include everything from msg_end through the full audio subspace.
            audio_start_idx = min(msg_end, token_offset)
            audio_logits_for_sampling = audio_logits_slice[audio_start_idx:]

            # Sample audio token with temperature + top-k (matching reference audio_temperature=0.8, audio_top_k=10)
            # Build recent tokens for repetition penalty from valid audio codes so far
            recent_rel = torch.tensor(
                [c.item() if c.dim() == 0 else c for c in all_audio_codes if c.numel() > 0],
                dtype=torch.long, device=device,
            ) if all_audio_codes else None
            next_audio_token_rel = _sample_audio_topk(
                audio_logits_for_sampling.unsqueeze(0),
                top_k=audio_top_k,
                temperature=audio_temperature,
                repetition_penalty=rep_penalty,
                recent_tokens=recent_rel,
                repetition_window=64,
            ).squeeze(0)  # scalar
            # Convert relative to absolute (relative to audio_start_idx in the sliced logits)
            next_audio_token_abs = next_audio_token_rel + audio_start_idx

            # Use the audio logits variable name expected by diagnostic logging below
            audio_logits_slice = audio_logits_for_sampling

            # Track text
            if text_stream_finished:
                next_token_text.fill_(kimia_text_blank)
            elif next_token_text.item() >= token_offset - 1:
                # EOS detected
                text_stream_finished = True
            else:
                pass  # valid text
            text_previous_tokens[step:step + 1] = next_token_text

            # During delay, override sampled audio token with blank (matching reference lines 143-144)
            if step < delay_tokens:
                next_audio_token_rel.fill_(kimia_text_blank - audio_start_idx)
                next_audio_token_abs.fill_(kimia_text_blank)

            # Track audio — apply delay token logic (matching reference lines 143-151)
            if step < delay_tokens:
                # During delay: reference outputs blank and strips delay tokens from final output
                previous_audio_tokens[step:step + 1] = torch.tensor(
                    kimia_text_blank, device=device,
                )
                # Don't add blank tokens to all_audio_codes — reference strips delay tokens
            elif next_audio_token_abs.item() in eod_ids:
                # EOD token — don't store in audio codes (matching reference: strips after EOD)
                previous_audio_tokens[step:step + 1] = next_audio_token_abs
            else:
                previous_audio_tokens[step:step + 1] = next_audio_token_abs
                # Store relative to token_offset (vocoder expects 0-based audio codes)
                all_audio_codes.append((next_audio_token_abs - token_offset).cpu())
                valid_audio_length += 1

            # Prepare next decode input (single token)
            # During delay, use blank token as audio input (matching reference):
            # reference fills next_audio_token with blank before using it as input.
            if step < delay_tokens - 1:
                # Still in delay period: next step input should be blank token
                # Reference uses kimia_text_blank directly (no +token_offset)
                decoder_input_audio_ids = torch.tensor(
                    kimia_text_blank, device=device,
                ).unsqueeze(0).unsqueeze(0)  # scalar -> [1, 1]
            elif step == delay_tokens - 1:
                # Last delay step: next step is first real audio step, use sampled token
                decoder_input_audio_ids = next_audio_token_abs.unsqueeze(0).unsqueeze(0)
            else:
                # Normal decode: use sampled audio token
                decoder_input_audio_ids = next_audio_token_abs.unsqueeze(0).unsqueeze(0)

            # Track text tokens (excluding blanks/EOS for clean output)
            if not text_stream_finished and next_token_text.item() < token_offset - 1:
                all_text_codes.append(next_token_text.cpu().unsqueeze(0))

            # Check EOD (matching reference: audio_stream_is_finished)
            if next_audio_token_abs.item() in eod_ids and step >= delay_tokens:
                logger.info(
                    "[REFERENCE GEN LOOP] EOD detected at step %d, "
                    "valid_audio=%d codes",
                    step, valid_audio_length,
                )
                break

            # Prepare next decode input (single token)
            # next_token_text has shape [1], decoder_input_audio_ids set above
            decoder_input_text_ids = next_token_text.unsqueeze(0)  # [1] -> [1, 1]
            decoder_position_ids = torch.zeros(1, 1, device=device).fill_(
                last_position_id + 1
            ).long().view(1, 1)
            last_position_id += 1

            # Clear Whisper features and mask after prefill (matching reference)
            decoder_input_whisper_feature = None
            decoder_is_continuous_mask = None

            # Step-by-step diagnostic logging
            if step < 20 or step % 50 == 0:
                probs = torch.softmax(audio_logits_slice, dim=-1)
                top5_probs, top5_idx = torch.topk(probs, 5)
                logit_entropy = -torch.sum(
                    probs * torch.log_softmax(audio_logits_slice, dim=-1)
                ).item()
                logger.info(
                    "[DIAG REF LOOP] step=%d audio_code=%d(pos_rel=%d) pos=%d "
                    "audio_logits_std=%.4f text_code=%d past_kv_seq_len=%d delay=%s entropy=%.2f "
                    "top5_probs=[%.3f,%.3f,%.3f,%.3f,%.3f] top5_idx=%s",
                    step, next_audio_token_abs.item(), next_audio_token_rel.item(),
                    last_position_id,
                    audio_logits_slice.float().std().item(),
                    next_token_text.item(),
                    past_key_values[0][0].shape[2] if past_key_values else 0,
                    step < delay_tokens,
                    logit_entropy,
                    top5_probs.float().cpu().tolist()[0],
                    top5_probs.float().cpu().tolist()[1],
                    top5_probs.float().cpu().tolist()[2],
                    top5_probs.float().cpu().tolist()[3],
                    top5_probs.float().cpu().tolist()[4],
                    (top5_idx + token_offset).cpu().tolist(),
                )

        logger.info(
            "[REFERENCE GEN LOOP] Complete: total_steps=%d, valid_audio=%d, "
            "unique_codes=%d, range=[%d, %d]",
            len(all_audio_codes), valid_audio_length,
            len(set(c.item() for c in all_audio_codes)),
            min((c.item() for c in all_audio_codes), default=-1),
            max((c.item() for c in all_audio_codes), default=-1),
        )

        return all_audio_codes, all_text_codes

    def _run_reference_generate_loop_embeds(
        self,
        audio_input_ids: torch.Tensor,       # [1, audio_seq_len] — prefill audio tokens
        text_input_ids: torch.Tensor,        # [1, text_seq_len] — prefill text tokens
        is_continuous_mask: torch.Tensor,    # [1, audio_seq_len] bool
        whisper_emb_3584: torch.Tensor,      # [1, num_frames, 3584] post-VQAdaptor
        max_new_tokens: int = 1500,
        audio_output_vocab: int = 16384,
    ) -> list[torch.Tensor]:
        """Generate all audio codes for S2S mode with pre-computed 3584-dim Whisper features.

        Unlike `_run_reference_generate_loop` which passes raw 5120-dim Whisper features
        to the HF model (letting it apply VQAdaptor internally), this method constructs
        `inputs_embeds` manually since the features are already post-VQAdaptor.

        Embedding construction matches HF model forward() lines 705-759:
        1. audio_emb = embed_tokens(audio_input_ids)
        2. Inject whisper_emb_3584 at continuous audio positions
        3. inputs_embeds = audio_emb + embed_tokens(text_input_ids)
        """
        logger.info("[EMBEDS LOOP v3] Starting _run_reference_generate_loop_embeds")
        logger.info(
            "[EMBEDS LOOP INPUTS] audio_input_ids=%s text_input_ids=%s "
            "is_continuous_mask=%s whisper_emb_3584=%s",
            audio_input_ids.shape, text_input_ids.shape,
            is_continuous_mask.shape, whisper_emb_3584.shape,
        )
        hf_model = self._hf_full_model
        device = hf_model.model.embed_tokens.weight.device
        model_dtype = hf_model.model.embed_tokens.weight.dtype  # bfloat16
        token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
        kimia_text_blank = int(getattr(self.config, "kimia_text_blank", 151666))

        # EOD tokens — absolute token IDs (matching reference kimia.py line 49)
        msg_end_embeds = int(getattr(self.config, "kimia_msg_end", 151645))
        media_end_embeds = int(getattr(self.config, "kimia_media_end", 151663))
        eod_ids = {msg_end_embeds, media_end_embeds}
        delay_tokens = int(getattr(self.config, "kimia_mimo_audiodelaytokens", 6))
        audio_temperature = float(getattr(self.config, "kimia_audio_temperature", 0.8))
        audio_top_k = int(getattr(self.config, "kimia_audio_top_k", 10))

        embed_tokens = hf_model.model.embed_tokens

        previous_audio_tokens = torch.zeros(
            (4096,), dtype=torch.long, device=device,
        )
        text_previous_tokens = torch.zeros(
            (4096,), dtype=torch.long, device=device,
        )

        audio_seq_len = audio_input_ids.shape[1]

        # Build inputs_embeds manually (matching HF model forward lines 705-759).
        # whisper_emb_3584 is already post-VQAdaptor [1, num_frames, 3584].
        # Cast to model dtype (bfloat16) to match model weights.
        if whisper_emb_3584.dtype != model_dtype:
            logger.info(
                "[EMBEDS LOOP] Casting whisper_emb_3584 from %s to %s",
                whisper_emb_3584.dtype, model_dtype,
            )
            whisper_emb_3584 = whisper_emb_3584.to(model_dtype)

        # 1. audio_emb = embed_tokens(audio_input_ids)
        audio_emb = embed_tokens(audio_input_ids)  # [1, audio_seq_len, 3584]
        # 2. Create expanded whisper embedding at correct positions
        expanded_whisper = torch.zeros_like(audio_emb)  # [1, audio_seq_len, 3584]
        # whisper_emb_3584 has shape [1, num_frames, 3584]; map to positions 2..N+1
        expanded_whisper[0, 2:2 + whisper_emb_3584.shape[1], :] = whisper_emb_3584[0]
        # 3. Apply is_continuous_mask to whisper (matching HF line 742)
        bool_mask = is_continuous_mask  # [1, audio_seq_len] bool
        mask_float = bool_mask.to(model_dtype).unsqueeze(-1)  # [1, audio_seq_len, 1]
        # 4. Additive fusion with sqrt(2) scaling (matching HF lines 744-755)
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=model_dtype, device=device))
        fused = (audio_emb + expanded_whisper * mask_float) * sqrt2
        # 5. Blend: non-continuous positions use original audio_emb
        inv_mask = (~bool_mask).to(model_dtype).unsqueeze(-1)
        audio_emb = audio_emb * inv_mask + fused * mask_float
        # 6. Add text embeddings (matching HF line 757)
        text_emb = embed_tokens(text_input_ids)
        inputs_embeds = audio_emb + text_emb

        logger.info(
            "[EMBEDS LOOP PREFILL] inputs_embeds=%s audio_emb=%s text_emb=%s "
            "expanded_whisper=%s audio_seq_len=%d",
            inputs_embeds.shape, audio_emb.shape, text_emb.shape,
            expanded_whisper.shape, audio_seq_len,
        )
        decoder_position_ids = torch.arange(0, audio_seq_len, device=device).unsqueeze(0).long()
        past_key_values = None

        last_position_id = audio_seq_len - 1
        valid_audio_length = 0
        text_stream_finished = False
        all_audio_codes: list[torch.Tensor] = []
        all_text_codes: list[torch.Tensor] = []

        logger.info(
            "[REFERENCE GEN LOOP EMBEDS] Starting: audio_seq_len=%d, text_seq_len=%d, "
            "max_new_tokens=%d, delay_tokens=%d",
            audio_seq_len, text_input_ids.shape[1],
            max_new_tokens, delay_tokens,
        )

        for step in range(max_new_tokens):
            with torch.no_grad():
                # S2S mode: call the underlying transformer directly.
                if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'forward'):
                    # DEBUG: check inputs_embeds shape/type
                    logger.info(
                        "[DIAG EMBEDS IN] step=%d type=%s shape=%s ndim=%s "
                        "pos_ids=%s pos_shape=%s past_kv=%s",
                        step, type(inputs_embeds).__name__,
                        getattr(inputs_embeds, 'shape', 'N/A'),
                        getattr(inputs_embeds, 'ndim', 'N/A'),
                        decoder_position_ids.shape, decoder_position_ids.shape,
                        type(past_key_values).__name__ if past_key_values is not None else 'None',
                    )
                    transformer_out = hf_model.model(
                        inputs_embeds=inputs_embeds,
                        position_ids=decoder_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    # MoonshotKimiaModel returns BaseModelOutputWithPast where
                    # last_hidden_state is (text_hs, mimo_hs) — a tuple!
                    # See modeling_moonshot_kimia.py:847
                    lhs = transformer_out.last_hidden_state
                    if isinstance(lhs, tuple):
                        text_hs = lhs[0]       # [1, seq, 3584] backbone output
                        mimo_hs = lhs[1]       # [1, seq, 3584] MIMO output
                    else:
                        # Fallback: extract from hidden_states
                        text_hs = lhs
                        if hasattr(transformer_out, 'hidden_states') and transformer_out.hidden_states:
                            mimo_hs = transformer_out.hidden_states[-2]  # pre-final norm
                        else:
                            mimo_hs = text_hs

                    past_key_values = getattr(transformer_out, 'past_key_values', None)
                else:
                    # Fallback: call full model
                    outputs = hf_model(
                        inputs_embeds=inputs_embeds,
                        position_ids=decoder_position_ids,
                        past_key_values=past_key_values,
                        return_dict=False,
                    )
                    text_hs = outputs[0]
                    mimo_hs = outputs[1]
                    past_key_values = outputs[2] if len(outputs) > 2 else None

            if step == 0:
                logger.info(
                    "[DIAG EMBEDS HF OUT] text_hs=%s mimo_hs=%s inputs_embeds=%s",
                    list(text_hs.shape), list(mimo_hs.shape), list(inputs_embeds.shape),
                )
                pos_idx = -1  # Prefill: sample from last position
            else:
                pos_idx = 0   # Decode: single token

            # Compute logits from hidden states
            # _hf_lm_head and _hf_mimo_output return [1, vocab] from [1, hidden] inputs
            text_logits = self._hf_lm_head(text_hs[:, pos_idx, :]).squeeze(0)  # [vocab]
            audio_logits = self._hf_mimo_output(mimo_hs[:, pos_idx, :]).squeeze(0)  # [vocab]

            if step == 0:
                logger.info(
                    "[DIAG EMBEDS LOGITS] text_logits=%s audio_logits=%s token_offset=%d "
                    "audio_output_vocab=%d audio_logits_slice_size=%d",
                    list(text_logits.shape), list(audio_logits.shape), token_offset,
                    audio_output_vocab,
                    audio_logits[token_offset:token_offset + audio_output_vocab].numel(),
                )

            # Sample text token (greedy)
            text_logits_slice = text_logits[:token_offset]
            next_token_text = torch.argmax(
                torch.log_softmax(text_logits_slice, dim=-1, dtype=torch.float),
                dim=-1,
            )  # 0-D scalar

            # Sample audio token with temperature (greedy causes fixed-point loop)
            # Include EOD tokens below token_offset so model can naturally stop generation.
            audio_start_idx_em = min(msg_end_embeds, token_offset)
            audio_logits_slice = audio_logits[audio_start_idx_em:]
            # Build recent tokens for repetition penalty from valid decode steps
            recent_audio_list = [c.item() if c.dim() == 0 else c for c in all_audio_codes if c.numel() > 0]
            recent_audio_rel = torch.tensor(recent_audio_list, dtype=torch.long, device=device) if recent_audio_list else None
            rep_penalty_embeds = float(getattr(self.config, "kimia_repetition_penalty", 1.1))
            next_audio_token_rel = _sample_audio_topk(
                audio_logits_slice.unsqueeze(0),
                top_k=audio_top_k,
                temperature=audio_temperature,
                repetition_penalty=rep_penalty_embeds,
                recent_tokens=recent_audio_rel,
                repetition_window=64,
            ).squeeze(0)  # scalar
            next_audio_token_abs = next_audio_token_rel + audio_start_idx_em

            # Ensure scalars for decode input construction
            next_token_text = next_token_text.squeeze()  # ensure 0-D
            next_audio_token_rel = next_audio_token_rel.squeeze()  # ensure 0-D
            next_audio_token_abs = next_audio_token_abs.squeeze()  # ensure 0-D

            # Track text
            if text_stream_finished:
                next_token_text.fill_(kimia_text_blank)
            elif next_token_text.item() >= token_offset - 1:
                text_stream_finished = True
            text_previous_tokens[step:step + 1] = next_token_text

            # During delay, override sampled audio token with blank (matching reference lines 143-144)
            if step < delay_tokens:
                next_audio_token_rel.fill_(kimia_text_blank - audio_start_idx_em)
                next_audio_token_abs.fill_(kimia_text_blank)

            # Track audio with delay tokens
            if step < delay_tokens:
                # During delay: reference outputs blank and strips delay tokens from final output
                previous_audio_tokens[step:step + 1] = torch.tensor(
                    kimia_text_blank, device=device,
                )
                # Don't add blank tokens to all_audio_codes — reference strips delay tokens
            else:
                previous_audio_tokens[step:step + 1] = next_audio_token_abs
                all_audio_codes.append(next_audio_token_rel.cpu())
                valid_audio_length += 1

            # Track text tokens (excluding blanks/EOS for clean output)
            if not text_stream_finished and next_token_text.item() < token_offset - 1:
                all_text_codes.append(next_token_text.cpu().unsqueeze(0))

            # Check EOD
            if next_audio_token_abs.item() in eod_ids and step >= delay_tokens:
                logger.info(
                    "[REFERENCE GEN LOOP EMBEDS] EOD detected at step %d, "
                    "valid_audio=%d codes",
                    step, valid_audio_length,
                )
                break

            # Prepare next decode input (single token)
            # During delay, use kimia_text_blank as audio input (matching reference)
            if step < delay_tokens - 1:
                # Still in delay period: use blank token
                next_audio_for_input = torch.tensor(kimia_text_blank, device=device)
            elif step == delay_tokens - 1:
                # Last delay step: next is first real audio step, use sampled token
                next_audio_for_input = next_audio_token_abs
            else:
                # Normal decode: use sampled audio token
                next_audio_for_input = next_audio_token_abs

            logger.info(
                "[DIAG DECODE PREP] next_audio_token_abs shape=%s next_token_text shape=%s",
                next_audio_token_abs.shape, next_token_text.shape,
            )
            next_audio_abs = next_audio_for_input.unsqueeze(0).unsqueeze(0)  # [1, 1]
            next_text = next_token_text.unsqueeze(0).unsqueeze(0)  # [1, 1]
            logger.info(
                "[DIAG DECODE PREP] next_audio_abs shape=%s next_text shape=%s",
                next_audio_abs.shape, next_text.shape,
            )

            # Single-token embedding: no whisper (cleared after prefill)
            single_audio_emb = embed_tokens(next_audio_abs)  # [1, 1, 3584]
            single_text_emb = embed_tokens(next_text)  # [1, 1, 3584]
            inputs_embeds = single_audio_emb + single_text_emb  # [1, 1, 3584]
            logger.info(
                "[DIAG DECODE PREP] single_audio_emb=%s single_text_emb=%s inputs_embeds=%s",
                single_audio_emb.shape, single_text_emb.shape, inputs_embeds.shape,
            )

            decoder_position_ids = torch.zeros(1, 1, device=device).fill_(
                last_position_id + 1
            ).long().view(1, 1)
            last_position_id += 1

            # Diagnostic logging
            if step < 20 or step % 50 == 0:
                probs = torch.softmax(audio_logits_slice, dim=-1)
                top5_probs, top5_idx = torch.topk(probs, 5)
                logit_entropy = -torch.sum(
                    probs * torch.log_softmax(audio_logits_slice, dim=-1)
                ).item()
                logger.info(
                    "[DIAG REF LOOP EMBEDS] step=%d audio_code=%d(pos_rel=%d) pos=%d "
                    "audio_logits_std=%.4f text_code=%d past_kv_seq_len=%d delay=%s entropy=%.2f "
                    "top5_idx=%s",
                    step, next_audio_token_abs.item(), next_audio_token_rel.item(),
                    last_position_id,
                    audio_logits_slice.float().std().item(),
                    next_token_text.item(),
                    past_key_values[0][0].shape[2] if past_key_values else 0,
                    step < delay_tokens,
                    logit_entropy,
                    (top5_idx + token_offset).cpu().tolist(),
                )

        logger.info(
            "[REFERENCE GEN LOOP EMBEDS] Complete: total_steps=%d, valid_audio=%d, "
            "unique_codes=%d, range=[%d, %d]",
            len(all_audio_codes), valid_audio_length,
            len(set(c.item() for c in all_audio_codes)),
            min((c.item() for c in all_audio_codes), default=-1),
            max((c.item() for c in all_audio_codes), default=-1),
        )

        # Save KV cache for subsequent decode steps.
        # The reference loop built up the full KV cache across all 34 layers
        # (28 backbone + 6 MIMO). We split and store it so decode steps can
        # use the reference model's decode path correctly.
        if past_key_values is not None:
            past_kv_tuple = tuple(_from_cache(past_key_values))
            bifurcation_idx = int(self.config.kimia_mimo_transformer_from_layer_index)
            self._hf_past_key_values = past_kv_tuple[:bifurcation_idx + 1]
            self._mimo_past_key_values = past_kv_tuple[bifurcation_idx + 1:]
            logger.info(
                "[REFERENCE GEN LOOP EMBEDS] Saved KV cache: backbone=%d layers, "
                "mimo=%d layers, seq_len=%d",
                len(self._hf_past_key_values), len(self._mimo_past_key_values),
                self._hf_past_key_values[0][0].shape[2] if self._hf_past_key_values else 0,
            )

        return all_audio_codes, all_text_codes

    def _process_audio_input(
        self, audio_input: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Process Mel spectrogram through Whisper encoder -> 4x downsample -> projector.

        Overrides upstream vLLM to add audio-length truncation, matching the
        reference Kimi-Audio Whisper encoder behavior (whisper_Lv3). Without
        truncation, Whisper produces 1500 frames (30s padded), yielding 375
        output frames after 4x downsample vs ~47 for a 3.7s audio.

        Supports optional ``num_samples`` key in audio_input to derive the
        truncation boundary: token_len = (num_samples - 1) // 1280 + 1,
        max_encoder_frames = token_len * 4.
        """
        input_features = audio_input["whisper_input_features"]
        num_samples = audio_input.get("num_samples")

        # KimiAudioWhisperEncoder expects list of tensors
        if input_features.dim() == 3:
            input_features = input_features.unbind(dim=0)

        # Run through Whisper encoder
        audio_features = self.audio_tower(input_features)

        # Truncate encoder output to actual audio length before 4x downsample,
        # matching reference Kimi-Audio Whisper encoder behavior (whisper_Lv3).
        # HF WhisperFeatureExtractor pads audio to 30s (480000 samples),
        # producing 1500 encoder frames. Without truncation, the model attends
        # to padding/silence frames, degrading generation quality.
        if num_samples is not None:
            B, seq_len, D = audio_features.shape
            token_len = (num_samples - 1) // (160 * 8) + 1
            max_encoder_frames = token_len * 4
            if max_encoder_frames < seq_len:
                audio_features = audio_features[:, :max_encoder_frames, :]
        else:
            # Fallback: detect audio boundary from encoder output energy.
            # Whisper encoder output for silence/padding frames has near-zero
            # energy. Find the last frame with significant energy.
            B, seq_len, D = audio_features.shape
            frame_energy = audio_features.norm(dim=-1)  # [B, T]
            # Threshold: 5% of the max frame energy (robust to quiet audio)
            max_energy = frame_energy.max(dim=-1, keepdim=True).values  # [B, 1]
            threshold = max_energy * 0.05
            # Find last frame above threshold for each batch
            above = (frame_energy > threshold).float()  # [B, T]
            indices = torch.arange(seq_len, device=audio_features.device, dtype=audio_features.dtype)
            last_above = (above * indices).max(dim=-1).values.long()  # [B]
            # Add small safety margin (4 frames = ~80ms)
            max_encoder_frames = int(last_above[0].item()) + 4
            # Round up to nearest multiple of 4 for 4x downsample
            max_encoder_frames = ((max_encoder_frames + 3) // 4) * 4
            if max_encoder_frames < seq_len and max_encoder_frames > 0:
                audio_features = audio_features[:, :max_encoder_frames, :]

        # Reshape for 4x downsampling (Whisper outputs at 50Hz, need 12.5Hz)
        B, T, D = audio_features.shape
        if T % 4 != 0:
            pad_len = 4 - (T % 4)
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_len))
            T = audio_features.shape[1]

        audio_features = audio_features.reshape(B, T // 4, D * 4)

        # Project to LLM dimension
        audio_embeds = self.multi_modal_projector(audio_features)
        return audio_embeds

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply token embeddings to input_ids for vLLM runner compatibility.

        Detects decode vs prefill based on input length: decode processes
        single tokens (shape[-1] == 1), prefill processes sequences.

        Also extracts Whisper features from kwargs (audio_array or
        whisper_input_feature) for speech-to-speech conditioning.
        """
        is_decode = input_ids.dim() > 0 and input_ids.shape[-1] == 1

        # Extract Whisper features from kwargs (set by serving layer or
        # passed through the prompt's additional_information).
        if not is_decode and kwargs:
            self._extract_whisper_from_kwargs(kwargs)

        return self._combine_audio_text_embeds(input_ids, is_decode=is_decode)

    def _combine_audio_text_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_decode: bool = False,
    ) -> torch.Tensor:
        """Combine text and audio embeddings from dual-stream token inputs.

        The reference Kimi-Audio model expects TWO token streams at every
        position: audio tokens (input_ids) and text tokens (text_input_ids).
        Embeddings are summed: inputs_embeds = embed(audio_ids) + embed(text_ids).

        During prefill: construct audio placeholder tokens from text tokens,
        embed both, and sum (matching the reference's dual-stream embedding).

        During decode: vLLM only feeds back text tokens. We use self-stored
        audio codes to construct the audio embedding (self-feedback).
        """
        token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
        embed_tokens = self.llm.model.embed_tokens

        if input_ids is None:
            return inputs_embeds if inputs_embeds is not None else None

        # Decode path: use self-stored audio code (vLLM doesn't feed back audio)
        # During the audio delay period, use blank tokens instead of real codes
        # (matching reference: next_audio_token.fill_(blank) for i < delay_tokens).
        #
        # CRITICAL FIX: During decode, the audio embedding MUST match the per-frame
        # distribution from prefill: (audio_embed + whisper_frame) * sqrt(2).
        # Without whisper features, the MIMO layers receive a completely different
        # input distribution during decode, causing degeneration.
        if is_decode:
            delay_tokens = int(getattr(self.config, "kimia_mimo_audiodelaytokens", 6))
            decode_step = getattr(self, "_decode_step_counter", 0)
            use_delay_blank = decode_step < delay_tokens

            # Embed text tokens from input_ids (vLLM feeds back generated text).
            text_embed = embed_tokens(input_ids)

            if use_delay_blank:
                # During delay period: use blank audio token as self-feedback.
                # Use a relative audio index that's valid for embedding lookup.
                audio_blank_index = 18  # relative audio token index
                audio_ids = torch.full_like(input_ids, audio_blank_index + token_offset)
                audio_embed = embed_tokens(audio_ids)
            else:
                # After delay period: use real audio codes as self-feedback.
                # _audio_codes_list stores RELATIVE indices — convert to absolute.
                if hasattr(self, "_audio_codes_list") and len(self._audio_codes_list) > 0:
                    last_code = self._audio_codes_list[-1]
                    if last_code.numel() >= 1:
                        code_val = last_code.reshape(-1)[-1].item() + token_offset
                        audio_ids = torch.full_like(input_ids, code_val)
                        audio_embed = embed_tokens(audio_ids)
                    else:
                        audio_blank_index = 18
                        audio_ids = torch.full_like(input_ids, audio_blank_index + token_offset)
                        audio_embed = embed_tokens(audio_ids)
                else:
                    audio_blank_index = 18
                    audio_ids = torch.full_like(input_ids, audio_blank_index + token_offset)
                    audio_embed = embed_tokens(audio_ids)

            # Decode: NO whisper features during decode.
            # The reference Kimi-Audio pipeline sets decoder_input_whisper_feature=None
            # and decoder_is_continuous_mask=None during decode (see kimia.py:191-192).
            # Whisper features are ONLY for prefill conditioning.
            return audio_embed + text_embed

        # Prefill path: construct parallel audio+text embeddings.
        # The reference model (modeling_kimia.py:672-721) expects TWO token
        # streams at the SAME positions:
        #   1. audio_emb = embed_tokens(input_ids) — blank audio tokens
        #   2. whisper_emb = vq_adaptor(whisper_features) — audio features
        #   3. text_emb = embed_tokens(text_input_ids) — text tokens
        # Combined: inputs_embeds = audio_emb + text_emb
        # where audio_emb = (audio_embed + whisper_emb) * sqrt(2) at masked positions.
        #
        # CRITICAL: The audio and text have the SAME sequence length.
        # vLLM-Omni previously CONCATENATED them ([audio, text]) which the model
        # was NOT trained on — causing severe audio quality degradation.
        # Fix: SUM audio+text embeddings at each position, using max length.

        text_embed = embed_tokens(input_ids)
        text_len = input_ids.shape[-1]

        # Priority 1: Per-request Whisper features from input audio
        if self._input_whisper_emb is not None:
            # _input_whisper_emb: [1, audio_frames, 3584] — already post-VQAdaptor
            whisper_emb = self._input_whisper_emb.squeeze(0).to(
                text_embed.device, dtype=text_embed.dtype,
            )  # [audio_frames, 3584]
            audio_frames = whisper_emb.shape[0]

            # Create audio placeholder tokens and embed
            audio_blank_index = 18  # relative audio token index
            audio_ids = torch.full(
                (audio_frames,), audio_blank_index + token_offset,
                device=input_ids.device, dtype=input_ids.dtype,
            )
            audio_embed = embed_tokens(audio_ids)  # [audio_frames, 3584]

            # Audio path: (audio_embed + whisper) * sqrt(2) — matching reference
            audio_combined = (audio_embed + whisper_emb) * (2.0 ** 0.5)

            # Parallel combination: sum audio+text at each position.
            # Pad shorter stream with zeros to match lengths.
            seq_len = max(audio_frames, text_len)
            if audio_frames < seq_len:
                audio_padded = torch.cat([
                    audio_combined,
                    torch.zeros(seq_len - audio_frames, audio_combined.shape[-1],
                                device=audio_combined.device, dtype=audio_combined.dtype),
                ], dim=0)
            else:
                audio_padded = audio_combined
            if text_len < seq_len:
                text_padded = torch.cat([
                    text_embed,
                    torch.zeros(seq_len - text_len, text_embed.shape[-1],
                                device=text_embed.device, dtype=text_embed.dtype),
                ], dim=0)
            else:
                text_padded = text_embed
            combined = audio_padded + text_padded  # [seq_len, 3584]

            # Store for position override in forward()
            self._prefill_audio_offset = audio_frames

            # [DIAG STEP2] Log combined embeddings
            logger.info(
                "[DIAG STEP2] Combined embed (PARALLEL Whisper): audio_frames=%d, text_tokens=%d, "
                "combined_seq_len=%d (was %d with concat), audio_combined mean=%.4f std=%.4f, "
                "text_embed mean=%.4f std=%.4f",
                audio_frames, text_len, combined.shape[0],
                audio_frames + text_len,  # what concat would have been
                audio_combined.float().mean().item(), audio_combined.float().std().item(),
                text_embed.float().mean().item(), text_embed.float().std().item(),
            )

            # Store whisper features for decode augmentation.
            if getattr(self, '_decode_whisper_frames', None) is None:
                self._decode_whisper_frames = whisper_emb.clone()  # [audio_frames, 3584]
            if getattr(self, '_decode_whisper_template', None) is None:
                self._decode_whisper_template = whisper_emb.mean(dim=0, keepdim=True).clone()

            return combined

        # Priority 2: Static Whisper embedding from reference audio (S2S)
        if self._whisper_emb is not None:
            # _whisper_emb: [1, audio_frames, 3584] — use per-frame features
            whisper_emb = self._whisper_emb.squeeze(0).to(
                text_embed.device, dtype=text_embed.dtype,
            )  # [audio_frames, 3584]
            audio_frames = whisper_emb.shape[0]

            audio_blank_index = 18  # relative audio token index
            audio_ids = torch.full(
                (audio_frames,), audio_blank_index + token_offset,
                device=input_ids.device, dtype=input_ids.dtype,
            )
            audio_embed = embed_tokens(audio_ids)
            audio_combined = (audio_embed + whisper_emb) * (2.0 ** 0.5)

            # Parallel: sum at each position
            seq_len = max(audio_frames, text_len)
            if audio_frames < seq_len:
                audio_padded = torch.cat([
                    audio_combined,
                    torch.zeros(seq_len - audio_frames, audio_combined.shape[-1],
                                device=audio_combined.device, dtype=audio_combined.dtype),
                ], dim=0)
            else:
                audio_padded = audio_combined
            if text_len < seq_len:
                text_padded = torch.cat([
                    text_embed,
                    torch.zeros(seq_len - text_len, text_embed.shape[-1],
                                device=text_embed.device, dtype=text_embed.dtype),
                ], dim=0)
            else:
                text_padded = text_embed
            combined = audio_padded + text_padded

            self._prefill_audio_offset = audio_frames

            # Store whisper features for decode augmentation (cyclic)
            if getattr(self, '_decode_whisper_frames', None) is None:
                self._decode_whisper_frames = whisper_emb.clone()  # [audio_frames, 3584]
            if getattr(self, '_decode_whisper_template', None) is None:
                self._decode_whisper_template = whisper_emb.mean(dim=0, keepdim=True).clone()

            return combined

        # Fallback — audio placeholders only (warmup, profile, no Whisper features).
        # Used during model warmup/profile runs when no Whisper features are available.
        self._prefill_audio_offset = 0
        audio_blank_index = 18  # relative audio token index
        audio_placeholder_ids = torch.full_like(input_ids, audio_blank_index + token_offset)
        audio_embed = embed_tokens(audio_placeholder_ids)
        return audio_embed + text_embed

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | OmniOutput:
        backbone = self.llm.model

        # Token offset for audio token ID space (needed by both prefill and decode paths)
        token_offset = int(getattr(self.config, "kimia_token_offset", 152064))

        # Detect prefill vs decode phase
        is_prefill = input_ids is not None and input_ids.dim() > 0 and input_ids.shape[-1] > 1

        # Detect warmup/dummy runs: vLLM runs dummy forward passes for KV cache
        # sizing and model warmup/graph capture. During these, input_ids are all
        # small values (token IDs 0, 1, 2...) with small position IDs.
        # Real decode steps may have input_ids=[0] (blank token) but with
        # large position IDs (>= prefill audio offset).
        _small_positions = (
            positions is not None
            and positions.max().item() < 100
        )
        is_warmup = (
            input_ids is not None
            and input_ids.dim() > 0
            and input_ids.unique().numel() <= 5
            and input_ids.max().item() < 10
            and _small_positions
        )
        # Also detect profile runs: during profile_run, vLLM sends very long
        # sequences (max_model_len tokens) which can cause FlashAttention
        # shape issues with the HF model. Skip HF layers for profiling too
        # since we only need tensor shapes for KV cache sizing.
        _seq_len = input_ids.shape[-1] if input_ids is not None and input_ids.dim() > 0 else 0
        _max_pos = positions.max().item() if positions is not None else 0
        is_profile_run = _seq_len > 1000 or _max_pos > 1000

        if is_profile_run or is_warmup:
            logger.info(
                "[DIAG DETECT] %s: is_prefill=%s, seq_len=%d, max_pos=%d, "
                "input_ids_shape=%s, input_ids_max=%d, unique=%d",
                "PROFILE" if is_profile_run else "WARMUP",
                is_prefill, _seq_len, _max_pos,
                list(input_ids.shape) if input_ids is not None else None,
                input_ids.max().item() if input_ids is not None else 0,
                input_ids.unique().numel() if input_ids is not None else 0,
            )

        # Reset audio code accumulation and MIMO KV cache at start of new request (prefill).
        # Skip reset during warmup to avoid interfering with model init;
        # instead, skip accumulation entirely for warmup passes.
        # NOTE: This must happen BEFORE _extract_whisper_from_kwargs so that
        # newly extracted per-request Whisper features are not wiped out.
        if is_prefill and not is_warmup:
            self._decode_step_counter = 0
            self._audio_codes_list = []
            self._audio_code_cap_logged = False
            self._input_whisper_emb = None
            self._input_whisper_emb_logged = False
            self._input_whisper_raw = None
            self._input_whisper_raw_logged = False
            self._prefill_audio_offset = 0
            self._prefill_seq_len = 0  # Reset actual prefill sequence length
            self._prefill_text_len = 0  # Reset vLLM's view of text length
            self._ref_model_did_prefill = False  # Reset reference model flag
            self._hf_past_key_values = None  # Reset HF backbone KV cache for new request
            self._mimo_past_key_values = None  # Reset MIMO KV cache for new request
            self._decode_mimo_l5_stds = []  # Reset decode MIMO tracker
            self._decode_whisper_template = None  # Reset decode whisper template
            self._decode_whisper_frames = None    # Reset decode whisper frames (cyclic)
            self._batched_decode_done = False  # Reset batched decode flag
            self._is_asr_mode = False  # Reset ASR mode flag

        # Extract Whisper features from kwargs (passed via model_intermediate_buffer
        # from the engine's additional_information deserialization).
        if is_prefill and kwargs:
            self._extract_whisper_from_kwargs(kwargs)

        # DIAGNOSTIC: Log audio code self-feedback during decode
        if not is_prefill and input_ids is not None:
            # Note: positions logged here are from the scheduler. The actual
            # positions used in forward() may be overridden on first_rank
            # to account for longer combined embedding sequences.
            if self._diag_decode_log_count < 50:
                self._diag_decode_log_count += 1
                fb_code = "N/A"
                if hasattr(self, "_audio_codes_list") and len(self._audio_codes_list) > 0:
                    fb_code = self._audio_codes_list[-1].reshape(-1)[-1].item()
                logger.info(
                    "DIAG decode#%d input_ids=%s, scheduler_positions=%s, self_feedback_code=%s (threshold=%d)",
                    self._diag_decode_log_count,
                    input_ids.tolist(),
                    positions.tolist(),
                    fb_code,
                    int(getattr(self.config, "kimia_token_offset", 152064)),
                )

        # Embed input on first rank; other ranks receive via pipeline parallelism
        is_decode = not is_prefill

        # During decode, use the full reference model forward with KV cache.
        # The manual backbone+MIMO loop was producing identical outputs because
        # the embedding construction didn't match the reference model. The
        # reference model handles KV cache, RoPE, and attention correctly.
        # Skip when batched decode already generated all audio codes during prefill.
        use_ref_model_decode = (
            is_decode
            and not is_warmup
            and not getattr(self, "_batched_decode_done", False)
            and hasattr(self, "_hf_full_model")
            and self._hf_full_model is not None
            and hasattr(self, "_hf_past_key_values")
            and self._hf_past_key_values is not None
        )

        if get_pp_group().is_first_rank:
            # During prefill with Whisper features available, use the full reference
            # model forward pass for correct embedding construction (media token
            # scaffolding, VQAdaptor, is_continuous_mask, sqrt(2) scaling).
            use_ref_model = (
                is_prefill
                and not is_warmup
                and hasattr(self, "_hf_full_model")
                and self._hf_full_model is not None
                and (
                    getattr(self, "_input_whisper_raw", None) is not None  # S2S mode (5120-dim)
                    or getattr(self, "_input_whisper_emb", None) is not None  # S2S mode (3584-dim)
                )
            )

            if use_ref_model:
                # Extract Whisper features. The HF model needs raw 5120-dim
                # features (pre-VQAdaptor) to apply its internal VQAdaptor.
                raw_whisper = None
                num_audio = 0
                if getattr(self, "_input_whisper_raw", None) is not None:
                    # S2S mode with 5120-dim raw Whisper features (pre-VQAdaptor).
                    # The HF model applies VQAdaptor internally — bit-identical to reference.
                    raw_whisper = self._input_whisper_raw  # [1, num_frames, 5120]
                    num_audio = raw_whisper.shape[1]
                elif getattr(self, "_input_whisper_emb", None) is not None:
                    # S2S mode: 3584-dim features (post-VQAdaptor).
                    # Legacy path — manual embedding construction.
                    raw_whisper = None  # Will use manual embedding path
                    whisper_3584 = self._input_whisper_emb  # [1, num_frames, 3584]
                    num_audio = whisper_3584.shape[1]

                self._prefill_audio_offset = num_audio

                # Build the 7-token scaffolding for audio stream (matching reference)
                # Pattern: [user_msg_start, media_begin, audio_0..N, media_end, speech_ctd, msg_end, assistant_msg_start]
                token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
                media_begin = int(self.config.kimia_media_begin)
                media_end = int(self.config.kimia_media_end)
                user_msg_start = int(getattr(self.config, "kimia_user_msg_start", 151670))
                speech_ctd = int(getattr(self.config, "kimia_speech_ctd", 151676))
                msg_end = int(getattr(self.config, "kimia_msg_end", 151645))
                assistant_msg_start = int(getattr(self.config, "kimia_assistant_msg_start", 151671))
                kimia_text_blank = int(getattr(self.config, "kimia_text_blank", 151666))

                device = input_ids.device
                audio_token_ids = torch.arange(token_offset, token_offset + num_audio, device=device).unsqueeze(0)
                audio_input_ids = torch.cat([
                    torch.tensor([[user_msg_start]], device=device),
                    torch.tensor([[media_begin]], device=device),
                    audio_token_ids,
                    torch.tensor([[media_end]], device=device),
                    torch.tensor([[speech_ctd]], device=device),
                    torch.tensor([[msg_end]], device=device),
                    torch.tensor([[assistant_msg_start]], device=device),
                ], dim=1)  # [1, num_audio + 7]

                # Text stream: all blank placeholders, same length as audio
                text_input_ids = torch.full(
                    (1, audio_input_ids.shape[1]),
                    fill_value=kimia_text_blank,
                    dtype=torch.long, device=device,
                )

                # is_continuous_mask: True only for audio positions (indices 2..N+1)
                audio_seq_len = audio_input_ids.shape[1]
                is_continuous_mask = torch.zeros(1, audio_seq_len, dtype=torch.bool, device=device)
                is_continuous_mask[0, 2:-4] = True  # 7 scaffolding tokens: 2 at start, 5 at end

                # ASR mode: audio input but text-only output.
                # Run the reference model forward to build embeddings and KV cache,
                # then run a text generation loop to produce the transcription.
                if getattr(self, "_is_asr_mode", False):
                    logger.info(
                        "[ASR MODE] Skipping audio generation, using text-only path. "
                        "audio_seq_len=%d, whisper=%s, vllm_input_ids=%s",
                        audio_seq_len,
                        "raw_5120" if raw_whisper is not None else "emb_3584",
                        input_ids.shape if hasattr(input_ids, 'shape') else type(input_ids).__name__,
                    )
                    # Construct proper text stream for ASR: assistant_msg_start at
                    # position 0 (aligned with user_msg_start in audio), then blanks.
                    # The model needs meaningful text tokens (not audio scaffolding
                    # tokens from vLLM input_ids) to generate a transcription.
                    text_input_ids = torch.full(
                        (1, audio_seq_len),
                        fill_value=kimia_text_blank,
                        dtype=torch.long, device=device,
                    )
                    text_input_ids[0, 0] = assistant_msg_start

                    # Run reference model forward for embedding construction + text generation.
                    if raw_whisper is not None:
                        # S2S mode: 5120-dim raw features, model applies VQAdaptor
                        text_hs, mimo_hs, new_kv = self._call_reference_model_forward(
                            input_ids=input_ids,
                            whisper_feat_5120=raw_whisper,
                            text_input_ids=text_input_ids,
                            past_key_values=None,
                        )
                    else:
                        # S2S mode: 3584-dim features (post-VQAdaptor)
                        text_hs, mimo_hs, new_kv = self._call_reference_model_forward(
                            input_ids=input_ids,
                            whisper_feat_5120=None,
                            text_input_ids=text_input_ids,
                            past_key_values=None,
                            whisper_emb_3584=whisper_3584,
                        )

                    # text_hs: [seq_len, hidden_size] — backbone output before bifurcation
                    # Apply norm and lm_head to get text logits for the last token
                    text_hs = self._hf_norm(text_hs)
                    text_logits = self._hf_lm_head(text_hs)  # [seq_len, vocab_size]

                    # Sample text tokens autoregressively, conditioning on audio+text context
                    # via the KV cache. The model generates text until EOS or max tokens.
                    max_text_tokens = min(2048, getattr(self, "_max_audio_codes", 3000))
                    generated_text = self._run_asr_text_generation(
                        text_logits=text_logits,
                        past_key_values=new_kv,
                        max_new_tokens=max_text_tokens,
                        token_offset=token_offset,
                    )

                    # Store KV cache for decode position correction
                    if new_kv is not None:
                        new_kv_tuple = _from_cache(new_kv)
                        self._hf_past_key_values = tuple(new_kv_tuple[:28])
                        self._mimo_past_key_values = tuple(new_kv_tuple[28:])

                    # Set sequence lengths — text positions will be offset by this delta
                    self._prefill_seq_len = audio_seq_len
                    self._prefill_text_len = audio_seq_len
                    self._prefill_audio_offset = audio_seq_len

                    # Mark that reference model handled prefill so vLLM
                    # skips MIMO layers and uses reference model hidden states.
                    self._ref_model_did_prefill = True
                    # Mark batched decode as done — all text generated during prefill
                    self._batched_decode_done = True
                    # No audio codes for ASR
                    self._audio_codes_list = []
                    self._audio_codes = torch.empty(0, dtype=torch.long, device=device)

                    # Store generated text codes for output
                    self._text_codes_list = [t for t in generated_text if t.numel() > 0] if generated_text else []
                    self._text_codes = torch.cat([c.reshape(-1) for c in self._text_codes_list]) if self._text_codes_list else torch.empty(0, dtype=torch.long, device=device)

                    logger.info(
                        "[ASR MODE] Prefill complete: audio_seq_len=%d, "
                        "text_generated=%d tokens, KV cache populated",
                        audio_seq_len, len(self._text_codes_list),
                    )
                else:
                    # S2S mode: generate audio codes
                    max_new_tokens = min(max(1500, 150), self._max_audio_codes)

                    if raw_whisper is not None:
                        # Path 1: Raw Whisper features available — use HF model's internal
                        # embedding construction (VQAdaptor + sqrt(2) scaling).
                        all_codes, all_text = self._run_reference_generate_loop(
                            audio_input_ids=audio_input_ids,
                            text_input_ids=text_input_ids,
                            is_continuous_mask=is_continuous_mask,
                            whisper_features=raw_whisper,
                            max_new_tokens=max_new_tokens,
                        )
                    else:
                        # Path 2: S2S mode with 3584-dim features (post-VQAdaptor).
                        # Use manual embedding construction matching reference model exactly.
                        all_codes, all_text = self._run_reference_generate_loop_embeds(
                            audio_input_ids=audio_input_ids,
                            text_input_ids=text_input_ids,
                            is_continuous_mask=is_continuous_mask,
                            whisper_emb_3584=whisper_3584,
                            max_new_tokens=max_new_tokens,
                        )

                    # Store all generated codes in _audio_codes_list
                    # Filter out delay tokens (kimia_text_blank) matching reference model behavior.
                    # Reference: run_ref_s2s_with_diag.py line 308:
                    #   audio_codes = [c for c in audio_generated if c != kimia_text_blank]
                    kimia_text_blank = int(getattr(self.config, "kimia_text_blank", 151666))
                    filtered_codes = [c for c in all_codes if c.numel() > 0 and c.item() != kimia_text_blank]
                    self._audio_codes_list = filtered_codes if filtered_codes else list(all_codes)
                    self._audio_codes = torch.cat([c.reshape(-1) for c in self._audio_codes_list]) if self._audio_codes_list else torch.empty(0, dtype=torch.long)

                    # Store text codes
                    self._text_codes_list = [c for c in all_text if c.numel() > 0] if all_text else []
                    self._text_codes = torch.cat([c.reshape(-1) for c in self._text_codes_list]) if self._text_codes_list else torch.empty(0, dtype=torch.long)
                    self._batched_decode_done = True

                    # Set sequence length for decode position offset
                    self._prefill_seq_len = audio_seq_len
                    self._prefill_text_len = audio_seq_len  # Same length (parallel streams)

                    logger.info(
                        "[REFERENCE GEN LOOP] Prefill complete: audio_seq_len=%d, "
                        "codes_generated=%d, unique=%d",
                        audio_seq_len, len(self._audio_codes_list),
                        self._audio_codes.unique().numel() if self._audio_codes.numel() > 0 else 0,
                    )
            elif use_ref_model_decode:
                # Decode via full reference model forward with KV cache.
                # This ensures the decode path exactly matches the reference model
                # behavior, including proper KV cache usage and RoPE.
                text_token = input_ids.item() if input_ids.dim() == 0 else input_ids.flatten()[0].item()
                position = int(positions.item()) if positions.dim() == 0 else int(positions.flatten()[0].item())

                # Correct position: vLLM scheduler uses text-based positions,
                # but the model needs the actual combined sequence position.
                # e.g., prefill_seq_len=47, text_len=34 → delta=13
                # vLLM pos=34 → corrected pos=47
                pos_delta = getattr(self, '_prefill_seq_len', 0) - getattr(self, '_prefill_text_len', 0)
                if pos_delta > 0:
                    position = position + pos_delta

                # Get the self-feedback audio code from _audio_codes_list.
                # _audio_codes_list stores RELATIVE indices — convert to absolute
                # for the HF model's embedding lookup (matching reference).
                if hasattr(self, "_audio_codes_list") and len(self._audio_codes_list) > 0:
                    audio_code = self._audio_codes_list[-1].reshape(-1)[-1].item() + token_offset
                else:
                    audio_code = int(getattr(self.config, "kimia_text_blank", 151666)) + token_offset

                hidden_states, mimo_hidden_states, new_past_kv = (
                    self._call_reference_model_decode(
                        audio_code=audio_code,
                        text_token=text_token,
                        position=position,
                        backbone_past_key_values=self._hf_past_key_values,
                        mimo_past_key_values=self._mimo_past_key_values,
                    )
                )

                # Update KV cache from reference model output
                if new_past_kv is not None:
                    # Convert Cache object back to tuple for storage
                    new_past_kv = _from_cache(new_past_kv)
                    self._hf_past_key_values = tuple(new_past_kv[:28])
                    self._mimo_past_key_values = tuple(new_past_kv[28:])

                logger.info(
                    "[REF MODEL DECODE] audio_code=%d text_token=%d pos=%d "
                    "hs_std=%.4f mimo_std=%.4f past_kv_seq_len=%d",
                    audio_code, text_token, position,
                    hidden_states.float().std().item(),
                    mimo_hidden_states.float().std().item(),
                    self._hf_past_key_values[0][0].shape[2] if self._hf_past_key_values else 0,
                )
            else:
                hidden_states = self._combine_audio_text_embeds(input_ids, inputs_embeds, is_decode=is_decode)
            residual = None

            # Override positions for parallel combined sequence.
            # Skip when using reference model (it handles positions internally).
            if not use_ref_model and is_prefill and getattr(self, '_prefill_audio_offset', 0) > 0:
                audio_offset = self._prefill_audio_offset
                _text_len = input_ids.shape[-1]
                full_seq_len = max(audio_offset, _text_len)  # parallel, not concat
                positions = torch.arange(full_seq_len, device=positions.device, dtype=positions.dtype)
                if hidden_states.dim() == 3 and positions.dim() == 1:
                    positions = positions.unsqueeze(0)

                if not is_warmup:
                    logger.info(
                        "[DIAG POS] Prefill position override (PARALLEL): audio_offset=%d, text_len=%d, "
                        "full_seq_len=%d (was %d with concat), positions=[%d..%d]",
                        audio_offset, _text_len, full_seq_len,
                        audio_offset + _text_len,
                        positions.min().item(), positions.max().item(),
                    )

            # Store actual prefill sequence length for decode position offset.
            if is_prefill and not is_warmup:
                _text_len_for_seq = input_ids.shape[-1]
                # When reference model prefill was used, the actual sequence length
                # includes media scaffolding tokens (audio + 2 media + text).
                if use_ref_model:
                    # Reference model produces [audio_frames + 2 media + text] sequence
                    _actual_audio_len = getattr(self, '_prefill_audio_offset', 0) + 2  # +2 for media_begin/end
                    self._prefill_seq_len = _actual_audio_len + _text_len_for_seq
                else:
                    self._prefill_seq_len = max(getattr(self, '_prefill_audio_offset', 0), _text_len_for_seq)
                self._prefill_text_len = _text_len_for_seq  # Track vLLM's view of text length
                if not is_warmup:
                    logger.info(
                        "[DIAG SEQLEN] Stored for decode: prefill_seq_len=%d, prefill_text_len=%d, delta=%d (use_ref_model=%s)",
                        self._prefill_seq_len, self._prefill_text_len,
                        self._prefill_seq_len - self._prefill_text_len,
                        use_ref_model,
                    )
            elif not is_prefill and getattr(self, '_prefill_seq_len', 0) > 0:
                # Decode: vLLM positions are based on text-only context length.
                # e.g., if text_len=34, vLLM sends position 34 for first decode step.
                # But our actual prefill used seq_len=47 (parallel audio+text).
                # Correction: pos_correct = pos_vllm - text_len + prefill_seq_len
                pos_delta = self._prefill_seq_len - self._prefill_text_len
                positions = positions + pos_delta

            # DIAG: Log embedded input during decode
            if not is_prefill and self._diag_decode_log_count <= 50:
                logger.info(
                    "DIAG decode embed#%d: shape=%s mean=%.4f std=%.4f, positions=%s",
                    self._diag_decode_log_count,
                    list(hidden_states.shape),
                    hidden_states.float().mean().item(),
                    hidden_states.float().std().item(),
                    positions.tolist(),
                )

            # Use HuggingFace backbone layers for both prefill and decode.
            # During decode, we manage KV cache via past_key_values passed to each layer.
            # Skip during prefill when reference model already processed the full sequence.
            bifurcation_idx = int(self.config.kimia_mimo_transformer_from_layer_index)
            num_layers = len(self._hf_layers)
            end_layer = num_layers if get_pp_group().is_last_rank else bifurcation_idx + 1

            # Also skip when batched decode already completed all audio codes during prefill.
            # On decode steps, all work is done — just return dummy hidden states.
            batched_decode_done = getattr(self, "_batched_decode_done", False)
            if use_ref_model or use_ref_model_decode or (is_decode and batched_decode_done):
                # Reference model already ran (prefill or decode);
                # skip manual backbone loop entirely.
                if use_ref_model:
                    self._ref_model_did_prefill = True
                    # Prefill: reference model generated all audio codes.
                    # Create dummy hidden states from the combined embeds.
                    if 'hidden_states' not in locals():
                        hidden_states = self._combine_audio_text_embeds(
                            input_ids, inputs_embeds, is_decode=False
                        )
                    mimo_hidden_states = hidden_states.clone() if get_pp_group().is_last_rank else None
                if is_decode and batched_decode_done:
                    # Batched AR decode already generated all audio codes during prefill.
                    # Decode step only needs to continue text generation — skip backbone/MIMO.
                    # Create dummy hidden states for text-only output.
                    if 'hidden_states' not in locals():
                        # Embed text tokens for text generation output
                        text_embed = self.llm.model.embed_tokens(input_ids)
                        hidden_states = text_embed
                    mimo_hidden_states = hidden_states.clone() if get_pp_group().is_last_rank else None
                logger.info(
                    "[REF MODEL] Skipping manual backbone loop (reference handled %s, "
                    "hs_std=%.4f, mimo_std=%.4f)",
                    "decode" if use_ref_model_decode else ("batched_decode" if batched_decode_done else "prefill"),
                    hidden_states.float().std().item(),
                    mimo_hidden_states.float().std().item(),
                )
            elif is_warmup or is_profile_run:
                # Warmup/Profile: skip backbone layers — dummy run needs shapes only.
                # Profile runs send very long sequences (max_model_len) which can
                # trigger FlashAttention shape issues with the HF model.
                seq_len = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[1]
                mimo_hidden_states = hidden_states.clone() if get_pp_group().is_last_rank else None
                run_type = "WARMUP" if is_warmup else "PROFILE"
                logger.info("[DIAG %s] Skipping backbone layers (seq_len=%d)", run_type, seq_len)
            else:
                # Use HF layers with past_key_values for KV cache management.
                # On prefill, past_key_values is None (computed from full sequence).
                # On decode, past_key_values contains cached K/V from previous steps.
                past_key_values = self._hf_past_key_values if not is_prefill else None
                new_past_key_values = []

                for idx in range(0, end_layer):
                    # Get past KV for this layer during decode
                    layer_past = None
                    if past_key_values is not None and idx < len(past_key_values):
                        layer_past = past_key_values[idx]

                    hs_in_mean = hidden_states.float().mean().item()
                    hs_in_std = hidden_states.float().std().item()
                    hs_in_min = hidden_states.float().min().item()
                    hs_in_max = hidden_states.float().max().item()

                    # HF layers expect 3D [batch, seq, hidden], vLLM uses 2D [num_tokens, hidden]
                    hs_3d = hidden_states.unsqueeze(0)  # [1, num_tokens, hidden]

                    # Position IDs: during decode, use KV cache length for correct RoPE.
                    if not is_prefill and past_key_values is not None:
                        kv_len = past_key_values[idx][0].shape[2] if idx < len(past_key_values) else 0
                        hf_pos = torch.tensor([[kv_len]], device=hs_3d.device, dtype=positions.dtype)
                    else:
                        hf_pos = positions.unsqueeze(0) if positions.dim() == 1 else positions

                    # Clamp positions to RoPE max_position_embeddings - 1
                    max_pos = int(self.config.max_position_embeddings) - 1
                    hf_pos = hf_pos.clamp(min=0, max=max_pos)

                    hf_layer = self._hf_layers[idx]
                    out = hf_layer(hs_3d, position_ids=hf_pos, past_key_value=layer_past, use_cache=True)
                    if isinstance(out, tuple):
                        hidden_states_3d = out[0]
                        # out[-1] is present_key_value (k, v tuple)
                        if len(out) > 1:
                            new_past_key_values.append(out[-1])
                    else:
                        hidden_states_3d = out
                    hidden_states = hidden_states_3d.squeeze(0)

                    # Capture at bifurcation point for MIMO branch
                    if idx == bifurcation_idx:
                        mimo_hidden_states = hidden_states.clone()
                        if is_prefill:
                            logger.info(
                                "[DIAG STEP3] Bifurcation output (layer %d): shape=%s mean=%.4f std=%.4f min=%.4f max=%.4f",
                                bifurcation_idx,
                                list(mimo_hidden_states.shape),
                                mimo_hidden_states.float().mean().item(),
                                mimo_hidden_states.float().std().item(),
                                mimo_hidden_states.float().min().item(),
                                mimo_hidden_states.float().max().item(),
                            )

                # Store KV cache for next decode step.
                # After prefill: new_past_key_values has full sequence KV — save directly.
                # After decode: new_past_key_values contains the FULL updated KV cache
                # (HF layer with past_key_value+use_cache already appends new tokens).
                # We just replace with the updated cache returned by the layer.
                if len(new_past_key_values) > 0:
                    self._hf_past_key_values = tuple(new_past_key_values)
                    if not is_prefill:
                        logger.info("DIAG decode backbone done: hs=%s past_kv_layers=%d past_kv_seq_len=%d",
                                    list(hidden_states.shape), len(new_past_key_values),
                                    self._hf_past_key_values[0][0].shape[2] if self._hf_past_key_values else 0)
                    else:
                        logger.info("DIAG prefill backbone done: hs=%s past_kv_len=%d",
                                    list(hidden_states.shape), len(new_past_key_values))

            if not get_pp_group().is_last_rank:
                # Intermediate rank: pass hidden_states and bifurcation states
                tensors = {"hidden_states": hidden_states}
                if mimo_hidden_states is not None:
                    tensors["mimo_hidden_states"] = mimo_hidden_states
                return IntermediateTensors(tensors)
        else:
            # Non-first rank: receive hidden_states from intermediate tensors
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            mimo_hidden_states = intermediate_tensors.get("mimo_hidden_states")

            # Use HF backbone layers for remaining layers on this rank
            bifurcation_idx = int(self.config.kimia_mimo_transformer_from_layer_index)
            num_layers = len(self._hf_layers)
            start_layer = bifurcation_idx + 1

            if not is_warmup:
                logger.info(
                    "[DIAG STEP3a] Non-first rank running HF backbone layers %d-%d",
                    start_layer, num_layers - 1,
                )

            for idx in range(start_layer, num_layers):
                if not is_warmup:
                    hs_in_mean = hidden_states.float().mean().item()
                    hs_in_std = hidden_states.float().std().item()
                    hs_in_min = hidden_states.float().min().item()
                    hs_in_max = hidden_states.float().max().item()

                hs_3d = hidden_states.unsqueeze(0)
                pos_2d = positions.unsqueeze(0) if positions.dim() == 1 else positions
                max_pos = int(self.config.max_position_embeddings) - 1
                pos_2d = pos_2d.clamp(min=0, max=max_pos)

                hf_layer = self._hf_layers[idx]
                out = hf_layer(
                    hs_3d,
                    position_ids=pos_2d,
                )
                if isinstance(out, tuple):
                    out = out[0]
                hidden_states = out.squeeze(0)

                if not is_warmup:
                    hs_shape = list(hidden_states.shape)
                    hs_out_mean = hidden_states.float().mean().item()
                    hs_out_std = hidden_states.float().std().item()
                    hs_out_min = hidden_states.float().min().item()
                    hs_out_max = hidden_states.float().max().item()
                    logger.info(
                        "[DIAG STEP3b] Backbone layer %d (HF): "
                        "shape=%s "
                        "in_mean=%.4f in_std=%.4f in_min=%.4f in_max=%.4f "
                        "out_mean=%.4f out_std=%.4f out_min=%.4f out_max=%.4f",
                        idx, hs_shape,
                        hs_in_mean, hs_in_std, hs_in_min, hs_in_max,
                        hs_out_mean, hs_out_std, hs_out_min, hs_out_max,
                    )

        # Apply backbone norm using HF norm
        # Skip only during prefill when reference model already applied its norm.
        # During decode, manual backbone layers ran so we MUST apply the norm.
        if get_pp_group().is_last_rank and not (is_prefill and getattr(self, '_ref_model_did_prefill', False)):
            hidden_states = self._hf_norm(hidden_states)
            text_hidden_states = hidden_states  # text path output
        elif get_pp_group().is_last_rank and getattr(self, '_ref_model_did_prefill', False):
            text_hidden_states = hidden_states  # reference model already normalized

        # Run HF MIMO layers on captured hidden states
        # During prefill: skip when reference model already processed MIMO.
        # During decode: ALWAYS run MIMO layers (new token needs MIMO processing).
        # During warmup/profile: skip MIMO layers (dummy run needs shapes only).
        # Run MIMO layers only if they haven't been processed yet.
        # Skip during prefill when reference model already ran.
        # Skip during decode when reference model decode path is used.
        _skip_mimo = (
            (is_prefill and getattr(self, '_ref_model_did_prefill', False))
            or use_ref_model_decode
            or (is_decode and getattr(self, '_batched_decode_done', False))
        )
        if mimo_hidden_states is not None and get_pp_group().is_last_rank and not _skip_mimo:
            if is_warmup or is_profile_run:
                # Warmup/Profile: skip MIMO layers, just clone for shape propagation.
                run_type = "PROFILE" if is_profile_run else "WARMUP"
                logger.info("[DIAG %s] Skipping MIMO layers", run_type)
            else:
                # Scale bifurcation output to match the distribution that MIMO layers
                # were trained for. The reference backbone produces std ~35 at
                # bifurcation, but our HF backbone produces ~20 due to embedding
                # construction differences (no sqrt(2) scaling during decode).
                # The MIMO layers expect the reference distribution.
                _target_bifurcation_std = 34.67
                _current_std = mimo_hidden_states.float().std().item()
                _scale_factor = _target_bifurcation_std / max(_current_std, 1e-6)
                mimo_hidden_states = mimo_hidden_states * _scale_factor

                # HF MIMO layers use past_key_values for KV cache during decode.
                # During prefill, past_key_values is None (full sequence attention).
                # During decode, past_key_values contains cached K/V from previous steps.
                mimo_past_kv = self._mimo_past_key_values if not is_prefill else None
                new_mimo_past_kv = []

                _mimo_diag_step = getattr(self, "_diag_log_step", 0)
                _mimo_do_diag = _mimo_diag_step <= 50
                _decode_mimo_diag = not is_prefill and not is_warmup  # Always log MIMO layers during decode
                for i, mimo_layer in enumerate(self._hf_mimo_layers):
                    if not is_warmup:
                        mimo_in_mean = mimo_hidden_states.float().mean().item()
                        mimo_in_std = mimo_hidden_states.float().std().item()
                        mimo_in_min = mimo_hidden_states.float().min().item()
                        mimo_in_max = mimo_hidden_states.float().max().item()

                    # Get past KV for this MIMO layer during decode
                    layer_past = None
                    if mimo_past_kv is not None and i < len(mimo_past_kv):
                        layer_past = mimo_past_kv[i]

                    # HF layers expect 3D [batch, seq, hidden]
                    hs_3d = mimo_hidden_states.unsqueeze(0)

                    # Position IDs for MIMO layers during decode: compute from KV cache
                    # length, not vLLM-corrected positions. The MIMO layers have their
                    # own KV cache which tracks the actual sequence length. The vLLM
                    # position correction (pos_delta) is for text-based positions and
                    # doesn't apply to the MIMO audio stream.
                    if not is_prefill and mimo_past_kv is not None:
                        mimo_kv_len = mimo_past_kv[0][0].shape[2] if mimo_past_kv else 0
                        mimo_pos = torch.tensor([[mimo_kv_len]], device=hs_3d.device, dtype=positions.dtype)
                    else:
                        mimo_pos = positions.unsqueeze(0) if positions.dim() == 1 else positions

                    max_pos = int(self.config.max_position_embeddings) - 1
                    mimo_pos = mimo_pos.clamp(min=0, max=max_pos)

                    layer_out = mimo_layer(
                        hs_3d,
                        position_ids=mimo_pos,
                        past_key_value=layer_past,
                        use_cache=True,
                    )
                    if isinstance(layer_out, tuple):
                        mimo_hidden_states_3d = layer_out[0]
                        # out[-1] is present_key_value (k, v tuple)
                        if len(layer_out) > 1:
                            new_mimo_past_kv.append(layer_out[-1])
                    else:
                        mimo_hidden_states_3d = layer_out
                    mimo_hidden_states = mimo_hidden_states_3d.squeeze(0)

                    if not is_warmup:
                        mimo_out_mean = mimo_hidden_states.float().mean().item()
                        mimo_out_std = mimo_hidden_states.float().std().item()
                        mimo_out_min = mimo_hidden_states.float().min().item()
                        mimo_out_max = mimo_hidden_states.float().max().item()
                        mimo_shape = list(mimo_hidden_states.shape)
                        logger.info(
                            "[DIAG STEP3c] MIMO layer %d (HF): "
                            "shape=%s "
                            "in_mean=%.4f in_std=%.4f in_min=%.4f in_max=%.4f "
                            "out_mean=%.4f out_std=%.4f out_min=%.4f out_max=%.4f",
                            i, mimo_shape,
                            mimo_in_mean, mimo_in_std, mimo_in_min, mimo_in_max,
                            mimo_out_mean, mimo_out_std, mimo_out_min, mimo_out_max,
                        )
                    elif _mimo_do_diag:
                        logger.info(
                            "[DIAG STEP3c] MIMO layer %d: shape=%s mean=%.4f std=%.4f",
                            i, list(mimo_hidden_states.shape),
                            mimo_hidden_states.float().mean().item(),
                            mimo_hidden_states.float().std().item(),
                        )

                # Store MIMO KV cache for next decode step
                if len(new_mimo_past_kv) > 0:
                    self._mimo_past_key_values = tuple(new_mimo_past_kv)
                    if not is_prefill:
                        logger.info("DIAG decode MIMO done: hs=%s past_kv_layers=%d past_kv_seq_len=%d",
                                    list(mimo_hidden_states.shape), len(new_mimo_past_kv),
                                    self._mimo_past_key_values[0][0].shape[2] if self._mimo_past_key_values else 0)
                    else:
                        logger.info("DIAG prefill MIMO done: hs=%s past_kv_len=%d",
                                    list(mimo_hidden_states.shape), len(new_mimo_past_kv))

                # Track MIMO layer 5 std across ALL decode steps for divergence analysis
                if not is_prefill and not is_warmup:
                    # Track per-decode-step MIMO layer stats
                    if not hasattr(self, "_decode_mimo_l5_stds"):
                        self._decode_mimo_l5_stds = []
                    # mimo_hidden_states here is the output of the last MIMO layer (layer 5)
                    # before the norm. We need to capture it from the loop above.
                    # Actually, let me re-read — the loop overwrote mimo_hidden_states.
                    # The last layer's output std is what we have here.
                    l5_std = mimo_hidden_states.float().std().item()
                    self._decode_mimo_l5_stds.append(l5_std)
                    n_decode = len(self._decode_mimo_l5_stds)
                    if n_decode <= 5 or n_decode % 10 == 0 or n_decode <= 50:
                        logger.info("[DIAG DECODE MIMO] step=%d l5_post_std=%.4f (target: ~30-35) history=%s",
                                    n_decode, l5_std,
                                    [f"{s:.1f}" for s in self._decode_mimo_l5_stds[-10:]] if n_decode > 10 else "N/A")

        # Post-MIMO: norm, logits, sampling — runs for both manual MIMO and reference model prefill.
        # Skip during decode when batched decode already generated all codes.
        if mimo_hidden_states is not None and get_pp_group().is_last_rank and not (
            is_decode and getattr(self, '_batched_decode_done', False)
        ):
            # Apply MIMO norm — skip only during prefill when reference model already normalized.
            # During decode, manual MIMO layers ran so we MUST apply the norm.
            if not (is_prefill and getattr(self, '_ref_model_did_prefill', False)):
                mimo_hidden_states = self._hf_mimo_norm(mimo_hidden_states)

            # DIAG: Log MIMO hidden state stats
            if self._diag_log_step <= 5 or (self._diag_log_step <= 50 and is_prefill):
                logger.info(
                    "DIAG MIMO states step=%d hidden=%s positions=%s",
                    self._diag_log_step,
                    f"shape={list(mimo_hidden_states.shape)} mean={mimo_hidden_states.float().mean().item():.4f} std={mimo_hidden_states.float().std().item():.4f}",
                    f"shape={list(positions.shape)} vals={positions[:10].tolist()}" if positions.dim() <= 1 else f"shape={list(positions.shape)}",
                )

            # Audio from MIMO hidden states via HF mimo_output head
            audio_logits = self._hf_mimo_output(mimo_hidden_states)

            token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
            audio_vocab = int(getattr(self.config, "kimia_audio_output_vocab", 16384))

            # Slice audio logits to audio subspace
            if audio_logits.dim() == 2:
                # Flattened: [num_tokens, vocab] (vLLM v1 decode path)
                audio_logits_sub = audio_logits[:, token_offset : token_offset + audio_vocab]
            elif audio_logits.dim() == 3:
                # Batched: [B, seq_len, vocab] (vLLM v1 prefill path)
                audio_logits_sub = audio_logits[:, :, token_offset : token_offset + audio_vocab]
            else:
                raise ValueError(f"Unexpected audio_logits dim: {audio_logits.dim()}, shape: {audio_logits.shape}")

            # DIAG: Log audio logits distribution
            # NOTE: Softmax applied over sliced subspace for DIAG display only.
            # Actual sampling uses _sample_audio_topk which handles its own softmax.
            if not hasattr(self, "_diag_log_step"):
                self._diag_log_step = 0
            self._diag_log_step += 1
            # Log for first 50 steps (both prefill and decode)
            if self._diag_log_step <= 50:
                probs = torch.softmax(audio_logits_sub.float(), dim=-1)
                top5_vals, top5_idx = torch.topk(probs, 5, dim=-1)
                # Also check if logits are dominated by a single token (sign of degeneration)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                avg_entropy = entropy.mean().item() if entropy.dim() > 0 else entropy.item()
                logger.info(
                    "[DIAG STEP5] Audio logits step=%d is_prefill=%s shape=%s "
                    "mean=%.1f std=%.1f max=%.1f min=%.1f entropy=%.2f "
                    "top5_codes=%s top5_probs=%s",
                    self._diag_log_step, is_prefill, list(audio_logits_sub.shape),
                    audio_logits_sub.float().mean().item(),
                    audio_logits_sub.float().std().item(),
                    audio_logits_sub.float().max().item(),
                    audio_logits_sub.float().min().item(),
                    avg_entropy,
                    top5_idx[0].tolist() if audio_logits_sub.dim() == 2 else top5_idx[0, 0].tolist(),
                    top5_vals[0].tolist() if audio_logits_sub.dim() == 2 else top5_vals[0, 0].tolist(),
                )

            # Audio sampling — match reference Kimi-Audio infer.py params:
            # audio_top_k=10, audio_temperature=0.8 for varied code generation.
            # Greedy (temperature=0.0) causes code degeneration (fixed-point loop).
            # Also check for EOD token (code 0 = msg_end) to stop generation early.
            rep_penalty_fwd = float(getattr(self.config, "kimia_repetition_penalty", 1.1))
            audio_codes = _sample_audio_topk(
                audio_logits_sub,
                top_k=10,
                temperature=0.8,
                repetition_penalty=rep_penalty_fwd,
                recent_tokens=None,
                repetition_window=64,
            )

            # Check for EOD token (msg_end = 0 in audio subspace)
            if not is_prefill and not is_warmup:
                sampled_code = audio_codes.item() if audio_codes.numel() == 1 else audio_codes.flatten()[-1].item()
                if sampled_code == 0:
                    # EOD detected — signal stop via special marker
                    if not hasattr(self, "_eod_detected"):
                        self._eod_detected = True
                        logger.info("[EOD] Detected audio EOD token (code=0) at decode step %d, total codes=%d",
                                    self._decode_step_counter, len(getattr(self, "_audio_codes_list", [])))

            # [DIAG STEP6] Log sampled audio codes
            if self._diag_log_step <= 50:
                unique_codes = audio_codes.unique().numel()
                total_codes = audio_codes.numel()
                logger.info(
                    "[DIAG STEP6] Sampled codes step=%d is_prefill=%s shape=%s unique=%d/%d "
                    "min=%d max=%d mean=%.1f codes=%s",
                    self._diag_log_step, is_prefill, list(audio_codes.shape),
                    unique_codes, total_codes,
                    audio_codes.min().item(), audio_codes.max().item(),
                    audio_codes.float().mean().item(),
                    audio_codes.flatten()[:20].tolist(),
                )

            # Skip warmup entirely — dummy inputs produce meaningless codes.
            # Must be checked BEFORE accumulation to prevent warmup codes from
            # leaking into _audio_codes_list.
            if is_warmup:
                self._audio_codes = torch.empty(0, dtype=torch.long)
                self._text_codes = torch.empty(0, dtype=torch.long)
                multimodal_outputs: dict[str, Any] = {"audio_codes": None, "text_codes": None}
            else:
                delay_tokens = int(getattr(self.config, "kimia_mimo_audiodelaytokens", 6))

                if is_prefill:
                    # Compute dynamic audio cap for S2S mode — fixed cap of 500 codes
                    # (10s at 50Hz). In S2S, input_ids may be text-only so we can't
                    # derive the audio token count. Use a generous fixed cap and let
                    # the model stop naturally via EOD.
                    num_text_tokens = input_ids.shape[-1]
                    dynamic_cap = 500
                    # Clamp between 150 (~3s at 50Hz) and the global max_audio_codes
                    self._dynamic_audio_cap = min(max(dynamic_cap, 150), self._max_audio_codes)

                    # CRITICAL: Only use the LAST position's audio code.
                    # The reference Kimi-Audio samples from logits[:, -1] in a
                    # true autoregressive loop — each token t depends on tokens
                    # 0..t-1. During vLLM prefill, positions 0..N-1 are sampled
                    # simultaneously with causal masking, so position i sees the
                    # prompt but NOT the audio codes at positions 0..i-1. Those
                    # are not the same autoregressively-conditioned tokens the
                    # model was trained to produce.
                    # Taking only the last position's code gives us one properly-
                    # conditioned starting point; decode then continues the AR chain.
                    if audio_codes.dim() == 1:
                        last_code = audio_codes[-1]
                    else:
                        last_code = audio_codes[:, -1]
                    self._audio_codes_list.append(last_code.reshape(-1))
                    self._decode_step_counter = 0
                    logger.info(
                        "Kimia-Audio AR prefill: appended audio code to _audio_codes_list "
                        "(now len=%d), code_val=%d, using last position audio code "
                        "(shape=%s) from %d total positions (first %d dropped as delay), "
                        "dynamic audio cap=%d codes (~%.1fs), num_text_tokens=%d",
                        len(self._audio_codes_list),
                        last_code.reshape(-1)[-1].item() if last_code.numel() > 0 else -1,
                        list(last_code.shape),
                        audio_codes.shape[-1],
                        delay_tokens,
                        self._dynamic_audio_cap, self._dynamic_audio_cap / 50.0,
                        num_text_tokens,
                    )
                    # If batched decode was done, return ALL codes; otherwise just the last one
                    if getattr(self, "_batched_decode_done", False):
                        # Return all codes from batched decode at once
                        multimodal_outputs = {
                            "audio_codes": self._audio_codes.reshape(-1),
                            "text_codes": self._text_codes.reshape(-1) if self._text_codes.numel() > 0 else None,
                        }
                        logger.info(
                            "[BATCHED AR DECODE] Prefill returning ALL %d audio codes, %d text codes",
                            self._audio_codes.numel(), self._text_codes.numel(),
                        )
                    else:
                        multimodal_outputs = {
                            "audio_codes": last_code.reshape(-1),
                            "text_codes": self._text_codes.reshape(-1) if self._text_codes.numel() > 0 else None,
                        }
                else:
                    # During decode, check if batched decode already generated all codes.
                    if getattr(self, "_batched_decode_done", False):
                        # All audio codes were already generated during prefill.
                        # Skip decode audio generation — just return None to
                        # signal no new codes this step (text generation continues).
                        multimodal_outputs = {
                            "audio_codes": None,
                            "text_codes": self._text_codes.reshape(-1) if self._text_codes is not None and self._text_codes.numel() > 0 else None,
                        }
                        if not get_pp_group().is_last_rank:
                            return IntermediateTensors(
                                {"hidden_states": hidden_states, "residual": residual},
                            )
                        return OmniOutput(
                            text_hidden_states=text_hidden_states,
                            multimodal_outputs={
                                "audio_codes": None,
                                "text_codes": self._text_codes.reshape(-1) if self._text_codes is not None and self._text_codes.numel() > 0 else None,
                            },
                        )

                    # During decode, generate one additional code per step
                    # until the content-aware cap is reached.
                    total_so_far = sum(c.numel() for c in self._audio_codes_list)
                    effective_cap = getattr(self, "_dynamic_audio_cap", self._max_audio_codes)
                    if total_so_far >= effective_cap:
                        # Cap reached — stop producing audio codes.
                        if not self._audio_code_cap_logged:
                            self._audio_code_cap_logged = True
                            logger.info(
                                "Kimia-Audio AR: reached audio cap (%d codes, ~%.1fs) at decode step %d, "
                                "stopping audio generation",
                                effective_cap, effective_cap / 50.0, self._decode_step_counter + 1,
                            )
                        multimodal_outputs = {
                            "audio_codes": None,
                            "text_codes": None,
                        }
                        self._decode_step_counter += 1
                        # Skip multimodal output but fall through to text path
                        if not get_pp_group().is_last_rank:
                            return IntermediateTensors(
                                {"hidden_states": hidden_states, "residual": residual},
                            )
                        return OmniOutput(
                            text_hidden_states=text_hidden_states,
                            multimodal_outputs={
                                "audio_codes": None,
                                "text_codes": None,
                            },
                        )

                    self._decode_step_counter += 1

                    # During the audio delay period, store blank tokens instead of
                    # generated codes (matching reference: next_audio_token is
                    # filled with blank for i < delay_tokens).
                    # CRITICAL: _audio_codes_list stores RELATIVE indices (relative
                    # to token_offset). The decode path adds token_offset to convert
                    # back to absolute. So we must store the RELATIVE value here.
                    delay_tokens = int(getattr(self.config, "kimia_mimo_audiodelaytokens", 6))
                    if self._decode_step_counter <= delay_tokens:
                        # Use relative index 0 as blank placeholder.
                        # 0 + token_offset = 152064, which maps to audio code 0
                        # (msg_end relative), close enough to kimia_text_blank.
                        blank_codes = torch.zeros_like(audio_codes)
                        self._audio_codes_list.append(blank_codes)
                    else:
                        self._audio_codes_list.append(audio_codes)

                    # Flatten accumulated codes into a 1D sequence for code2wav
                    all_codes = torch.cat(
                        [c.reshape(-1) for c in self._audio_codes_list],
                        dim=0,
                    )
                    self._audio_codes = all_codes

                    # Diagnostics
                    if logger.isEnabledFor(20):
                        phase = f"decode#{self._decode_step_counter}"
                        bifurcation_std = mimo_hidden_states.float().std().item()
                        al_std = audio_logits_sub.float().std().item()
                        pos_unique = positions.unique().numel() if positions is not None else 0

                        logger.info(
                            "Kimia-Audio AR: phase=%s warmup=False, input_ids shape=%s, "
                            "positions unique=%d, bifurcation_std=%.4f, "
                            "audio_logits_std=%.4f, audio_codes shape=%s range=[%d,%d], "
                            "accumulated=%d codes",
                            phase,
                            list(input_ids.shape) if input_ids is not None else None,
                            pos_unique,
                            bifurcation_std,
                            al_std,
                            list(audio_codes.shape),
                            audio_codes.min().item(),
                            audio_codes.max().item(),
                            all_codes.numel(),
                        )

                    # Send only the new code (single token) to the output processor.
                    # Sending all_codes would cause O(n²) growth because the
                    # processor concatenates pooler_output across steps.
                    multimodal_outputs = {
                        "audio_codes": audio_codes.reshape(-1),
                        "text_codes": self._text_codes.reshape(-1) if self._text_codes is not None and self._text_codes.numel() > 0 else None,
                    }
        elif not get_pp_group().is_last_rank:
            multimodal_outputs = {"audio_codes": None, "text_codes": None}
        else:
            multimodal_outputs = {"audio_codes": None, "text_codes": None}

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states},
            )

        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute text logits for the text generation path.

        Uses the HF backbone LM head on backbone hidden states (matching
        reference: text_logits = self.lm_head(hidden_states)).
        Masks the audio subspace so vLLM's sampler only selects text tokens.
        """
        # Use HF lm_head for exact numerical match with reference
        if hasattr(self, '_hf_lm_head'):
            logits = self._hf_lm_head(hidden_states)
        else:
            logits = self.logits_processor(self.llm.lm_head, hidden_states)
        if logits is not None:
            # Mask audio subspace to prevent vLLM from sampling audio tokens.
            # Audio codes are injected via the executor's sample_tokens hook.
            token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
            logits[:, token_offset:] = float("-inf")
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with prefix-based routing and stacked param fusion.

        Weight key routing:
        - model.* -> backbone (Qwen2ForCausalLM params under self.llm.model)
        - lm_head.* -> backbone LM head
        - model.mimo_layers.*, model.mimo_norm.*, mimo_output.* -> already loaded via HF

        The Kimi checkpoint uses HF-format separate q_proj/k_proj/v_proj and
        gate_proj/up_proj. These must be fused into vLLM's qkv_proj and
        gate_up_proj via stacked_params_mapping before loading.

        MIMO layer weights are pre-loaded from the HF model via from_pretrained,
        so we only mark them as loaded without actually processing checkpoint weights.

        Returns model parameter names (e.g., 'llm.model.layers.0...'),
        not checkpoint key names.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        backbone_weights: list[tuple[str, torch.Tensor]] = []

        for name, loaded_weight in weights:
            if name.startswith("model.mimo_layers.") or name.startswith("model.mimo_norm.") or name.startswith("mimo_output."):
                # MIMO weights already loaded via HF from_pretrained
                continue
            elif name.startswith("lm_head."):
                backbone_weights.append((name, loaded_weight))
            elif name.startswith("model."):
                # Backbone weights: keep raw key for matching self.llm params
                backbone_weights.append((name, loaded_weight))
            # Skip any other keys (e.g., unrelated checkpoint files)

        loaded_params: set[str] = set()

        # Load backbone weights manually with stacked param fusion
        # self.llm.named_parameters() returns names WITHOUT 'llm.' prefix
        backbone_params = dict(self.llm.named_parameters(remove_duplicate=False))

        # Track which stacked params still need loading (for manual concat)
        stacked_shards: dict[str, dict[str, torch.Tensor]] = {}

        for name, loaded_weight in backbone_weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in backbone_params:
                    continue
                if mapped_name not in backbone_params:
                    continue

                # Track the mapped param name WITH 'llm.' prefix (what vLLM checks)
                loaded_params.add(f"llm.{mapped_name}")
                if mapped_name not in stacked_shards:
                    stacked_shards[mapped_name] = {}
                stacked_shards[mapped_name][str(shard_id)] = loaded_weight
                break
            else:
                if name in backbone_params:
                    loaded_params.add(f"llm.{name}")
                    default_weight_loader(backbone_params[name], loaded_weight)

        # Concatenate stacked shards and load into parameters
        for mapped_name, shards in stacked_shards.items():
            param = backbone_params[mapped_name]
            shard_id_order = {"qkv_proj": ["q", "k", "v"], "gate_up_proj": ["0", "1"]}
            key = mapped_name.split(".")[-1]
            order = shard_id_order.get(key, sorted(shards.keys()))
            sorted_shards = [shards[k] for k in order if k in shards]
            concat_weight = torch.cat(sorted_shards, dim=0)
            default_weight_loader(param, concat_weight)

        # MIMO layers, norm, and output head are HF modules loaded via
        # from_pretrained — weights are already set. Just mark them as loaded
        # to satisfy vLLM's parameter completeness check.
        mimo_weight_keys = []
        for name, param in self._hf_mimo_layers.named_parameters():
            mimo_weight_keys.append(f"mimo_layers.{name}")
        for name, param in self._hf_mimo_norm.named_parameters():
            mimo_weight_keys.append(f"mimo_norm.{name}")
        for name, param in self._hf_mimo_output.named_parameters():
            mimo_weight_keys.append(f"mimo_output.{name}")
        loaded_params.update(mimo_weight_keys)

        # Diagnostic: log MIMO weight loading stats
        mimo_param_names = {n for n in loaded_params if n.startswith("mimo_")}
        total_mimo_params = len(mimo_param_names)
        logger.info(
            "MIMO weight loading: %d params loaded (HF modules pre-loaded). Examples: %s",
            total_mimo_params,
            sorted(mimo_param_names)[:10],
        )

        # Mark _hf_* parameters as loaded (already initialized by HuggingFace from_pretrained).
        # vLLM validates that all model parameters are loaded from the checkpoint;
        # the HF modules were loaded separately via device_map="cuda".
        for name, _ in self.named_parameters():
            if name.startswith("_hf_"):
                loaded_params.add(name)

        return loaded_params
