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
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.interfaces import SupportsRealtime
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
from vllm.model_executor.models.utils import make_layers
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)




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
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config

        # Patch rope_theta onto config for transformers 5.8+ compatibility.
        if not hasattr(self.config, 'rope_theta'):
            rope_params = getattr(self.config, 'rope_parameters', {})
            self.config.rope_theta = rope_params.get('rope_theta', 1000000.0)

        # Fix incorrect kimia_audio_output_vocab in the checkpoint config
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

        # Build Qwen2Config matching Kimi-Audio parameters for vLLM decoder layers
        rope_theta = getattr(self.config, 'rope_theta', None) or self.config.rope_parameters.get('rope_theta', 1000000.0)
        self._qwen2_config = Qwen2Config(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            rope_theta=rope_theta,
            num_hidden_layers=self.config.num_hidden_layers,
        )
        # Ensure rope_parameters exists for vLLM's Qwen2Attention
        if not hasattr(self._qwen2_config, 'rope_parameters') or self._qwen2_config.rope_parameters is None:
            self._qwen2_config.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        # Ensure hidden_act exists (Qwen2MLP requires it)
        if not hasattr(self._qwen2_config, 'hidden_act') or self._qwen2_config.hidden_act is None:
            self._qwen2_config.hidden_act = "silu"

        # Embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        # Backbone: 28x Qwen2DecoderLayer (vLLM-native, CUDA-graph compatible)
        num_backbone_layers = self.config.num_hidden_layers  # 28
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_backbone_layers,
            lambda prefix: Qwen2DecoderLayer(
                config=self._qwen2_config,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        # Backbone norm
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Text output head
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=f"{prefix}.lm_head",
        )

        # MIMO branch: 6x Qwen2DecoderLayer (vLLM-native, CUDA-graph compatible)
        # These branch off at layer 21 (kimia_mimo_transformer_from_layer_index)
        num_mimo_layers = 6
        _, _, self.mimo_layers = make_layers(
            num_mimo_layers,
            lambda prefix: Qwen2DecoderLayer(
                config=self._qwen2_config,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.mimo_layers",
        )

        # MIMO norm and output head (simple linear, not ParallelLMHead which requires sampler)
        self.mimo_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        from vllm.model_executor.layers.linear import ReplicatedLinear
        self.mimo_output = ReplicatedLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.mimo_output",
        )

        # Bifurcation point: hidden states are cloned after this layer index
        self._bifurcation_layer = int(
            getattr(self.config, "kimia_mimo_transformer_from_layer_index", 21)
        )

        # Logits processor for text generation path
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
        )

        # Track decode steps for delay token application
        self._decode_step_counter: int = 0
        self._diag_decode_log_count: int = 0
        self._diag_log_step: int = 0

        # Accumulated audio codes across prefill + decode steps
        self._audio_codes_list: list[torch.Tensor] = []
        self._audio_codes: torch.Tensor | None = None
        self._text_codes_list: list[torch.Tensor] = []
        self._text_codes: torch.Tensor | None = None

        # Cap audio codes to prevent runaway generation
        self._max_audio_codes: int = int(
            getattr(self.config, "kimia_max_audio_codes", 3000),
        )
        self._audio_code_cap_logged: bool = False

        # Whisper embedding for reference audio (TTS mode)
        self._whisper_emb: torch.Tensor | None = None
        self._speaker_emb: torch.Tensor | None = None
        self._load_whisper_embedding()

        # Per-request Whisper features from user input audio (S2S mode)
        self._input_whisper_emb: torch.Tensor | None = None
        self._input_whisper_emb_logged: bool = False
        self._input_whisper_raw: torch.Tensor | None = None
        self._input_whisper_raw_logged: bool = False
        self._is_asr_mode: bool = False
        self._asr_text_hidden_states: torch.Tensor | None = None
        self._asr_text_seq_len: int = 0

        # Lazy-loaded Whisper feature extractor for runtime extraction
        self._whisper_extractor: Any | None = None

        # Mark model as having multimodal outputs
        self.have_multimodal_outputs = True

        logger.info(
            "KimiAudio vLLM-native model initialized: %d backbone layers + %d MIMO layers, "
            "bifurcation at layer %d, vocab=%d",
            num_backbone_layers, num_mimo_layers, self._bifurcation_layer, self.config.vocab_size,
        )

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed_tokens

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
        embed_tokens = self.embed_tokens

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
        """Forward pass using vLLM-native decoder layers.

        Uses vLLM's Qwen2DecoderLayer for both backbone (28 layers) and MIMO
        (6 layers) branches, with vLLM's pre-allocated KV cache.

        vLLM layer forward signature: layer(positions, hidden_states, residual)
        → (hidden_states, residual)
        """
        # Token offset for audio token ID space
        token_offset = int(getattr(self.config, "kimia_token_offset", 152064))

        # Detect prefill vs decode phase
        is_prefill = input_ids is not None and input_ids.dim() > 0 and input_ids.shape[-1] > 1

        # Detect warmup/dummy runs using shape + value range checks (graph-compatible).
        # Real decode produces valid token IDs >= 100. Warmup uses tiny dummy values
        # (typically 0-5). We use max() comparison which is graph-safe (no .item()).
        _seq_len = input_ids.shape[-1] if input_ids is not None and input_ids.dim() > 0 else 0
        _is_dummy_values = (
            input_ids is not None
            and input_ids.dim() > 0
            and input_ids.max() < 10
        )
        is_warmup = _seq_len <= 2 and _is_dummy_values
        is_profile_run = _seq_len > 1000

        if is_profile_run or is_warmup:
            logger.info(
                "[DIAG DETECT] %s: is_prefill=%s, seq_len=%d",
                "PROFILE" if is_profile_run else "WARMUP",
                is_prefill, _seq_len,
            )

        # Reset state at start of new request (prefill)
        if is_prefill and not is_warmup:
            self._decode_step_counter = 0
            self._audio_codes_list = []
            self._audio_code_cap_logged = False
            self._input_whisper_emb = None
            self._input_whisper_emb_logged = False
            self._input_whisper_raw = None
            self._input_whisper_raw_logged = False
            self._prefill_audio_offset = 0
            self._prefill_seq_len = 0
            self._prefill_text_len = 0
            self._is_asr_mode = False
            self._decode_whisper_template = None
            self._decode_whisper_frames = None

        # Extract Whisper features from kwargs
        if is_prefill and kwargs:
            self._extract_whisper_from_kwargs(kwargs)

        # DIAGNOSTIC: Log audio code self-feedback during decode (graph-compatible)
        if not is_prefill and not is_warmup and input_ids is not None:
            if self._diag_decode_log_count < 50:
                self._diag_decode_log_count += 1
                logger.info(
                    "DIAG decode#%d input_ids_shape=%s, positions_shape=%s",
                    self._diag_decode_log_count,
                    list(input_ids.shape),
                    list(positions.shape),
                )

        # Embed input on first rank
        is_decode = not is_prefill

        if get_pp_group().is_first_rank:
            hidden_states = self._combine_audio_text_embeds(input_ids, inputs_embeds, is_decode=is_decode)
            residual = None

            # Override positions for parallel combined sequence during prefill
            if is_prefill and not is_warmup and getattr(self, '_prefill_audio_offset', 0) > 0:
                audio_offset = self._prefill_audio_offset
                _text_len = input_ids.shape[-1]
                full_seq_len = max(audio_offset, _text_len)
                positions = torch.arange(full_seq_len, device=positions.device, dtype=positions.dtype)
                if hidden_states.dim() == 3 and positions.dim() == 1:
                    positions = positions.unsqueeze(0)

                if not is_warmup:
                    logger.info(
                        "[DIAG POS] Prefill position override (PARALLEL): audio_offset=%d, text_len=%d, "
                        "full_seq_len=%d, positions=[%d..%d]",
                        audio_offset, _text_len, full_seq_len,
                        positions.min().item(), positions.max().item(),
                    )

            # Store prefill sequence length for decode position offset
            if is_prefill and not is_warmup:
                _text_len_for_seq = input_ids.shape[-1]
                self._prefill_seq_len = max(getattr(self, '_prefill_audio_offset', 0), _text_len_for_seq)
                self._prefill_text_len = _text_len_for_seq
                if not is_warmup:
                    logger.info(
                        "[DIAG SEQLEN] Stored for decode: prefill_seq_len=%d, prefill_text_len=%d, delta=%d",
                        self._prefill_seq_len, self._prefill_text_len,
                        self._prefill_seq_len - self._prefill_text_len,
                    )
            elif not is_prefill and getattr(self, '_prefill_seq_len', 0) > 0:
                pos_delta = self._prefill_seq_len - self._prefill_text_len
                positions = positions + pos_delta

            # DIAG: Log embedded input during decode (graph-compatible)
            if not is_prefill and not is_warmup and self._diag_decode_log_count <= 50:
                logger.info(
                    "DIAG decode embed#%d: shape=%s",
                    self._diag_decode_log_count,
                    list(hidden_states.shape),
                )
        else:
            # Non-first rank: receive hidden_states from intermediate tensors
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors.get("residual")

        # Run backbone layers using vLLM-native pattern
        # vLLM Qwen2DecoderLayer: forward(positions, hidden_states, residual) → (hidden_states, residual)
        bifurcation_idx = self._bifurcation_layer
        num_layers = len(self.layers)
        end_layer = num_layers if get_pp_group().is_last_rank else bifurcation_idx + 1

        if is_warmup or is_profile_run:
            # Warmup/Profile: skip layers — dummy run needs shapes only
            seq_len = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[1]
            mimo_hidden_states = hidden_states.clone() if get_pp_group().is_last_rank else None
            run_type = "WARMUP" if is_warmup else "PROFILE"
            logger.info("[DIAG %s] Skipping backbone layers (seq_len=%d)", run_type, seq_len)
        elif get_pp_group().is_first_rank or (not get_pp_group().is_first_rank and not get_pp_group().is_last_rank):
            # Run backbone layers up to bifurcation point (or all on first rank for non-PP)
            for idx in range(0, end_layer):
                # vLLM layers work with 2D [num_tokens, hidden] directly
                layer = self.layers[idx]
                hidden_states, residual = layer(positions, hidden_states, residual)

                # Capture at bifurcation point for MIMO branch
                if idx == bifurcation_idx:
                    mimo_hidden_states = hidden_states.clone()

        if not get_pp_group().is_last_rank:
            tensors = {"hidden_states": hidden_states}
            if mimo_hidden_states is not None:
                tensors["mimo_hidden_states"] = mimo_hidden_states
            if residual is not None:
                tensors["residual"] = residual
            return IntermediateTensors(tensors)

        # Last rank: continue with remaining backbone layers (if PP)
        if not get_pp_group().is_first_rank:
            start_layer = bifurcation_idx + 1
            for idx in range(start_layer, num_layers):
                hidden_states, residual = self.layers[idx](positions, hidden_states, residual)

        # Apply backbone norm
        hidden_states = self.norm(hidden_states)
        text_hidden_states = hidden_states

        # Run MIMO layers
        if mimo_hidden_states is not None:
            if is_warmup or is_profile_run:
                logger.info("[DIAG %s] Skipping MIMO layers", "PROFILE" if is_profile_run else "WARMUP")
            else:
                mimo_residual = None
                for i, mimo_layer in enumerate(self.mimo_layers):
                    mimo_hidden_states, mimo_residual = mimo_layer(positions, mimo_hidden_states, mimo_residual)

            # Apply MIMO norm
            mimo_hidden_states = self.mimo_norm(mimo_hidden_states)

            # Compute audio logits
            audio_logits_out = self.mimo_output(mimo_hidden_states)
            if isinstance(audio_logits_out, tuple):
                audio_logits = audio_logits_out[0]
            else:
                audio_logits = audio_logits_out

            # Slice to audio subspace
            audio_vocab = int(getattr(self.config, "kimia_audio_output_vocab", 16384))
            if audio_logits.dim() == 2:
                audio_logits_sub = audio_logits[:, token_offset : token_offset + audio_vocab]
            elif audio_logits.dim() == 3:
                audio_logits_sub = audio_logits[:, :, token_offset : token_offset + audio_vocab]
            else:
                raise ValueError(f"Unexpected audio_logits dim: {audio_logits.dim()}, shape: {audio_logits.shape}")

            # Sample audio codes
            audio_codes = _sample_audio_topk(
                audio_logits_sub,
                top_k=10,
                temperature=0.8,
                repetition_penalty=float(getattr(self.config, "kimia_repetition_penalty", 1.1)),
                recent_tokens=None,
                repetition_window=64,
            )

            # Accumulate audio codes
            if is_warmup:
                self._audio_codes = torch.empty(0, dtype=torch.long)
                self._text_codes = torch.empty(0, dtype=torch.long)
                multimodal_outputs = {"audio_codes": None, "text_codes": None}
            else:
                delay_tokens = int(getattr(self.config, "kimia_mimo_audiodelaytokens", 6))

                if is_prefill:
                    # During prefill, only use the last position's audio code
                    # (properly AR-conditioned)
                    if audio_codes.dim() == 1:
                        last_code = audio_codes[-1]
                    else:
                        last_code = audio_codes[:, -1]
                    self._audio_codes_list.append(last_code.reshape(-1))
                    self._decode_step_counter = 0

                    # Compute dynamic audio cap
                    num_text_tokens = input_ids.shape[-1]
                    dynamic_cap = 500
                    self._dynamic_audio_cap = min(max(dynamic_cap, 150), self._max_audio_codes)

                    multimodal_outputs = {
                        "audio_codes": last_code.reshape(-1),
                        "text_codes": self._text_codes.reshape(-1) if hasattr(self, '_text_codes') and self._text_codes is not None and self._text_codes.numel() > 0 else None,
                    }
                else:
                    # Decode step: accumulate codes
                    self._decode_step_counter += 1
                    effective_cap = getattr(self, "_dynamic_audio_cap", self._max_audio_codes)
                    total_so_far = sum(c.numel() for c in self._audio_codes_list)
                    if total_so_far >= effective_cap:
                        if not self._audio_code_cap_logged:
                            self._audio_code_cap_logged = True
                            logger.info(
                                "Kimia-Audio AR: reached audio cap (%d codes, ~%.1fs) at decode step %d",
                                effective_cap, effective_cap / 50.0, self._decode_step_counter,
                            )
                        multimodal_outputs = {"audio_codes": None, "text_codes": None}
                        return OmniOutput(
                            text_hidden_states=text_hidden_states,
                            multimodal_outputs=multimodal_outputs,
                        )

                    # Store codes (blank during delay period)
                    if self._decode_step_counter <= delay_tokens:
                        blank_codes = torch.zeros_like(audio_codes)
                        self._audio_codes_list.append(blank_codes)
                    else:
                        self._audio_codes_list.append(audio_codes)

                    all_codes = torch.cat([c.reshape(-1) for c in self._audio_codes_list], dim=0)
                    self._audio_codes = all_codes

                    multimodal_outputs = {
                        "audio_codes": audio_codes.reshape(-1),
                        "text_codes": self._text_codes.reshape(-1) if hasattr(self, '_text_codes') and self._text_codes is not None and self._text_codes.numel() > 0 else None,
                    }
        else:
            multimodal_outputs = {"audio_codes": None, "text_codes": None}

        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs=multimodal_outputs,
        )
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute text logits for the text generation path.

        Uses the vLLM-native lm_head on backbone hidden states.
        Masks the audio subspace so vLLM's sampler only selects text tokens.
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None:
            # Mask audio subspace to prevent vLLM from sampling audio tokens.
            token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
            logits[:, token_offset:] = float("-inf")
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with prefix-based routing and stacked param fusion.

        Weight key routing:
        - model.embed_tokens.* -> self.embed_tokens
        - model.layers.{i}.* -> self.layers.{i}.* (with q/k/v→qkv, gate/up→gate_up merge)
        - model.norm.* -> self.norm
        - lm_head.* -> self.lm_head
        - model.mimo_layers.{i}.* -> self.mimo_layers.{i}.* (with merge)
        - model.mimo_norm.* -> self.mimo_norm
        - mimo_output.* -> self.mimo_output

        The Kimi checkpoint uses HF-format separate q_proj/k_proj/v_proj and
        gate_proj/up_proj. These must be fused into vLLM's qkv_proj and
        gate_up_proj via stacked_params_mapping.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Collect all model parameters (names WITHOUT module prefix)
        model_params = dict(self.named_parameters(remove_duplicate=False))

        # Track which stacked params still need loading
        stacked_shards: dict[str, dict[str, torch.Tensor]] = {}
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Route checkpoint keys to module parameter names.
            # Checkpoint uses "model." prefix but named_parameters() returns without it.
            # Checkpoint: "model.layers.{i}.*" → Module: "layers.{i}.*"
            # Checkpoint: "model.mimo_layers.{i}.*" → Module: "mimo_layers.{i}.*"
            # Checkpoint: "model.embed_tokens.*" → Module: "embed_tokens.*"
            # Checkpoint: "model.norm.*" → Module: "norm.*"
            # Checkpoint: "model.mimo_norm.*" → Module: "mimo_norm.*"
            # Checkpoint: "model.lm_head.*" → Module: "lm_head.*"
            # Checkpoint: "mimo_output.*" → Module: "mimo_output.*"
            if name.startswith("model."):
                param_name = name[len("model."):]
            else:
                param_name = name

            if param_name not in model_params:
                # Check if it matches a stacked param pattern
                is_stacked = False
                for _, weight_name, _ in stacked_params_mapping:
                    if weight_name in param_name:
                        is_stacked = True
                        break
                if not is_stacked:
                    continue  # Unknown param, skip

            # Check stacked param mapping
            for param_name_stem, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in param_name:
                    continue
                mapped_name = param_name.replace(weight_name, param_name_stem)
                if mapped_name not in model_params:
                    continue
                loaded_params.add(mapped_name)
                if mapped_name not in stacked_shards:
                    stacked_shards[mapped_name] = {}
                stacked_shards[mapped_name][str(shard_id)] = loaded_weight
                break
            else:
                if param_name in model_params:
                    loaded_params.add(param_name)
                    default_weight_loader(model_params[param_name], loaded_weight)

        # Concatenate stacked shards and load
        for mapped_name, shards in stacked_shards.items():
            param = model_params[mapped_name]
            key = mapped_name.split(".")[-1]  # e.g., "qkv_proj" or "gate_up_proj"
            order = {"qkv_proj": ["q", "k", "v"], "gate_up_proj": ["0", "1"]}.get(key, sorted(shards.keys()))
            sorted_shards = [shards[k] for k in order if k in shards]
            if sorted_shards:
                concat_weight = torch.cat(sorted_shards, dim=0)
                default_weight_loader(param, concat_weight)

        return loaded_params
