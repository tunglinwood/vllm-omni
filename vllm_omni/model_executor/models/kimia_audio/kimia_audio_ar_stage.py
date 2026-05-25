# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimia Audio AR stage: S2S model extending vLLM upstream with MIMO layers.

This module implements the fused_thinker_talker stage for Kimi-Audio S2S (speech-to-speech).
It inherits from vLLM upstream's ASR model and adds MIMO layers for audio generation.

Architecture:
- Inherits from vLLM upstream: Whisper encoder, VQAdaptor, embeddings, 28-layer backbone, lm_head
- Adds: 6 MIMO decoder layers branching at layer 21, mimo_norm, mimo_output (audio head)
- Fixes: VQAdaptor GELU → SiLU (upstream bug, reference uses SiLU)
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Iterable
from typing import Any

import numpy as np
import torch
from transformers import Qwen2Config
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsRealtime
from vllm.model_executor.models.kimi_audio import (
    KimiAudioForConditionalGeneration,
    KimiAudioMultiModalProjector,
    KimiAudioWhisperEncoder,
)
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
    # Apply repetition penalty — match reference: only activate after we have
    # more tokens than the repetition window. The reference KimiASampler checks
    # `len(recent_tokens) > audio_repetition_window_size` (64), which means
    # the penalty doesn't kick in until enough audio codes have been generated.
    # Applying it too early reduces diversity and causes repetitive audio.
    if repetition_penalty > 1.0 and recent_tokens is not None and recent_tokens.numel() > repetition_window:
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
        """Yield TokensPrompt with Whisper features as audio segments accumulate.

        Uses a custom WhisperFeatureExtractor for backward compatibility.
        Prefer get_input_stream_with_tower for new code.
        """
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

    async def get_input_stream_with_tower(
        self,
        audio_tower: KimiAudioWhisperEncoder,
        projector: KimiAudioMultiModalProjector,
        sampling_rate: int = 16000,
    ) -> AsyncGenerator[TokensPrompt, None]:
        """Yield TokensPrompt with Whisper features using upstream audio_tower + projector.

        This replaces the custom WhisperFeatureExtractor path with vLLM upstream's
        KimiAudioWhisperEncoder for feature extraction.
        """
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                self._ended = True
                break

            self._accumulated.append(chunk)
            self._total_samples += len(chunk)

            if self._total_samples >= self._segment_samples:
                full_audio = np.concatenate(self._accumulated)
                segment = full_audio[: self._segment_samples]
                remainder = full_audio[self._segment_samples :]

                # Extract Whisper features using upstream audio_tower + projector
                # (same as upstream _process_audio_input, but for raw audio)
                with torch.no_grad():
                    audio_tensor = torch.from_numpy(segment).float().cuda()
                    # audio_tower expects list of tensors (Mel spectrogram features)
                    # For raw audio, we use the HF processor to get Mel features
                    audio_features = audio_tower([audio_tensor])

                    # 4x downsample (same as upstream _process_audio_input)
                    B, T, D = audio_features.shape
                    if T % 4 != 0:
                        pad_len = 4 - (T % 4)
                        audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_len))
                        T = audio_features.shape[1]
                    audio_features = audio_features.reshape(B, T // 4, D * 4)

                    # Project to LLM dimension
                    whisper_emb = projector(audio_features)

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

        if self._total_samples > 0:
            full_audio = np.concatenate(self._accumulated)
            with torch.no_grad():
                audio_tensor = torch.from_numpy(full_audio).float().cuda()
                audio_features = audio_tower([audio_tensor])
                B, T, D = audio_features.shape
                if T % 4 != 0:
                    pad_len = 4 - (T % 4)
                    audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_len))
                    T = audio_features.shape[1]
                audio_features = audio_features.reshape(B, T // 4, D * 4)
                whisper_emb = projector(audio_features)
            yield TokensPrompt(
                prompt_token_ids=[],
                mm_processor_kwargs={
                    "whisper_input_feature": whisper_emb,
                },
            )


class KimiAudioFusedForConditionalGeneration(
    KimiAudioForConditionalGeneration,
    CustomProcessMixin,
    SupportsRealtime,
):
    """Kimi-Audio model with MIMO layers for speech-to-speech generation.

    Inherits from vLLM upstream's ASR model (Whisper, VQAdaptor, embeddings,
    backbone, lm_head) and adds:
    - 6 MIMO decoder layers branching at layer 21
    - Audio output head (mimo_output)
    - S2S dual-stream embedding with Whisper injection
    - Fixed VQAdaptor activation (GELU -> SiLU)

    The S2S-specific embedding construction and forward pass override the
    upstream ASR-only implementations.
    """

    realtime_max_tokens = 128
    """Maximum audio code tokens to generate per streaming audio segment."""

    enforce_eager = True
    prefer_model_sampler = True
    """Use model's sample() method for audio code sampling outside CUDA graph."""
    requires_raw_input_tokens = True
    """Pass input_ids to forward() alongside inputs_embeds for Whisper injection."""

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[TokensPrompt, None]:
        """Buffer streaming audio and yield TokensPrompt with Whisper features.

        Uses vLLM upstream's KimiAudioWhisperEncoder for feature extraction,
        not a custom WhisperFeatureExtractor.
        """
        from vllm.config import VllmConfig

        sampling_rate = 16000
        segment_duration_s = 5.0

        buffer = KimiaRealtimeBuffer(
            sampling_rate=sampling_rate,
            segment_duration_s=segment_duration_s,
        )

        async def feed_audio():
            async for chunk in audio_stream:
                await buffer.append_audio(chunk)
            await buffer.append_audio(None)

        audio_task = asyncio.create_task(feed_audio())

        # Build upstream audio_tower + projector for realtime extraction
        torch.set_default_dtype(torch.bfloat16)
        vllm_config = VllmConfig(model_config=model_config)
        vllm_config.model_config.multimodal_config = None  # minimal config, we build manually

        # Reuse upstream's KimiAudioWhisperEncoder
        audio_tower = KimiAudioWhisperEncoder(
            vllm_config=vllm_config,
            prefix="audio_tower",
        )
        audio_tower = audio_tower.cuda().bfloat16()
        audio_tower.eval()

        # Reuse upstream's KimiAudioMultiModalProjector
        projector = KimiAudioMultiModalProjector(
            whisper_dim=getattr(model_config.hf_config, "kimia_adaptor_input_dim", 5120),
            llm_dim=model_config.hf_config.hidden_size,
            prefix="multi_modal_projector",
        )
        projector = projector.cuda().bfloat16()
        projector.eval()

        # Apply SiLU fix (same as _fix_vq_adaptor_activation)
        def silu_forward(audio_features):
            hidden = projector.vq_adaptor_layers_0(audio_features)
            hidden = torch.nn.functional.silu(hidden)
            hidden = projector.vq_adaptor_layers_3(hidden)
            hidden = projector.vq_adaptor_layers_4(hidden)
            return hidden

        projector.forward = silu_forward

        # Load weights from shards
        from safetensors.torch import load_file  # noqa: I001
        import glob

        model_dir = model_config.model.rstrip("/")
        shards = sorted(glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors")))
        all_weights = []
        for shard_path in shards:
            ckpt = load_file(shard_path)
            all_weights.extend(ckpt.items())

        # Map weight names for audio_tower and projector
        from vllm.model_executor.models.kimi_audio import (
            KimiAudioForConditionalGeneration as UpstreamKimiAudio,
        )

        mapper = UpstreamKimiAudio.hf_to_vllm_mapper
        for name, weight in all_weights:
            mapped = name
            if mapper:
                for orig, new in mapper.orig_to_new_prefix.items():
                    if name.startswith(orig):
                        mapped = new + name[len(orig) :]
                        break
            if mapped.startswith("audio_tower.") or mapped.startswith("multi_modal_projector."):
                # Find matching param and copy weight
                for param_name, param in list(audio_tower.named_parameters()) + list(projector.named_parameters()):
                    if param_name == mapped:
                        param.data.copy_(weight)
                        break

        try:
            async for tokens_prompt in buffer.get_input_stream_with_tower(
                audio_tower,
                projector,
                sampling_rate=sampling_rate,
            ):
                yield tokens_prompt
        finally:
            audio_task.cancel()

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # Ensure multimodal_config is set — the upstream class's
        # _mark_tower_model context manager requires it. For S2S,
        # Whisper features are passed via additional_information,
        # but the upstream still creates the audio tower.
        if vllm_config.model_config.multimodal_config is None:
            from vllm.config import MultiModalConfig

            object.__setattr__(
                vllm_config.model_config,
                "multimodal_config",
                MultiModalConfig(),
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Fix VQAdaptor: GELU -> SiLU (upstream bug, reference uses SiLU)
        self._fix_vq_adaptor_activation()

        # Store vllm_config for MIMO layer initialization
        self.vllm_config = vllm_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config

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

        # Build Qwen2Config matching Kimi-Audio parameters for MIMO decoder layers
        rope_theta = getattr(self.config, "rope_theta", None)
        rope_theta = rope_theta or self.config.rope_parameters.get("rope_theta", 1_000_000.0)
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
        if not hasattr(self._qwen2_config, "rope_parameters") or self._qwen2_config.rope_parameters is None:
            self._qwen2_config.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        if not hasattr(self._qwen2_config, "hidden_act") or self._qwen2_config.hidden_act is None:
            self._qwen2_config.hidden_act = "silu"

        # MIMO branch: 6 Qwen2DecoderLayer branching off backbone at layer 21
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

        # MIMO norm and output head
        self.mimo_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.mimo_output = ReplicatedLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.mimo_output",
        )

        # Bifurcation point: hidden states cloned after this layer index
        self._bifurcation_layer = int(getattr(self.config, "kimia_mimo_transformer_from_layer_index", 21))

        # Logits processor for text generation path
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
        )

        # Track decode steps for delay token application
        self._decode_step_counter: int = 0

        # Accumulated audio codes across prefill + decode steps
        self._audio_codes_list: list[torch.Tensor] = []
        self._audio_codes: torch.Tensor | None = None
        self._text_codes_list: list[torch.Tensor] = []
        self._text_codes: torch.Tensor | None = None

        # Cap audio codes to prevent runaway generation.
        # The reference model uses max_new_tokens = int(12.5 * 120) - input_codes
        # which limits total audio to ~12.5 seconds. At ~75 codes/sec (reference
        # rate), that's ~937 total codes. We use 1000 as a safe upper bound.
        self._max_audio_codes: int = int(
            getattr(self.config, "kimia_max_audio_codes", 1000),
        )

        # Audio EOS tokens — when the model predicts one of these, stop
        # generating audio codes (same as reference eod_ids check).
        # Reference: eod_ids = [msg_end, media_end]
        self._audio_eod_ids: set[int] = set()
        # media_end from config
        if hasattr(self.config, "kimia_media_end"):
            self._audio_eod_ids.add(int(self.config.kimia_media_end))
        # msg_end: same as eos_token_id[1] in Kimia config
        if hasattr(self.config, "eos_token_ids"):
            eos_ids = self.config.eos_token_ids
            if isinstance(eos_ids, list):
                for eid in eos_ids:
                    self._audio_eod_ids.add(int(eid))
            else:
                self._audio_eod_ids.add(int(eos_ids))
        # Fallback: hardcode known token IDs if config keys missing
        if not self._audio_eod_ids:
            self._audio_eod_ids = {151645, 151663}  # msg_end, media_end

        self._audio_eos_triggered: bool = False

        # Per-request Whisper features from user input audio (S2S mode)
        self._input_whisper_emb: torch.Tensor | None = None
        self._input_whisper_emb_logged: bool = False
        self._input_whisper_raw: torch.Tensor | None = None
        self._input_whisper_raw_logged: bool = False
        self._is_asr_mode: bool = False
        self._asr_text_hidden_states: torch.Tensor | None = None
        self._asr_text_seq_len: int = 0

        # Mark model as having multimodal outputs
        self.have_multimodal_outputs = True

        # Access backbone layers via upstream's language_model
        # These are Qwen2DecoderLayer instances created by vLLM upstream
        self.layers = self.language_model.model.layers
        self.start_layer = getattr(self.language_model.model, "start_layer", 0)
        self.end_layer = getattr(self.language_model.model, "end_layer", len(self.layers))
        self.norm = self.language_model.model.norm

        logger.info(
            "KimiaS2S model initialized: vLLM upstream backbone (%d layers) + %d MIMO layers, "
            "bifurcation at layer %d, vocab=%d",
            len(self.layers),
            num_mimo_layers,
            self._bifurcation_layer,
            self.config.vocab_size,
        )

    def _fix_vq_adaptor_activation(self):
        """Fix upstream's GELU -> SiLU in VQAdaptor to match reference model."""
        if not hasattr(self, "multi_modal_projector"):
            return
        projector = self.multi_modal_projector
        if not hasattr(projector, "vq_adaptor_layers_0"):
            return

        def silu_forward(audio_features):
            hidden = projector.vq_adaptor_layers_0(audio_features)
            hidden = torch.nn.functional.silu(hidden)
            hidden = projector.vq_adaptor_layers_3(hidden)
            hidden = projector.vq_adaptor_layers_4(hidden)
            return hidden

        projector.forward = silu_forward

    def _extract_whisper_from_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Extract Whisper features from kwargs passed by the vLLM engine.

        For S2S serving: runner provides pre-extracted 5120-dim Whisper features.
        We store them as _input_whisper_emb; _combine_audio_text_embeds applies
        the multi_modal_projector (VQAdaptor) to get 3584-dim features.
        """
        # Store raw audio array if present (for diagnostics)
        audio_array = kwargs.get("audio_array")
        if audio_array is None:
            buffer = kwargs.get("model_intermediate_buffer")
            if buffer is not None and isinstance(buffer, list) and buffer:
                first_entry = buffer[0]
                if isinstance(first_entry, dict):
                    audio_array = first_entry.get("audio_array")
                    sample_rate = first_entry.get("audio_sample_rate")
                    if audio_array is not None and sample_rate is not None:
                        kwargs["audio_sample_rate"] = sample_rate

        if audio_array is not None:
            if isinstance(audio_array, (list, tuple)):  # noqa: UP038
                flat = []
                for item in audio_array:
                    if isinstance(item, (list, tuple)):  # noqa: UP038
                        flat.extend(item)
                    else:
                        flat.append(item)
                audio_array = np.array(flat, dtype=np.float32)
            elif isinstance(audio_array, np.ndarray):
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                audio_array = audio_array.reshape(-1)
            elif hasattr(audio_array, "shape") and len(getattr(audio_array, "shape", ())) > 1:
                audio_array = audio_array.reshape(-1).astype(np.float32)
            self._audio_array = audio_array
            self._audio_sample_rate = kwargs.get("audio_sample_rate", 16000)

        # ASR mode check
        is_asr = kwargs.get("is_asr_mode")
        if is_asr is None:
            buffer = kwargs.get("model_intermediate_buffer")
            if isinstance(buffer, list) and buffer and isinstance(buffer[0], dict):
                is_asr = buffer[0].get("is_asr_mode")
        if is_asr is None:
            runtime_info = kwargs.get("runtime_additional_information")
            if isinstance(runtime_info, dict):
                is_asr = runtime_info.get("is_asr_mode")
            elif isinstance(runtime_info, list) and runtime_info:
                is_asr = runtime_info[0].get("is_asr_mode") if isinstance(runtime_info[0], dict) else None
        if is_asr is not None:
            self._is_asr_mode = bool(is_asr)

        # Mode 1: Pre-extracted Whisper features or Mel spectrograms at top level
        whisper_feature = kwargs.get("whisper_input_feature")
        if whisper_feature is not None:
            if isinstance(whisper_feature, (torch.Tensor, np.ndarray)):  # noqa: UP038
                if isinstance(whisper_feature, np.ndarray):
                    whisper_feature = torch.from_numpy(whisper_feature)

                # Detect Mel spectrogram: shape [B, 128, T] or [128, T]
                # Mel spectrograms have 128 frequency bins (Whisper config)
                mel_dim = 128
                is_mel = False
                if whisper_feature.dim() == 3 and whisper_feature.shape[1] == mel_dim:
                    is_mel = True
                elif whisper_feature.dim() == 2 and whisper_feature.shape[0] == mel_dim:
                    is_mel = True

                if is_mel:
                    # Route through upstream audio_tower pipeline:
                    # Mel -> WhisperEncoder -> 4x downsample -> projector -> 3584-dim
                    # audio_tower conv layers are bfloat16 on CUDA, so cast and move
                    device = self.audio_tower.conv1.weight.device
                    whisper_feature = whisper_feature.to(device=device, dtype=torch.bfloat16)
                    audio_input = {"whisper_input_features": whisper_feature}

                    # Diagnostic: print Mel spectrogram stats before audio_tower
                    if os.environ.get("KIMIA_DIAG_AUDIO", "0") == "1":
                        mel = whisper_feature.float()
                        print(f"    [DIAG] Mel input: shape={list(mel.shape)}, "
                              f"mean={mel.mean():.6f}, std={mel.std():.6f}, "
                              f"min={mel.min():.6f}, max={mel.max():.6f}")

                    self._input_whisper_emb = self._process_audio_input(audio_input)

                    # Diagnostic: print 3584-dim output stats after full pipeline
                    if os.environ.get("KIMIA_DIAG_AUDIO", "0") == "1":
                        emb = self._input_whisper_emb.float()
                        print(f"    [DIAG] After audio_tower+projector: shape={list(emb.shape)}, "
                              f"mean={emb.mean():.6f}, std={emb.std():.6f}, "
                              f"min={emb.min():.6f}, max={emb.max():.6f}")
                else:
                    # Pre-extracted features (5120-dim or 3584-dim) — store directly
                    self._input_whisper_emb = whisper_feature
            whisper_raw = kwargs.get("whisper_raw")
            if whisper_raw is not None and isinstance(whisper_raw, (torch.Tensor, np.ndarray)):  # noqa: UP038
                if isinstance(whisper_raw, np.ndarray):
                    whisper_raw = torch.from_numpy(whisper_raw).cuda()
                elif whisper_raw.device.type != "cuda":
                    whisper_raw = whisper_raw.cuda()
                self._input_whisper_raw = whisper_raw
            return

        # Mode 2: Per-request Whisper features via model_intermediate_buffer
        buffer = kwargs.get("model_intermediate_buffer")
        if buffer is not None and isinstance(buffer, list):
            for req_info in buffer:
                if isinstance(req_info, dict) and "whisper_input_feature" in req_info:
                    whisper_feature = req_info["whisper_input_feature"]
                    if isinstance(whisper_feature, (torch.Tensor, np.ndarray)):  # noqa: UP038
                        if isinstance(whisper_feature, np.ndarray):
                            whisper_feature = torch.from_numpy(whisper_feature)

                        # Detect Mel spectrogram: shape [B, 128, T] or [128, T]
                        mel_dim = 128
                        is_mel = False
                        if whisper_feature.dim() == 3 and whisper_feature.shape[1] == mel_dim:
                            is_mel = True
                        elif whisper_feature.dim() == 2 and whisper_feature.shape[0] == mel_dim:
                            is_mel = True

                        if is_mel:
                            # audio_tower conv layers are bfloat16 on CUDA, so cast and move
                            device = self.audio_tower.conv1.weight.device
                            whisper_feature = whisper_feature.to(device=device, dtype=torch.bfloat16)
                            audio_input = {"whisper_input_features": whisper_feature}
                            audio_inputs = self._parse_and_validate_audio_input(**audio_input)

                            # Diagnostic: print Mel spectrogram stats
                            if os.environ.get("KIMIA_DIAG_AUDIO", "0") == "1":
                                mel = whisper_feature.float()
                                print(f"    [DIAG] Mel input (buffer): shape={list(mel.shape)}, "
                                      f"mean={mel.mean():.6f}, std={mel.std():.6f}, "
                                      f"min={mel.min():.6f}, max={mel.max():.6f}")

                            self._input_whisper_emb = self._process_audio_input(audio_inputs)

                            # Diagnostic: print 3584-dim output stats
                            if os.environ.get("KIMIA_DIAG_AUDIO", "0") == "1":
                                emb = self._input_whisper_emb.float()
                                print(f"    [DIAG] After audio_tower+projector (buffer): shape={list(emb.shape)}, "
                                      f"mean={emb.mean():.6f}, std={emb.std():.6f}, "
                                      f"min={emb.min():.6f}, max={emb.max():.6f}")
                        else:
                            self._input_whisper_emb = whisper_feature
                        whisper_raw = req_info.get("whisper_raw")
                        if whisper_raw is not None and isinstance(whisper_raw, (torch.Tensor, np.ndarray)):  # noqa: UP038
                            if isinstance(whisper_raw, np.ndarray):
                                whisper_raw = torch.from_numpy(whisper_raw).cuda()
                            elif whisper_raw.device.type != "cuda":
                                whisper_raw = whisper_raw.cuda()
                            self._input_whisper_raw = whisper_raw
                        return
                    if isinstance(whisper_feature, dict) and "tensor_data" in whisper_feature:
                        try:
                            buf = bytes(whisper_feature["tensor_data"])
                            shape = tuple(whisper_feature["tensor_shape"])
                            dtype_name = whisper_feature.get("tensor_dtype", "float32")
                            dtype_map = {
                                "float32": np.float32,
                                "float16": np.float16,
                                "bfloat16": np.float16,
                                "float64": np.float64,
                            }
                            arr = np.frombuffer(buf, dtype=dtype_map.get(dtype_name, np.float32))
                            arr = arr.reshape(shape)
                            self._input_whisper_emb = torch.from_numpy(arr)
                            return
                        except Exception as e:
                            logger.warning("Failed to deserialize Whisper features: %s", e)
                            return

    def _process_audio_input(
        self, audio_input: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Process Mel spectrogram through Whisper encoder -> 4x downsample -> projector.

        Overrides upstream to add audio-length truncation, matching the reference
        Kimi-Audio Whisper encoder behavior. Without truncation, Whisper produces
        1500 frames (30s padded), yielding 375 output frames vs the expected ~47
        for a 3.7s audio.

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
            # Use arghmax-like: find the last True position
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
        """Apply token embeddings with S2S dual-stream Whisper injection."""
        is_decode = input_ids.dim() > 0 and input_ids.shape[-1] == 1
        if not is_decode and kwargs:
            self._extract_whisper_from_kwargs(kwargs)
        return self._combine_audio_text_embeds(input_ids, is_decode=is_decode)

    def _inject_whisper_into_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inject Whisper features into pre-built inputs_embeds using input_ids for positioning.

        Called from forward() when both input_ids and inputs_embeds are provided.
        The runner fills inputs_embeds with token embeddings, and we overlay
        Whisper features at audio token positions.
        """
        # Apply multi_modal_projector (VQAdaptor) to convert 5120-dim → 3584-dim
        whisper_raw = self._input_whisper_emb
        if whisper_raw.dim() == 4:
            whisper_raw = whisper_raw.reshape(1, -1, whisper_raw.shape[-1])
        elif whisper_raw.dim() == 2:
            whisper_raw = whisper_raw.unsqueeze(0)

        # Apply projector only if features are still 5120-dim
        if whisper_raw.shape[-1] == 5120:
            whisper_3584 = (
                self.multi_modal_projector(
                    whisper_raw.to(device=inputs_embeds.device, dtype=torch.bfloat16)
                )
                .squeeze(0)
                .to(inputs_embeds.device, dtype=inputs_embeds.dtype)
            )
        else:
            whisper_3584 = whisper_raw.squeeze(0).to(
                inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        # Determine audio positions from input_ids
        if input_ids is not None:
            ids = input_ids.squeeze(0) if input_ids.dim() == 2 else input_ids
            media_begin = int(self.config.kimia_media_begin)
            media_end = int(self.config.kimia_media_end)
            mb_pos = (ids == media_begin).nonzero(as_tuple=True)[0]
            me_pos = (ids == media_end).nonzero(as_tuple=True)[0]

            if len(mb_pos) > 0 and len(me_pos) > 0:
                seg_len = len(mb_pos)
                for seg_idx in range(seg_len):
                    start_pos = mb_pos[seg_idx].item()
                    end_pos = me_pos[seg_idx].item()
                    feat_len = end_pos - (start_pos + 1)
                    if feat_len > 0:
                        # Inject Whisper at audio positions
                        inputs_embeds[start_pos + 1 : end_pos] = whisper_3584[:feat_len]

                # Apply scaling: (embeds + whisper) * sqrt(2) at audio positions
                # This matches the prefill path scaling in _combine_audio_text_embeds
                audio_mask = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)
                for seg_idx in range(seg_len):
                    audio_mask[mb_pos[seg_idx].item() + 1 : me_pos[seg_idx].item()] = True

                # Apply the same (embed + whisper) * sqrt(2) scaling
                inputs_embeds[audio_mask] = inputs_embeds[audio_mask] * (2.0**0.5)

                if getattr(self, "_decode_whisper_frames", None) is None:
                    self._decode_whisper_frames = whisper_3584.clone()
                if getattr(self, "_decode_whisper_template", None) is None:
                    self._decode_whisper_template = whisper_3584.mean(dim=0, keepdim=True).clone()

        return inputs_embeds

    def _combine_audio_text_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_decode: bool = False,
        text_input_ids: torch.Tensor | None = None,
        is_continuous_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine text and audio embeddings with Whisper injection for S2S."""
        token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
        # Access embed_tokens from upstream's language_model
        embed_tokens_fn = self.language_model.model.embed_tokens

        if input_ids is None:
            # Runner path: inputs_embeds pre-built without Whisper injection.
            # Inject Whisper features using stored audio position info.
            if inputs_embeds is not None and self._input_whisper_emb is not None:
                return self._inject_whisper_into_embeds(inputs_embeds)
            return inputs_embeds if inputs_embeds is not None else None

        # Decode path
        if is_decode:
            delay_tokens = int(getattr(self.config, "kimia_mimo_audiodelaytokens", 6))
            decode_step = getattr(self, "_decode_step_counter", 0)
            use_delay_blank = decode_step < delay_tokens

            # After text EOS, vLLM feeds back the EOS token as input_ids during
            # audio-only decode steps. The reference model explicitly uses the
            # blank token (kimia_text_blank = 151666) instead. The model was
            # trained with blank tokens during audio-only decode, so feeding the
            # wrong token corrupts the combined embedding (audio_embed + text_embed)
            # that goes through the backbone to the MIMO layers.
            # Detect audio-only decode: audio codes have been generated AND the
            # current input_ids is specifically a text EOS token (not just any text).
            text_blank_index = int(getattr(self.config, "kimia_text_blank_index", 151666))
            is_audio_only_decode = False
            if not use_delay_blank and torch.cuda.is_current_stream_capturing() is False:
                if hasattr(self, "_audio_codes_list") and len(self._audio_codes_list) > 0:
                    last_id = input_ids.reshape(-1)[-1].item()
                    # Check if input is one of the text EOS tokens. During mixed
                    # decode, input is a meaningful text token (varies each step).
                    # After text EOS, input becomes the EOS token (repeated each step).
                    _text_eos_ids = {151667, 151645, 151663}  # assistant_turn, msg_end, media_end
                    if last_id in _text_eos_ids:
                        is_audio_only_decode = True

            if is_audio_only_decode:
                # Use blank token for text embedding, matching reference model
                text_ids = torch.full_like(input_ids, text_blank_index)
                text_embed = embed_tokens_fn(text_ids)
            else:
                text_embed = embed_tokens_fn(input_ids)

            if use_delay_blank:
                audio_blank_index = 18
                audio_ids = torch.full_like(input_ids, audio_blank_index + token_offset)
                audio_embed = embed_tokens_fn(audio_ids)
            elif torch.cuda.is_current_stream_capturing():
                # During CUDA graph capture, use blank audio embedding to avoid
                # graph-incompatible .item() and Python state access.
                audio_blank_index = 18
                audio_ids = torch.full_like(input_ids, audio_blank_index + token_offset)
                audio_embed = embed_tokens_fn(audio_ids)
            else:
                if hasattr(self, "_audio_codes_list") and len(self._audio_codes_list) > 0:
                    last_code = self._audio_codes_list[-1]
                    if last_code.numel() >= 1:
                        code_val = last_code.reshape(-1)[-1].item() + token_offset
                        audio_ids = torch.full_like(input_ids, code_val)
                        audio_embed = embed_tokens_fn(audio_ids)
                    else:
                        audio_blank_index = 18
                        audio_ids = torch.full_like(input_ids, audio_blank_index + token_offset)
                        audio_embed = embed_tokens_fn(audio_ids)
                else:
                    audio_blank_index = 18
                    audio_ids = torch.full_like(input_ids, audio_blank_index + token_offset)
                    audio_embed = embed_tokens_fn(audio_ids)

            return audio_embed + text_embed

        # Prefill path
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        if text_input_ids is not None and text_input_ids.dim() == 2:
            text_input_ids = text_input_ids.squeeze(0)
        if is_continuous_mask is not None and is_continuous_mask.dim() == 2:
            is_continuous_mask = is_continuous_mask.squeeze(0)

        # Use runner-provided inputs_embeds as base (already embedded via embed_input_ids)
        # instead of re-embedding input_ids, to match the runner's embedding computation
        if inputs_embeds is not None and inputs_embeds.dim() == 2:
            audio_emb = inputs_embeds.squeeze(0)
        else:
            audio_emb = embed_tokens_fn(input_ids)
        seq_len = audio_emb.shape[0]

        if text_input_ids is not None and text_input_ids.numel() > 0:
            text_emb = embed_tokens_fn(text_input_ids)
        else:
            text_emb = None

        # Derive is_continuous_mask from S2S scaffolding if not provided
        if is_continuous_mask is None:
            if self._input_whisper_emb is not None:
                media_begin = int(self.config.kimia_media_begin)
                media_end = int(self.config.kimia_media_end)
                mb_pos = (input_ids == media_begin).nonzero(as_tuple=True)[0]
                me_pos = (input_ids == media_end).nonzero(as_tuple=True)[0]
                if len(mb_pos) > 0 and len(me_pos) > 0:
                    is_continuous_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
                    is_continuous_mask[mb_pos[0].item() + 1 : me_pos[0].item()] = True
                else:
                    # Media tokens not in input_ids — derive from Whisper features.
                    # When audio_len > seq_len, expand the embedding to include
                    # audio positions after text tokens.
                    whisper_raw = self._input_whisper_emb
                    if whisper_raw.dim() == 4:
                        whisper_raw = whisper_raw.reshape(1, -1, whisper_raw.shape[-1])
                    if whisper_raw.dim() == 3:
                        audio_len = whisper_raw.shape[1]
                    elif whisper_raw.dim() == 2:
                        audio_len = whisper_raw.shape[0]
                    else:
                        audio_len = 0

                    if audio_len > 0:
                        text_len = seq_len  # Current seq_len is text-only

                        # Build proper audio_input_ids with scaffolding tokens
                        # matching the reference model's format:
                        # [text_tokens, media_begin, audio_blanks, media_end]
                        media_begin = int(self.config.kimia_media_begin)
                        media_end = int(self.config.kimia_media_end)
                        audio_blank_index = 18  # Reference blank token ID

                        audio_blanks = torch.full(
                            (audio_len,),
                            audio_blank_index + token_offset,
                            device=input_ids.device,
                            dtype=input_ids.dtype,
                        )
                        full_audio_input_ids = torch.cat([
                            input_ids,
                            torch.tensor([media_begin], device=input_ids.device, dtype=input_ids.dtype),
                            audio_blanks,
                            torch.tensor([media_end], device=input_ids.device, dtype=input_ids.dtype),
                        ])

                        # Embed the full sequence — this gives correct token
                        # embeddings for all positions including scaffolding.
                        audio_emb = embed_tokens_fn(full_audio_input_ids)
                        combined_len = len(full_audio_input_ids)
                        seq_len = combined_len

                        # Build is_continuous_mask: True only at audio_blank positions
                        is_continuous_mask = torch.zeros(seq_len, dtype=torch.bool, device=audio_emb.device)
                        is_continuous_mask[text_len + 1 : text_len + 1 + audio_len] = True

                        # Apply multi_modal_projector if needed
                        if whisper_raw.shape[-1] == 5120:
                            whisper_3584 = (
                                self.multi_modal_projector(
                                    whisper_raw.to(device=audio_emb.device, dtype=torch.bfloat16)
                                )
                                .squeeze(0)
                                .to(audio_emb.device, dtype=audio_emb.dtype)
                            )
                        else:
                            whisper_3584 = whisper_raw.squeeze(0).to(audio_emb.device, dtype=audio_emb.dtype)

                        # Inject Whisper: overlay onto audio_blank embeddings
                        # using the same (audio_emb + whisper) * sqrt(2) scaling
                        audio_mask_3d = is_continuous_mask.to(audio_emb.dtype).unsqueeze(-1)
                        whisper_expanded = torch.zeros_like(audio_emb)
                        whisper_expanded[text_len + 1 : text_len + 1 + audio_len] = whisper_3584
                        encoder_input = (audio_emb + whisper_expanded) * (2.0**0.5)
                        audio_emb = audio_emb * (~audio_mask_3d.to(torch.bool)) + encoder_input * audio_mask_3d.to(torch.bool)

                        # Store for position ID override
                        self._prefill_audio_offset = audio_len

                        return audio_emb if text_emb is None else audio_emb + text_emb
                    else:
                        logger.warning(
                            "_combine: Cannot derive audio_len from whisper shape=%s",
                            list(whisper_raw.shape),
                        )
                        self._prefill_audio_offset = 0
                        return audio_emb if text_emb is None else audio_emb + text_emb
            else:
                # No whisper features — no audio prefill
                self._prefill_audio_offset = 0
                return audio_emb if text_emb is None else audio_emb + text_emb

        # Inject Whisper features at audio positions
        if self._input_whisper_emb is not None:
            whisper_raw = self._input_whisper_emb
            if whisper_raw.dim() == 4:
                whisper_raw = whisper_raw.reshape(1, -1, whisper_raw.shape[-1])
            elif whisper_raw.dim() == 2:
                whisper_raw = whisper_raw.unsqueeze(0)

            # Apply multi_modal_projector only if features are still 5120-dim
            # (e.g., from realtime buffer). Test scripts and runner may already
            # provide 3584-dim post-VQAdaptor features.
            if whisper_raw.shape[-1] == 5120:
                whisper_3584 = (
                    self.multi_modal_projector(
                        whisper_raw.to(device=audio_emb.device, dtype=torch.bfloat16)
                    )
                    .squeeze(0)
                    .to(audio_emb.device, dtype=audio_emb.dtype)
                )
            else:
                whisper_3584 = whisper_raw.squeeze(0).to(
                    audio_emb.device, dtype=audio_emb.dtype
                )

            media_begin = int(self.config.kimia_media_begin)
            media_end = int(self.config.kimia_media_end)
            media_start_pos = (input_ids == media_begin).nonzero(as_tuple=True)[0]
            media_end_pos = (input_ids == media_end).nonzero(as_tuple=True)[0]

            expanded_whisper = torch.zeros(seq_len, audio_emb.shape[-1], device=audio_emb.device, dtype=audio_emb.dtype)
            for seg_idx in range(len(media_start_pos)):
                start_pos = media_start_pos[seg_idx].item()
                end_pos = media_end_pos[seg_idx].item()
                feat_len = end_pos - (start_pos + 1)
                if feat_len > 0:
                    expanded_whisper[start_pos + 1 : end_pos, :] = whisper_3584[:feat_len, :]

            is_continuous_mask_3d = is_continuous_mask.to(audio_emb.dtype).unsqueeze(-1)
            whisper_masked = expanded_whisper * is_continuous_mask_3d
            encoder_input = (audio_emb + whisper_masked) * (2.0**0.5)
            audio_emb = audio_emb * (~is_continuous_mask_3d.to(torch.bool)) + encoder_input * is_continuous_mask_3d.to(
                torch.bool
            )

            # Track the actual number of audio tokens (not full seq_len).
            # The reference model's position IDs during prefill only count audio
            # tokens (audio_input_ids.shape[1]), not scaffolding tokens.
            # Using full seq_len causes RoPE position mismatch during decode:
            # reference queries position N (audio count), vLLM queries N+6
            # (seq_len including scaffolding), causing MIMO to attend to wrong
            # KV cache entries and produce incorrect audio logits.
            audio_token_count = is_continuous_mask.sum().item()
            self._prefill_audio_offset = int(audio_token_count)

            if getattr(self, "_decode_whisper_frames", None) is None:
                self._decode_whisper_frames = whisper_3584.clone()
            if getattr(self, "_decode_whisper_template", None) is None:
                self._decode_whisper_template = whisper_3584.mean(dim=0, keepdim=True).clone()

            if os.environ.get("KIMIA_DIAG_EMBED", "0") == "1":
                result = audio_emb if text_emb is None else audio_emb + text_emb
                torch.save(result.detach().cpu(), "/tmp/vllm_embed_output.pt")
                # Save intermediates for debugging
                torch.save(
                    {
                        "audio_emb": audio_emb.detach().cpu(),
                        "text_emb": text_emb.detach().cpu() if text_emb is not None else None,
                        "whisper_3584": whisper_3584.detach().cpu(),
                        "is_continuous_mask": is_continuous_mask.detach().cpu(),
                        "input_ids": input_ids.detach().cpu(),
                        "text_input_ids": text_input_ids.detach().cpu() if text_input_ids is not None else None,
                    },
                    "/tmp/vllm_embed_intermediates.pt",
                )
                return result

            return audio_emb if text_emb is None else audio_emb + text_emb

        # Fallback: no whisper features (TTS mode or non-S2S).
        # Set audio offset to 0 — no prefill audio tokens to account for.
        self._prefill_audio_offset = 0
        return audio_emb if text_emb is None else audio_emb + text_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | OmniOutput:
        """Forward pass with bifurcation at layer 21 for MIMO branch."""
        token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
        # Detect prefill: either multi-token input_ids OR multi-token inputs_embeds.
        # S2S prefill uses inputs_embeds (not input_ids), so we must check both.
        # Priority: input_ids shape (correct) > inputs_embeds shape[0] (2D runner path).
        if input_ids is not None and input_ids.numel() > 0:
            _seq_len = input_ids.shape[-1] if input_ids.dim() > 0 else 0
        elif inputs_embeds is not None and inputs_embeds.dim() >= 1:
            _seq_len = inputs_embeds.shape[0] if inputs_embeds.dim() == 2 else inputs_embeds.shape[1]
        else:
            _seq_len = 0
        is_prefill = _seq_len > 1
        is_profile_run = _seq_len > 1000

        # Reset state at start of new request
        if is_prefill:
            self._decode_step_counter = 0
            self._sample_step_counter = 0
            self._audio_codes_list = []
            self._audio_eos_triggered = False
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

        if is_prefill and kwargs:
            self._extract_whisper_from_kwargs(kwargs)

        # Embed input on first rank
        is_decode = not is_prefill

        if get_pp_group().is_first_rank:
            hidden_states = self._combine_audio_text_embeds(
                input_ids,
                inputs_embeds,
                is_decode=is_decode,
                text_input_ids=kwargs.get("text_input_ids"),
                is_continuous_mask=kwargs.get("is_continuous_mask"),
            )

            # Override positions for combined sequence during prefill
            if is_prefill and getattr(self, "_prefill_audio_offset", 0) > 0:
                # Use actual hidden_states sequence length for positions
                if hidden_states.dim() == 3:
                    full_seq_len = hidden_states.shape[1]
                elif hidden_states.dim() == 2:
                    full_seq_len = hidden_states.shape[0]
                else:
                    audio_offset = self._prefill_audio_offset
                    _text_len = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[1]
                    full_seq_len = max(audio_offset, _text_len)
                positions = torch.arange(full_seq_len, device=positions.device, dtype=positions.dtype)
                if hidden_states.dim() == 3 and positions.dim() == 1:
                    positions = positions.unsqueeze(0)

            if is_prefill:
                # Use actual hidden_states length when expanded (S2S with Whisper)
                if hidden_states.dim() == 3:
                    self._prefill_seq_len = hidden_states.shape[1]
                elif hidden_states.dim() == 2:
                    self._prefill_seq_len = hidden_states.shape[0]
                else:
                    _text_len_for_seq = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[1]
                    self._prefill_seq_len = max(getattr(self, "_prefill_audio_offset", 0), _text_len_for_seq)
                _text_len_for_seq = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[1]
                self._prefill_text_len = _text_len_for_seq
            elif not is_prefill and getattr(self, "_prefill_seq_len", 0) > 0:
                pos_delta = self._prefill_seq_len - self._prefill_text_len
                positions = positions + pos_delta
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # Run backbone via self.language_model.model with bifurcation hook at layer 21
        # to capture mimo_hidden_states
        bifurcation_idx = self._bifurcation_layer

        # Register hook on bifurcation layer — needed on all ranks that process it
        # to populate mimo_hidden_states for intermediate tensor forwarding
        mimo_hidden_states = None
        hook_handle = None
        bifurcation_layer = self.language_model.model.layers[bifurcation_idx]

        def _bifurcation_hook(module, input, output):
            nonlocal mimo_hidden_states
            # Qwen2DecoderLayer returns (hidden_states, residual) where
            # hidden_states is the raw MLP output and residual carries the
            # skip connection. The actual layer output is hidden_states + residual.
            # Match the reference model's behavior where MIMO layers receive the
            # complete layer output (mlp_out + residual), not the raw MLP output.
            if isinstance(output, tuple) and len(output) >= 2:
                hs = output[0]
                res = output[1]
                mimo_hidden_states = (hs + res).clone()
            else:
                hs = output[0] if isinstance(output, tuple) else output
                mimo_hidden_states = hs.clone()
            # Capture bifurcation output for debugging
            if os.environ.get("CAPTURE_MIMO", "0") == "1":
                capture_dir = os.environ.get("CAPTURE_MIMO_DIR", "/tmp/mimo_capture")
                os.makedirs(capture_dir, exist_ok=True)
                torch.save({
                    "hidden_states": hs.detach().float().cpu(),
                }, os.path.join(capture_dir, "bifurcation_output.pt"))
            # DIAG: Log bifurcation stats
            _log_mimo = os.environ.get("DIAG_MIMO", "0") == "1"
            if _log_mimo:
                _bif_std = mimo_hidden_states.float().std().item()
                _bif_shape = list(mimo_hidden_states.shape)
                _bif_nonzero = (mimo_hidden_states.abs() > 1e-6).sum().item()
                logger.info(f"[BIFURC DIAG] shape={_bif_shape} std={_bif_std:.4f} nonzero={_bif_nonzero}/{mimo_hidden_states.numel()}")

        hook_handle = bifurcation_layer.register_forward_hook(_bifurcation_hook)

        try:
            # Delegate to self.language_model.model — bypass __call__ to avoid
            # torch.compile tracing diagnostic hooks in scripts like compare_layer_by_layer
            result = self.language_model.model.forward(
                input_ids=input_ids if not get_pp_group().is_first_rank else None,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=hidden_states if get_pp_group().is_first_rank else None,
            )

            # Non-last ranks: return IntermediateTensors
            if not get_pp_group().is_last_rank:
                # Include mimo_hidden_states if captured at bifurcation
                hs_tensor = result["hidden_states"]
                res = result.get("residual")
                tensors = {"hidden_states": hs_tensor}
                if mimo_hidden_states is not None:
                    tensors["mimo_hidden_states"] = mimo_hidden_states
                if res is not None:
                    tensors["residual"] = res
                return IntermediateTensors(tensors)

            # self.language_model.model already applied self.norm — result IS normed
            # Result may be tuple (hidden_states, aux_hidden_states) if EAGLE enabled
            if isinstance(result, tuple):
                hidden_states = result[0]
            else:
                hidden_states = result
            text_hidden_states = hidden_states
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        # Run MIMO layers on the bifurcated hidden states. During prefill, MIMO
        # processes audio scaffolding tokens (ID 152082 = blank 18 + offset 152064)
        # whose hidden states already contain Whisper-injected features from the
        # backbone. This populates the MIMO KV cache with Whisper-informed audio
        # context, matching the reference model's behavior. During decode, MIMO
        # runs on newly generated audio codes and attends to both prefill cache
        # entries and previously generated codes.
        # Skip during profile/warmup runs to avoid populating the KV cache
        # with dummy (zero) values that corrupt actual inference. Also skip
        # if the input is all-zeros (warmup decode before actual prefill).
        _has_real_prefill = getattr(self, "_prefill_audio_offset", 0) > 0
        if mimo_hidden_states is not None and not is_profile_run and _has_real_prefill:
            # Use backbone position IDs for MIMO layers in both prefill and decode.
            # The bifurcation hidden states come from backbone layer 21 at the
            # current positions. MIMO must use the same position IDs to ensure
            # RoPE embeddings match between the backbone and MIMO attention.
            audio_offset = getattr(self, "_prefill_audio_offset", 0)
            if not is_prefill and audio_offset > 0:
                # Decode: use backbone position for MIMO to match RoPE embedding.
                # The bifurcation hidden states come from backbone layer 21 at
                # the current decode position. MIMO must use the same position
                # IDs as the backbone to ensure RoPE consistency.
                mimo_positions = positions
            else:
                # Prefill: use backbone positions for MIMO layers.
                # The MIMO layers process all tokens through bifurcation, but
                # only audio token positions matter for the KV cache.
                mimo_positions = positions

            # === CAPTURE HOOK: save MIMO layer I/O for debugging ===
            _capture_mimo = os.environ.get("CAPTURE_MIMO", "0") == "1"
            _mimo_capture_dir = os.environ.get("CAPTURE_MIMO_DIR", "/tmp/mimo_capture")
            if _capture_mimo:
                os.makedirs(_mimo_capture_dir, exist_ok=True)
            for layer_idx, mimo_layer in enumerate(self.mimo_layers):
                # Pass None as residual to match reference model behavior.
                # The reference creates a fresh residual from each layer's
                # input (residual = hidden_states before norm). vLLM's fused
                # residual carrying pattern accumulates residuals across layers,
                # which changes the norm statistics and produces different output.
                # With residual=None, Qwen2DecoderLayer does:
                #   residual = hidden_states (this layer's own input)
                #   hidden_states = input_layernorm(hidden_states)
                #   hidden_states = self_attn(hidden_states)
                #   hidden_states, residual = post_attention_layernorm(hidden_states, residual)
                #   hidden_states = mlp(hidden_states)
                #   return hidden_states, residual
                # This matches the reference: output = mlp(attn(norm(input))) + input
                # Pass None as residual to match reference model behavior.
                # The reference creates a fresh residual from each layer's
                # input (residual = hidden_states before norm). vLLM's fused
                # residual carrying pattern accumulates residuals across layers,
                # which changes the norm statistics and produces different output.
                if _capture_mimo:
                    torch.save({
                        "input": mimo_hidden_states.detach().float().cpu(),
                        "positions": mimo_positions.detach().cpu(),
                    }, os.path.join(_mimo_capture_dir, f"mimo_layer_{layer_idx}_input.pt"))

                # DIAG: Log MIMO layer stats for both prefill and decode
                _log_mimo = os.environ.get("DIAG_MIMO", "0") == "1"
                if _log_mimo and layer_idx == 0:
                    _mimo_in_std = mimo_hidden_states.float().std().item()
                    _mimo_shape = list(mimo_hidden_states.shape)
                    _mimo_nonzero = (mimo_hidden_states.abs() > 1e-6).sum().item()
                    logger.info(
                        "[MIMO DIAG] prefill=%s layer=%d shape=%s input_std=%.4f nonzero=%d",
                        is_prefill, layer_idx, _mimo_shape, _mimo_in_std, _mimo_nonzero,
                    )

                # Granular capture: hook self_attn and mlp outputs
                if _capture_mimo:
                    _orig_attn = mimo_layer.self_attn.forward
                    _orig_mlp = mimo_layer.mlp.forward
                    def _wrapped_attn(*args, **kwargs):
                        out = _orig_attn(*args, **kwargs)
                        torch.save({
                            "attn_output": out.detach().float().cpu(),
                            "positions": mimo_positions.detach().cpu(),
                        }, os.path.join(_mimo_capture_dir, f"mimo_layer_{layer_idx}_attn_output.pt"))
                        return out
                    def _wrapped_mlp(*args, **kwargs):
                        out = _orig_mlp(*args, **kwargs)
                        torch.save({
                            "mlp_output": out.detach().float().cpu(),
                        }, os.path.join(_mimo_capture_dir, f"mimo_layer_{layer_idx}_mlp_output.pt"))
                        return out
                    mimo_layer.self_attn.forward = _wrapped_attn
                    mimo_layer.mlp.forward = _wrapped_mlp

                mimo_hidden_states, _ = mimo_layer(mimo_positions, mimo_hidden_states, None)

                # DIAG: Log output stats
                if _log_mimo:
                    _mimo_out_std = mimo_hidden_states.float().std().item()
                    _mimo_out_mean = mimo_hidden_states.float().mean().item()
                    logger.info(f"[MIMO DIAG] prefill={is_prefill} layer={layer_idx} output_std={_mimo_out_std:.4f} output_mean={_mimo_out_mean:.6f}")

                # Restore original forward
                if _capture_mimo:
                    mimo_layer.self_attn.forward = _orig_attn
                    mimo_layer.mlp.forward = _orig_mlp

                if _capture_mimo:
                    torch.save({
                        "output": mimo_hidden_states.detach().float().cpu(),
                    }, os.path.join(_mimo_capture_dir, f"mimo_layer_{layer_idx}_output.pt"))
            if _capture_mimo:
                torch.save({
                    "mimo_norm_input": mimo_hidden_states.detach().float().cpu(),
                }, os.path.join(_mimo_capture_dir, "before_norm.pt"))

            mimo_hidden_states = self.mimo_norm(mimo_hidden_states)
            if _capture_mimo:
                torch.save({
                    "mimo_norm_output": mimo_hidden_states.detach().float().cpu(),
                }, os.path.join(_mimo_capture_dir, "after_norm.pt"))

            audio_logits_out = self.mimo_output(mimo_hidden_states)
            if _capture_mimo:
                torch.save({
                    "mimo_logits": audio_logits_out.detach().float().cpu() if isinstance(audio_logits_out, torch.Tensor) else audio_logits_out[0].detach().float().cpu(),
                }, os.path.join(_mimo_capture_dir, "mimo_logits.pt"))
            if isinstance(audio_logits_out, tuple):
                audio_logits = audio_logits_out[0]
            else:
                audio_logits = audio_logits_out

            # Extract audio logits subspace (audio codes only, starting at
            # token_offset). The reference model stops on EOS tokens but those
            # are in the text vocab space and never have high probability in
            # the audio logits — use _max_audio_codes cap instead.
            audio_vocab = int(getattr(self.config, "kimia_audio_output_vocab", 16384))
            if audio_logits.dim() == 2:
                audio_logits_sub = audio_logits[:, token_offset : token_offset + audio_vocab]  # [batch, audio_vocab]
            elif audio_logits.dim() == 3:
                audio_logits_sub = audio_logits[:, :, token_offset : token_offset + audio_vocab]  # [batch, seq, audio_vocab]
            else:
                raise ValueError(f"Unexpected audio_logits dim: {audio_logits.dim()}, shape: {audio_logits.shape}")

            # Store full unsliced logits for audio EOS detection in sample()
            self._full_audio_logits = audio_logits

            # Return audio_logits via multimodal_outputs so they are
            # updated during CUDA graph replay (return values are part of
            # the graph, instance attributes are not). The runner extracts
            # them and passes to model.sample().
            self._audio_logits = audio_logits_sub

            if is_profile_run:
                self._audio_codes = torch.empty(0, dtype=torch.long)
                self._audio_codes_list = []
                self._decode_step_counter = 0
                self._sample_step_counter = 0
                self._text_codes = torch.empty(0, dtype=torch.long)
                self._full_audio_logits = None
                multimodal_outputs = {
                    "audio_codes": torch.empty(0, dtype=torch.long),
                    "audio_logits": audio_logits_sub,
                    "text_codes": None,
                }
            else:
                # Audio codes are sampled in sample() outside CUDA graph.
                # Return audio_logits for the runner to pass to sample().
                # Only increment during decode, not prefill — prefill populates
                # the KV cache at positions [0, ..., _prefill_seq_len-1], and
                # decode should start at position _prefill_seq_len (not +1).
                if not is_prefill:
                    self._decode_step_counter += 1
                multimodal_outputs = {
                    "audio_codes": torch.empty(0, dtype=torch.long),
                    "audio_logits": audio_logits_sub,
                    "text_codes": None,
                }
        else:
            # Prefill: no MIMO computation, or mimo_hidden_states not captured
            # (non-bifurcation rank). Return empty multimodal_outputs.
            multimodal_outputs = {"audio_codes": None, "text_codes": None}

        # Final return
        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
        audio_logits: torch.Tensor | None = None,
    ) -> None:
        """Sample audio codes from audio logits."""
        if audio_logits is None or audio_logits.numel() == 0:
            return

        delay_tokens = int(getattr(self.config, "kimia_mimo_audiodelaytokens", 6))

        # Track sample steps independently of forward() since CUDA graph
        # replay doesn execute Python, so _decode_step_counter stays stale.
        step = getattr(self, "_sample_step_counter", 0)
        self._sample_step_counter = step + 1

        # Skip sampling during delay token period — the reference model does
        # NOT include delay blank codes in the output (slices them out via
        # previous_audio_tokens[audio_delay_tokens:audio_delay_tokens+valid_length]).
        # Don't append blank codes to the output list.
        if step < delay_tokens:
            return

        # Check max codes BEFORE sampling — the reference model stops appending
        # audio codes once the limit is reached. If we sample first then check,
        # _audio_codes will exceed the limit and the scheduler will read the
        # inflated count, causing extra decode steps.
        if len(self._audio_codes_list) >= self._max_audio_codes:
            return

        # Check for audio EOS — if the model predicted an EOD token in the
        # full logits, stop generating audio codes (same as reference eod_ids
        # check: audio_stream_is_finished = next_audio_token.item() in eod_ids).
        if not getattr(self, "_audio_eos_triggered", False):
            full_logits = getattr(self, "_full_audio_logits", None)
            if full_logits is not None:
                # Take last position: [batch, vocab] or [vocab]
                if full_logits.dim() == 3:
                    last_logits = full_logits[:, -1, :]
                elif full_logits.dim() == 2:
                    last_logits = full_logits
                else:
                    last_logits = full_logits.view(1, -1)

                # Check if argmax is an EOD token
                top_token = last_logits.argmax(dim=-1).item()
                if top_token in self._audio_eod_ids:
                    logger.info(
                        f"[Kimia-Audio] EOS triggered at step {step}: "
                        f"top_token={top_token} in eod_ids={self._audio_eod_ids}, "
                        f"generated {len(self._audio_codes_list)} audio codes"
                    )
                    self._audio_eos_triggered = True
                    return

        # Stop generating if EOS was triggered
        if getattr(self, "_audio_eos_triggered", False):
            return

        # Sample from audio logits using the same params as the reference
        temperature = float(getattr(self.config, "kimia_audio_temperature", 0.8))
        top_k = int(getattr(self.config, "kimia_audio_top_k", 10))
        repetition_penalty = float(getattr(self.config, "kimia_audio_repetition_penalty", 1.0))

        # Gather recent tokens for repetition penalty
        recent_tokens = None
        if len(self._audio_codes_list) > 0:
            recent_tokens = torch.cat(self._audio_codes_list)

        # Handle 3D logits [batch, seq, vocab] — take last position
        if audio_logits.dim() == 3:
            step_logits = audio_logits[:, -1, :]
        else:
            step_logits = audio_logits

        # DIAG: Log audio logits distribution statistics at key steps
        _diag_logits = os.environ.get("DIAG_MIMO", "0") == "1"
        if _diag_logits:
            _logits_fp32 = step_logits.float()
            _logits_shape = list(_logits_fp32.shape)
            _top5_vals, _top5_idx = torch.topk(_logits_fp32, 5, dim=-1)
            _logits_std = _logits_fp32.std().item()
            _logits_entropy = -torch.softmax(_logits_fp32, dim=-1) * torch.log_softmax(_logits_fp32, dim=-1)
            _entropy_val = _logits_entropy.sum(dim=-1).item()
            logger.info(
                f"[LOGITS DIAG] step={step+1} shape={_logits_shape} logits_std={_logits_std:.2f} entropy={_entropy_val:.2f} "
                f"top5_codes={_top5_idx[0].tolist()} top5_vals={_top5_vals[0].tolist()}"
            )

        sampled_code = _sample_audio_topk(
            step_logits,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            recent_tokens=recent_tokens,
        )

        if _diag_logits:
            logger.info(f"[LOGITS DIAG] step={step+1} sampled_code={sampled_code.item()}")

        # sampled_code is [batch] or scalar — flatten to 1D
        sampled_code = sampled_code.reshape(-1)

        # Clamp to valid audio vocab range
        audio_vocab = int(getattr(self.config, "kimia_audio_output_vocab", 16384))
        sampled_code = sampled_code.clamp(0, audio_vocab - 1)

        self._audio_codes_list.append(sampled_code)
        self._audio_codes = torch.cat(self._audio_codes_list)

        # Log audio code diversity at key intervals
        n = len(self._audio_codes_list)
        if n <= 3 or n % 20 == 0 or n >= 78:
            codes_list = [c.item() for c in self._audio_codes_list]
            unique = len(set(codes_list))
            logger.info(f"[AUDIO CODES] step={n}: unique={unique}/{n}, last5={codes_list[-5:]}")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Compute text logits, masking audio subspace."""
        logits = self.logits_processor(self.language_model.lm_head, hidden_states)
        if logits is not None:
            token_offset = int(getattr(self.config, "kimia_token_offset", 152064))
            logits[:, token_offset:] = float("-inf")
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights: upstream handles backbone, we handle MIMO."""
        # Materialize the weights iterator so we can iterate it twice
        # (once for super(), once for MIMO loading).
        weights_list = list(weights)

        # Call upstream to load backbone + lm_head + VQAdaptor + Whisper
        loaded = super().load_weights(iter(weights_list))

        # Load MIMO-specific weights from checkpoint if present
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        def _is_stacked_key(name):
            for _, wname, _ in stacked_params_mapping:
                if wname in name:
                    return True
            return False

        params_dict = dict(self.named_parameters())

        # Pass 1: Load non-stacked MIMO params (layernorms, o_proj, down_proj,
        # biases). Skip stacked params — they are handled in Pass 2.
        for name, loaded_weight in weights_list:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mimo_layers" not in name and "mimo_norm" not in name and "mimo_output" not in name:
                continue

            # Strip "model." prefix — most checkpoint keys have it but vLLM
            # params_dict may not. Some keys (like mimo_output.weight) lack the
            # prefix in the checkpoint but have it in vLLM params.
            param_name = name[len("model.") :] if name.startswith("model.") else name

            # Skip stacked params — collected and concatenated in Pass 2
            if _is_stacked_key(name):
                continue

            # Try with and without "model." prefix
            param = params_dict.get(param_name)
            if param is None and not param_name.startswith("model."):
                param = params_dict.get(f"model.{param_name}")
                if param is not None:
                    param_name = f"model.{param_name}"
            if param is None:
                continue

            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader:
                weight_loader(param, loaded_weight)
            else:
                param.data.copy_(loaded_weight)
            loaded.add(param_name)

        # Pass 2: Collect and concatenate stacked MIMO param shards (q/k/v →
        # qkv_proj, gate/up → gate_up_proj)
        stacked_shards: dict[str, dict[str, torch.Tensor]] = {}
        for name, loaded_weight in weights_list:
            if "mimo_layers" not in name:
                continue
            if not _is_stacked_key(name):
                continue

            # Strip "model." prefix — checkpoint keys have it but vLLM
            # params_dict doesn't
            param_name = name[len("model.") :] if name.startswith("model.") else name
            for pname, wname, sid in stacked_params_mapping:
                if wname not in param_name:
                    continue
                mapped_name = param_name.replace(wname, pname)
                if mapped_name not in params_dict:
                    continue
                if mapped_name not in stacked_shards:
                    stacked_shards[mapped_name] = {}
                stacked_shards[mapped_name][str(sid)] = loaded_weight
                break

        for mapped_name, shards in stacked_shards.items():
            param = params_dict[mapped_name]
            parts = mapped_name.split(".")
            key = parts[-2] if len(parts) >= 2 else parts[-1]
            default_order = {"qkv_proj": ["q", "k", "v"], "gate_up_proj": ["0", "1"]}
            order = default_order.get(key, sorted(shards.keys()))
            sorted_shards = [shards[k] for k in order if k in shards]
            if sorted_shards:
                concat_weight = torch.cat(sorted_shards, dim=0)
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader:
                    weight_loader(param, concat_weight)
                else:
                    param.data.copy_(concat_weight)
                loaded.add(mapped_name)

        # Mark remaining MIMO params as handled (randomly initialized if not
        # in checkpoint) so vLLM doesn't error on uninitialized weights.
        for name in params_dict:
            if "mimo_layers" in name or "mimo_norm" in name or "mimo_output" in name:
                loaded.add(name)

        return loaded


# Register with vLLM's own MODEL_REGISTRY so the V1 engine can resolve
# the architecture without falling back to TransformersForCausalLM.
def _register_with_vllm_registry() -> None:
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        ModelRegistry.register_model(
            "KimiAudioFusedForConditionalGeneration",
            "vllm_omni.model_executor.models.kimia_audio.kimia_audio_ar_stage:KimiAudioFusedForConditionalGeneration",
        )
    except Exception:
        pass  # Best-effort registration; may fail in limited contexts


_register_with_vllm_registry()
del _register_with_vllm_registry
