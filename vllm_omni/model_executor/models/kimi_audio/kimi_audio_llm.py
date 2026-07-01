# Copyright 2025 vLLM-Omni Team
"""Stage 0: Kimi Audio LLM with bifurcation for dual output (text + audio)."""

from typing import Any, Optional

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm.logger import init_logger

# Import custom processor for Kimi Audio
from vllm_omni.model_executor.models.kimi_audio.custom_processor import (
    CustomKimiAudioMultiModalProcessor,
)
from vllm.model_executor.models.kimi_audio import (
    KimiAudioProcessingInfo,
    KimiAudioDummyInputsBuilder,
)

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Monkey-patch the Kimi tokenizer to fix initialization order
# This must be done before the tokenizer is loaded
def _patch_kimi_tokenizer():
    """Patch Kimi tokenizer to initialize _special_tokens_map before setting special tokens."""
    try:
        import transformers_modules
        import importlib
        import sys

        # Try to import the tokenizer module
        for module_name in list(sys.modules.keys()):
            if 'tokenization_kimia' in module_name:
                module = sys.modules[module_name]
                if hasattr(module, 'TikTokenTokenizer'):
                    # Get the original __init__
                    original_init = module.TikTokenTokenizer.__init__

                    # Create a wrapper that initializes _special_tokens_map first
                    def patched_init(self, vocab_file, **kwargs):
                        # Initialize _special_tokens_map before calling original __init__
                        self._special_tokens_map = {}
                        return original_init(self, vocab_file, **kwargs)

                    # Replace the __init__
                    module.TikTokenTokenizer.__init__ = patched_init
                    break
    except Exception:
        # If patching fails, continue anyway
        pass

# Apply the patch at module load time
_patch_kimi_tokenizer()


# Monkey-patch the output length calculation to match reference implementation
def _patch_output_length_calculation():
    """Patch the output length calculation to match reference implementation."""
    try:
        import vllm.model_executor.models.kimi_audio as kimi_audio_module

        # Save original function
        original_func = kimi_audio_module._get_feat_extract_output_lengths

        def _custom_get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
            """Custom output length calculation matching reference implementation.

            Reference: token_len = (L - 1) // (160 * 8) + 1, then * 4
            where L is audio length in samples.
            Whisper produces 100 mel frames per second (16000 samples / 160 hop = 100 frames).
            So L = input_lengths * 160 (convert mel frames to samples).
            """
            # input_lengths is in mel frames, convert to audio samples
            L = input_lengths * 160
            # Reference formula
            token_len = (L - 1) // (160 * 8) + 1
            return token_len * 4

        # Replace the function
        kimi_audio_module._get_feat_extract_output_lengths = _custom_get_feat_extract_output_lengths

        logger.debug("Patched _get_feat_extract_output_lengths to match reference implementation")
    except Exception as e:
        logger.warning("Failed to patch output length calculation: %s", e)


# Apply the patch
_patch_output_length_calculation()


class KimiAudioCustomWhisperEncoder(nn.Module):
    """Custom Whisper encoder that matches reference implementation.

    Key differences from standard Whisper:
    1. Processes audio in 30-second chunks
    2. Slices encoder output to token_len * 4
    3. token_len = (L - 1) // (160 * 8) + 1 where L is audio samples
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm.model_executor.models.kimi_audio import KimiAudioWhisperEncoder

        # Use the upstream encoder as base
        self.encoder = KimiAudioWhisperEncoder(
            vllm_config=vllm_config,
            prefix=prefix,
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Forward pass with custom slicing logic.

        Args:
            input_features: [B, 128, T] mel spectrogram

        Returns:
            features: [B, token_len * 4, hidden_dim] sliced features
        """
        # input_features shape: [B, 128, T] where T is number of mel frames
        batch_size, num_mel_bins, num_frames = input_features.shape

        # Run through standard encoder
        # Output: [B, num_frames // 2, hidden_dim] (due to stride-2 conv)
        encoder_output = self.encoder(input_features)

        # Calculate token_len based on reference implementation
        # Reference: token_len = (L - 1) // (160 * 8) + 1
        # where L is audio length in samples
        # Whisper uses 10ms hop (160 samples at 16kHz), so:
        # L = num_frames * 160 (approximate)
        L = num_frames * 160
        token_len = (L - 1) // (160 * 8) + 1
        target_length = token_len * 4

        # Slice or pad to target length
        actual_length = encoder_output.shape[1]

        if actual_length >= target_length:
            # Slice to target length
            encoder_output = encoder_output[:, :target_length, :]
        else:
            # Pad with zeros if needed
            padding = torch.zeros(
                batch_size,
                target_length - actual_length,
                encoder_output.shape[2],
                dtype=encoder_output.dtype,
                device=encoder_output.device,
            )
            encoder_output = torch.cat([encoder_output, padding], dim=1)

        return encoder_output


@MULTIMODAL_REGISTRY.register_processor(
    CustomKimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder,
)
class KimiAudioLLMForConditionalGeneration(nn.Module, SupportsMultiModal):
    """Stage 0: Shared backbone → bifurcation → text + audio logits.

    Architecture:
    - Layers 0-21: Shared backbone (from Qwen2)
    - Bifurcation at layer 21: Clone hidden states
    - Text path: Layers 22-27 → lm_head (text logits)
    - Audio path: 6 MIMO layers → mimo_norm → mimo_output (audio logits)
    """

    # Mark as generative model so vllm's runner validation passes
    is_text_generation_model = True
    # Mark as producing multimodal outputs (audio logits for Stage 1)
    have_multimodal_outputs = True

    # Dual streaming extension points (from HiggsAudioV2 pattern)
    prefer_model_sampler = True  # Use custom sampling for dual streams
    has_postprocess = True  # Enable per-request state sync
    postprocess_uses_hidden_states = True
    postprocess_uses_multimodal_outputs = True
    postprocess_uses_req_infos = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config

        # Import upstream vllm components
        from vllm.model_executor.models.kimi_audio import (
            KimiAudioMultiModalProjector,
        )

        # Use custom Whisper encoder that matches reference implementation
        self.audio_tower = KimiAudioCustomWhisperEncoder(
            vllm_config=vllm_config,
            prefix=f"{prefix}model.audio_tower",
        )

        # Audio input processing (reuse from upstream vllm)
        # Note: audio_tower is already set above, don't overwrite

        # Project Whisper output (1280) to VQ adaptor input (5120)
        # Get Whisper output dimension from the whisper config subfolder
        from transformers import WhisperConfig
        import os
        model_path = vllm_config.model_config.model
        whisper_config_path = os.path.join(model_path, "whisper-large-v3")
        if os.path.exists(whisper_config_path):
            whisper_cfg = WhisperConfig.from_pretrained(whisper_config_path)
            whisper_output_dim = whisper_cfg.d_model
        else:
            whisper_output_dim = 1280  # Default for Whisper Large v3

        adaptor_input_dim = getattr(self.config, "kimia_adaptor_input_dim", 5120)
        if whisper_output_dim != adaptor_input_dim:
            self.whisper_projection = nn.Linear(whisper_output_dim, adaptor_input_dim)
        else:
            self.whisper_projection = None

        self.multi_modal_projector = KimiAudioMultiModalProjector(
            whisper_dim=adaptor_input_dim,  # Whisper encoder output dim after projection
            llm_dim=self.config.hidden_size,  # LLM input dim
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )

        # Qwen2 backbone (layers 0-27)
        # Use "language_model" prefix to match upstream vLLM's WeightsMapper
        self.model = init_vllm_registered_model(
            vllm_config.with_hf_config(self.config, architectures=["Qwen2ForCausalLM"]),
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # NEW: MIMO layers (6 layers) - audio-specific transformer layers
        # These reuse the Qwen2 decoder layer structure
        from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

        # Get cache_config and quant_config from vllm_config for proper KV caching
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.mimo_layers = nn.ModuleList([
            Qwen2DecoderLayer(
                config=self.config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"mimo_layers.{i}"),
            )
            for i in range(self.config.kimia_mimo_layers)  # 6 layers
        ])

        # NEW: Audio output head
        from vllm.model_executor.layers.layernorm import RMSNorm
        self.mimo_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps
        )

        # Audio output projection (supports tensor parallelism)
        # Tied with lm_head - same weights
        self.mimo_output = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.vocab_size,  # 168448 - full vocab including text and audio
            gather_output=True,  # Gather across TP ranks
            bias=False,  # No bias, matching checkpoint
        )

        # Text logits processor
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=1.0,
        )

        # Dual streaming state (per-slot management following HiggsAudioV2 pattern)
        # These are lazily initialized in sample() to avoid issues with distributed setup
        # self._audio_state: dict[int, dict[str, Any]] = {}
        # self._slot_output_len: dict[int, int] = {}
        # self._text_stream_finished: dict[int, bool] = {}
        self._pending_audio_logits: Optional[torch.Tensor] = None

        # Special tokens (from tokenizer)
        self._audio_delay: int = 6  # First 6 audio tokens are BLANK
        self._blank_token_id: int = 151666  # <|im_kimia_text_blank|>
        self._text_eos_id: int = 151667  # <|im_kimia_text_eos|>
        self._token_offset: int = 152064  # Audio tokens start here

        # Audio tokenizer for discrete audio tokens
        try:
            import sys
            sys.path.insert(0, '/root/learning/Kimi-Audio')
            from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
            logger.info("Loading Glm4Tokenizer for discrete audio tokenization")
            self.audio_tokenizer = Glm4Tokenizer("THUDM/glm-4-voice-tokenizer")
            import torch
            self.audio_tokenizer = self.audio_tokenizer.to(torch.cuda.current_device())
            logger.info("Glm4Tokenizer loaded successfully")
        except Exception as e:
            logger.warning("Failed to load Glm4Tokenizer: %s. Audio tokenization unavailable.", e, exc_info=True)
            self.audio_tokenizer = None

        # Store audio tokens for current request (will be set in _parse_and_validate_audio_input)
        self._current_audio_tokens: Optional[list[int]] = None

    def _log_tensor_stats(self, name: str, tensor: torch.Tensor) -> None:
        """Log tensor statistics (only in eager mode, not during CUDA graph capture).

        Args:
            name: Label for the tensor (e.g., "After layer 21")
            tensor: Tensor to log statistics for
        """
        if not torch.cuda.is_current_stream_capturing():
            try:
                mean_val = tensor.mean().item()
                std_val = tensor.std().item()
                logger.debug("%s: mean=%.4f, std=%.4f", name, mean_val, std_val)
            except Exception as e:
                logger.debug("%s: error logging stats: %s", name, e)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        multimodal_embeddings: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        sampling_metadata: Optional[torch.Tensor] = None,
        logits_index: Optional[int] = None,
        sampler=None,
        additional_information: Optional[dict] = None,
        runtime_additional_information: Optional[list] = None,
        **kwargs,
    ) -> OmniOutput:
        """Forward pass with bifurcation at layer 21."""

        # Check both additional_information and runtime_additional_information
        # runtime_additional_information is a list (one per request in batch)
        # We need to extract the first element for single-request batch
        effective_additional_info = additional_information
        if effective_additional_info is None and runtime_additional_information is not None:
            if isinstance(runtime_additional_information, list) and len(runtime_additional_information) > 0:
                effective_additional_info = runtime_additional_information[0]
            elif isinstance(runtime_additional_information, dict):
                effective_additional_info = runtime_additional_information

        logger.debug("forward: additional_information keys: %s",
                     list(effective_additional_info.keys()) if effective_additional_info else None)
        logger.debug("forward: runtime_additional_information length: %s",
                     len(runtime_additional_information) if runtime_additional_information else None)

        # Extract raw audio from additional_information and tokenize it
        logger.info("forward: effective_additional_info keys: %s",
                     list(effective_additional_info.keys()) if effective_additional_info else None)
        logger.info("forward: _current_audio_tokens is None: %s", self._current_audio_tokens is None)

        if effective_additional_info is not None and self._current_audio_tokens is None:
            try:
                deferred_data = effective_additional_info.get("deferred_multi_modal_data", {})
                logger.info("forward: deferred_data keys: %s", list(deferred_data.keys()) if deferred_data else None)
                if deferred_data:
                    audio_data_list = deferred_data.get("audio", [])
                    logger.info("forward: audio_data_list length: %d", len(audio_data_list) if audio_data_list else 0)
                    if audio_data_list and len(audio_data_list) > 0:
                        raw_audio = audio_data_list[0]
                        logger.info("forward: Raw audio type: %s, size: %s", type(raw_audio),
                                   len(raw_audio) if hasattr(raw_audio, '__len__') else 'N/A')
                        logger.info("forward: Raw audio content: %s", str(raw_audio)[:500])

                        discrete_tokens = self._tokenize_audio_to_discrete_tokens(raw_audio)
                        if discrete_tokens is not None:
                            self._current_audio_tokens = discrete_tokens
                            logger.info("✅ forward: Tokenized audio into %d discrete tokens", len(discrete_tokens))
                            logger.info("   First 5 tokens: %s", discrete_tokens[:5])
                            logger.info("   Last 5 tokens: %s", discrete_tokens[-5:])
                        else:
                            logger.error("❌ forward: Could not tokenize audio")
                    else:
                        logger.warning("forward: No audio data in deferred_data")
                else:
                    logger.warning("forward: No deferred_data in additional_information")
            except Exception as e:
                logger.error("❌ ERROR extracting audio from additional_information: %s", e, exc_info=True)

        # Replace BLANK tokens in input_ids with actual audio tokens
        # This is critical: model expects discrete audio tokens, not BLANK tokens
        logger.info("forward: _current_audio_tokens available: %s", self._current_audio_tokens is not None)
        logger.info("forward: input_ids shape: %s", input_ids.shape if input_ids is not None else None)

        if self._current_audio_tokens is not None and input_ids is not None:
            try:
                blank_mask = (input_ids == self._blank_token_id)
                num_blank_positions = blank_mask.sum()  # Keep as tensor, no .item()

                # Guard logging with CUDA graph capture detection
                if not torch.cuda.is_current_stream_capturing():
                    logger.info("forward: Found %d BLANK tokens (id=%d) in input_ids", num_blank_positions.item(), self._blank_token_id)
                    logger.info("forward: Have %d audio tokens to insert", len(self._current_audio_tokens))

                if self._current_audio_tokens:  # Non-empty check (Python-level, OK)
                    audio_tokens_tensor = torch.tensor(
                        self._current_audio_tokens,
                        dtype=input_ids.dtype,
                        device=input_ids.device
                    )
                    # CUDA graph compatibility: compare Python int with tensor
                    # During capture, this comparison is evaluated at trace time
                    # In eager mode, it's evaluated each time
                    num_blank_int = num_blank_positions.item() if not torch.cuda.is_current_stream_capturing() else audio_tokens_tensor.shape[0]

                    if audio_tokens_tensor.shape[0] == num_blank_int:
                        # Use torch.where instead of masked assignment for CUDA graph compatibility
                        input_ids = torch.where(blank_mask, audio_tokens_tensor, input_ids)
                        if not torch.cuda.is_current_stream_capturing():
                            logger.info("✅ Replaced %d BLANK tokens with %d audio tokens",
                                       num_blank_int, len(self._current_audio_tokens))
                    else:
                        # Shape mismatch - only log in eager mode
                        if not torch.cuda.is_current_stream_capturing():
                            logger.error("❌ MISMATCH: %d BLANK positions but %d audio tokens - cannot replace!",
                                       num_blank_int, len(self._current_audio_tokens))
                else:
                    if not torch.cuda.is_current_stream_capturing():
                        logger.error("❌ No audio tokens available - audio will not be processed!")
            except Exception as e:
                if not torch.cuda.is_current_stream_capturing():
                    logger.error("❌ ERROR replacing BLANK tokens: %s", e, exc_info=True)

        # 1. Embed inputs (reuse upstream fusion)
        if inputs_embeds is not None:
            # Use provided embeddings if available
            inputs_embeds_fused = inputs_embeds
        else:
            inputs_embeds_fused = self.embed_input_ids(input_ids, multimodal_embeddings)

        # 2. Forward through layers 0-21 (first 22 layers)
        hidden_states, residual = self._forward_to_layer_21(input_ids, positions, inputs_embeds_fused)

        # Debug logging (CUDA graph compatible)
        if not torch.cuda.is_current_stream_capturing():
            try:
                hidden_mean = hidden_states.mean().item() if not torch.isnan(hidden_states.mean()) else float('nan')
                hidden_std = hidden_states.std().item() if not torch.isnan(hidden_states.std()) else float('nan')
                logger.debug("After layer 21: shape=%s, residual=%s, mean=%.4f, std=%.4f",
                             hidden_states.shape,
                             None if residual is None else residual.shape,
                             hidden_mean, hidden_std)
            except Exception as e:
                logger.debug("After layer 21: shape=%s (debug error: %s)", hidden_states.shape, e)

        # 3. Bifurcation: clone hidden states AFTER layer 21
        # Reference: modeling_moonshot_kimia.py line 788-789
        # The audio path gets the output of layer 21
        text_hidden_states = hidden_states.clone()
        audio_hidden_states = hidden_states.clone()
        text_residual = residual.clone() if residual is not None else None
        audio_residual = residual.clone() if residual is not None else None

        # 4. Text path: layers 22-27 → return hidden_states for compute_logits
        for layer in self.model.model.layers[22:]:  # Layers 22-27
            text_hidden_states, text_residual = layer(positions, text_hidden_states, text_residual)

        # Apply final norm (takes both hidden_states and residual in vLLM)
        text_hidden_states, _ = self.model.model.norm(text_hidden_states, text_residual)

        # Debug logging (CUDA graph compatible)
        if not torch.cuda.is_current_stream_capturing():
            try:
                logger.debug("Text path: shape=%s, mean=%.4f, std=%.4f",
                             text_hidden_states.shape,
                             text_hidden_states.mean().item(),
                             text_hidden_states.std().item())
            except Exception:
                logger.debug("Text path: shape=%s", text_hidden_states.shape)

        # 5. Audio path: 6 MIMO layers → mimo_output
        # Audio path starts from layer 21 output (already in audio_hidden_states)
        for mimo_layer in self.mimo_layers:
            audio_hidden_states, audio_residual = mimo_layer(positions, audio_hidden_states, audio_residual)

        # Apply final norm (takes both hidden_states and residual in vLLM)
        audio_hidden_states, _ = self.mimo_norm(audio_hidden_states, audio_residual)
        audio_logits_output = self.mimo_output(audio_hidden_states)

        # ColumnParallelLinear may return tuple (output, bias) or just tensor
        if isinstance(audio_logits_output, tuple):
            audio_logits = audio_logits_output[0]
        else:
            audio_logits = audio_logits_output

        # Debug logging (CUDA graph compatible)
        if not torch.cuda.is_current_stream_capturing():
            try:
                logger.debug("Audio path: shape=%s, logits=%s, mean=%.4f, std=%.4f",
                             audio_hidden_states.shape, audio_logits.shape,
                             audio_hidden_states.mean().item(),
                             audio_hidden_states.std().item())
            except Exception:
                logger.debug("Audio path: shape=%s, logits=%s",
                             audio_hidden_states.shape, audio_logits.shape)

        # Debug: Log top 5 audio logits (CUDA graph compatible)
        if not torch.cuda.is_current_stream_capturing() and audio_logits.shape[0] > 0:
            top5_audio = torch.topk(audio_logits[0], 5)
            logger.debug("Audio path: top5 token_ids=%s, logits=%s",
                         top5_audio.indices.tolist(), top5_audio.values.tolist())
            num_audio_range = (top5_audio.indices >= 152064).sum().item()
            logger.debug("Audio path: %d/5 top tokens in audio range [152064, 168447]", num_audio_range)

        # 6. Store audio logits for sample() to use
        self._pending_audio_logits = audio_logits

        # 7. Return text hidden_states (not logits) and audio logits via OmniOutput
        # vLLM's compute_logits will handle text logits via lm_head
        return OmniOutput(
            text_hidden_states=text_hidden_states,  # Return hidden_states, not logits
            multimodal_outputs={"audio_logits": audio_logits},
        )

    def _forward_to_layer_21(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward through layers 0-21 (first 22 layers) and return hidden states and residual."""
        hidden_states = inputs_embeds
        residual = None

        # Forward through layers 0-21 (first 22 layers)
        for layer in self.model.model.layers[:22]:  # Layers 0-21
            hidden_states, residual = layer(positions, hidden_states, residual)

        # Ensure residual has correct shape (squeeze extra dimensions if present)
        if residual is not None and residual.dim() > hidden_states.dim():
            residual = residual.view(hidden_states.shape)

        return hidden_states, residual

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor] | None:
        """Process audio input and return multimodal embeddings.

        This method is called by vLLM's multimodal processing pipeline.
        It processes audio inputs through the Whisper encoder and VQAdaptor.
        """
        # Parse audio input from kwargs
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        # Process audio through Whisper encoder and VQAdaptor
        audio_embeds = self._process_audio_input(audio_input)

        # Log that audio features successfully reached the model
        logger.info(
            "embed_multimodal: Successfully processed audio features -> shape=%s, dtype=%s",
            list(audio_embeds.shape),
            audio_embeds.dtype,
        )

        # Return as list of 2D tensors, one per batch item
        if audio_embeds.dim() == 3:
            # Unbind batch dimension: [B, T, D] -> list of B tensors [T, D]
            return list(audio_embeds.unbind(dim=0))
        else:
            # Single sample: [T, D] -> wrap in list
            return [audio_embeds]

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[dict]:
        """Parse and validate audio input from kwargs."""
        logger.debug("_parse_and_validate_audio_input: kwargs keys: %s", list(kwargs.keys()))

        # Check if audio_tokens are available
        audio_tokens = kwargs.get("audio_tokens", None)
        if audio_tokens is not None:
            if isinstance(audio_tokens, (list, tuple)) and len(audio_tokens) > 0:
                self._current_audio_tokens = list(audio_tokens)
                logger.debug("Stored %d audio tokens from kwargs", len(self._current_audio_tokens))
            else:
                logger.warning("audio_tokens is empty or not a list")

        # Check if raw audio is available
        raw_audio = kwargs.get("audio", None)
        if raw_audio is not None:
            logger.debug("Raw audio found in kwargs, type: %s", type(raw_audio))
            discrete_tokens = self._tokenize_audio_to_discrete_tokens(raw_audio)
            if discrete_tokens is not None:
                self._current_audio_tokens = discrete_tokens
                logger.debug("Tokenized audio into %d discrete tokens", len(discrete_tokens))
            else:
                logger.warning("Could not tokenize audio from kwargs")

        # Look for audio features in kwargs
        whisper_features = kwargs.get("whisper_input_features", None)
        feature_attention_mask = kwargs.get("feature_attention_mask", None)

        logger.debug("whisper_features type: %s, feature_attention_mask: %s",
                     type(whisper_features),
                     feature_attention_mask.shape if feature_attention_mask is not None else None)

        # If whisper_features not provided, check for raw audio input
        if whisper_features is None:
            raw_audio = kwargs.get("audio", None)
            if raw_audio is not None:
                whisper_features = self._extract_whisper_features(raw_audio)
                if whisper_features is not None:
                    discrete_tokens = self._tokenize_audio_to_discrete_tokens(raw_audio)
                    if discrete_tokens is not None:
                        self._current_audio_tokens = discrete_tokens
                        logger.debug("Stored %d discrete audio tokens", len(discrete_tokens))
                    else:
                        logger.warning("Could not tokenize audio, will use BLANK tokens")
                        self._current_audio_tokens = None

                    return {
                        "whisper_input_features": whisper_features,
                        "feature_attention_mask": None,
                    }
            return None

        # If we have whisper_features but also raw audio, tokenize the audio
        raw_audio = kwargs.get("audio", None)
        if raw_audio is not None:
            discrete_tokens = self._tokenize_audio_to_discrete_tokens(raw_audio)
            if discrete_tokens is not None:
                self._current_audio_tokens = discrete_tokens
                logger.debug("Stored %d discrete audio tokens", len(discrete_tokens))
            else:
                logger.warning("Could not tokenize audio, will use BLANK tokens")
                self._current_audio_tokens = None

        return {
            "whisper_input_features": whisper_features,
            "feature_attention_mask": feature_attention_mask,
        }

    def _extract_whisper_features(self, raw_audio: Any) -> Optional[torch.Tensor]:
        """Extract Whisper features from raw audio input.

        Args:
            raw_audio: Raw audio input (waveform tensor, numpy array, or file path)

        Returns:
            Whisper features tensor, or None if extraction fails
        """
        import numpy as np

        # Convert raw audio to tensor
        if isinstance(raw_audio, np.ndarray):
            wav_tensor = torch.from_numpy(raw_audio).float()
        elif isinstance(raw_audio, torch.Tensor):
            wav_tensor = raw_audio.float()
        elif isinstance(raw_audio, (str, bytes)):
            # Load audio from file path
            try:
                import torchaudio
                wav_tensor, sr = torchaudio.load(raw_audio)
                if sr != 16000:
                    wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
                if wav_tensor.dim() == 2:
                    wav_tensor = wav_tensor.mean(dim=0, keepdim=True)  # Mix to mono
            except Exception as e:
                logger.error("Failed to load audio from %s: %s", raw_audio, e)
                return None
        else:
            logger.error("Unsupported audio type: %s", type(raw_audio))
            return None

        # Ensure tensor has batch dimension
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)  # [T] -> [1, T]

        # Move to device
        wav_tensor = wav_tensor.to(next(self.audio_tower.parameters()).device)

        # Extract Whisper features using the audio tower
        try:
            if hasattr(self.audio_tower, 'tokenize_waveform'):
                whisper_features = self.audio_tower.tokenize_waveform(wav_tensor)
            else:
                whisper_features = self.audio_tower(wav_tensor)

            # Reshape features if needed (reference implementation does this)
            if whisper_features.dim() == 3:
                whisper_features = whisper_features.reshape(
                    whisper_features.shape[0],
                    int(whisper_features.shape[1] // 4),
                    whisper_features.shape[2] * 4,
                )

            logger.debug("Extracted Whisper features: shape=%s", whisper_features.shape)
            return whisper_features

        except Exception as e:
            logger.error("Failed to extract Whisper features: %s", e, exc_info=True)
            return None

    def _tokenize_audio_to_discrete_tokens(self, raw_audio: Any) -> Optional[list[int]]:
        """Tokenize audio into discrete tokens using Glm4Tokenizer.

        Args:
            raw_audio: Raw audio input (waveform tensor, numpy array, bytes, or file path)

        Returns:
            List of discrete audio token IDs, or None if tokenization fails
        """
        logger.info("_tokenize_audio_to_discrete_tokens: called with type=%s", type(raw_audio))

        if self.audio_tokenizer is None:
            logger.error("❌ Audio tokenizer not available, cannot tokenize audio")
            return None

        temp_path = None
        try:
            import numpy as np

            # Convert raw audio to file path for tokenizer
            if isinstance(raw_audio, bytes):
                # Raw audio bytes - load with soundfile (more compatible than torchaudio)
                import io
                import soundfile as sf
                logger.info("  Loading audio from bytes (size: %d bytes)", len(raw_audio))
                wav_tensor, sr = sf.read(io.BytesIO(raw_audio), dtype='float32')
                # Convert to torch tensor and ensure correct shape [channels, samples]
                wav_tensor = torch.from_numpy(wav_tensor)
                if wav_tensor.dim() == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)  # [1, samples]
                elif wav_tensor.dim() == 2:
                    wav_tensor = wav_tensor.T  # [samples, channels] -> [channels, samples]
                logger.info("  Loaded audio: shape=%s, sample_rate=%d", wav_tensor.shape, sr)
                if sr != 16000:
                    logger.info("  Resampling from %d to 16000 Hz", sr)
                    # Use scipy for resampling to avoid torchaudio/torchcodec dependency
                    from scipy import signal
                    # wav_tensor shape: [channels, samples]
                    num_samples_16k = int(wav_tensor.shape[1] * 16000 / sr)
                    wav_resampled = signal.resample(wav_tensor.numpy(), num_samples_16k, axis=1)
                    wav_tensor = torch.from_numpy(wav_resampled).float()
                # Save to temp file for tokenizer using soundfile
                with __import__('tempfile').NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                # soundfile expects [samples, channels] format
                wav_for_sf = wav_tensor.squeeze(0).numpy() if wav_tensor.dim() == 2 else wav_tensor.numpy()
                sf.write(temp_path, wav_for_sf, 16000)
                audio_path = temp_path
                logger.info("  Saved to temp file: %s", audio_path)
            elif isinstance(raw_audio, str):
                # Check if it's a data URL
                if raw_audio.startswith('data:'):
                    # Data URL format: data:audio/wav;base64,...
                    import base64
                    logger.info("  Processing data URL")
                    # Extract base64 data after the comma
                    comma_idx = raw_audio.find(',')
                    if comma_idx == -1:
                        logger.error("❌ Invalid data URL format")
                        return None
                    b64_data = raw_audio[comma_idx + 1:]
                    audio_bytes = base64.b64decode(b64_data)
                    # Recursively process as bytes
                    return self._tokenize_audio_to_discrete_tokens(audio_bytes)
                else:
                    audio_path = raw_audio
                    logger.info("  Using audio path: %s", audio_path)
            elif isinstance(raw_audio, np.ndarray):
                wav_tensor = torch.from_numpy(raw_audio).float()
                if wav_tensor.dim() == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)
                with __import__('tempfile').NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                # soundfile expects [samples, channels] format
                wav_for_sf = wav_tensor.squeeze(0).numpy() if wav_tensor.dim() == 2 else wav_tensor.numpy()
                sf.write(temp_path, wav_for_sf, 16000)
                audio_path = temp_path
                logger.info("  Converted numpy array to temp file: %s", audio_path)
            elif isinstance(raw_audio, torch.Tensor):
                if raw_audio.dim() == 1:
                    raw_audio = raw_audio.unsqueeze(0)
                with __import__('tempfile').NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                # soundfile expects [samples, channels] format
                wav_for_sf = raw_audio.float().squeeze(0).numpy() if raw_audio.dim() == 2 else raw_audio.float().numpy()
                sf.write(temp_path, wav_for_sf, 16000)
                audio_path = temp_path
                logger.info("  Converted tensor to temp file: %s", audio_path)
            elif isinstance(raw_audio, list):
                # List of audio items - take the first one
                if len(raw_audio) > 0:
                    logger.info("  Extracting first item from list of %d items", len(raw_audio))
                    logger.info("  First item type: %s", type(raw_audio[0]))
                    if isinstance(raw_audio[0], (list, dict)):
                        logger.info("  First item: %s", str(raw_audio[0])[:200])
                    return self._tokenize_audio_to_discrete_tokens(raw_audio[0])
                else:
                    logger.error("❌ Empty audio list")
                    return None
            else:
                logger.error("❌ Unsupported audio type: %s", type(raw_audio))
                return None

            logger.info("  Calling audio_tokenizer.tokenize(audio_path=%s)", audio_path)
            wav_tokens = self.audio_tokenizer.tokenize(audio_path=audio_path)
            logger.info("  Raw tokens shape: %s, dtype: %s", wav_tokens.shape, wav_tokens.dtype)
            logger.info("  Adding token offset: %d", self._token_offset)
            wav_tokens = wav_tokens + self._token_offset
            wav_tokens_list = wav_tokens.squeeze(0).cpu().numpy().tolist()

            logger.info("✅ Tokenized audio into %d discrete tokens", len(wav_tokens_list))
            logger.info("   Token range: [%d, %d]", min(wav_tokens_list), max(wav_tokens_list))
            return wav_tokens_list

        except Exception as e:
            logger.error("❌ Failed to tokenize audio: %s", e, exc_info=True)
            return None
        finally:
            # Clean up temp file
            if temp_path is not None:
                try:
                    import os
                    os.unlink(temp_path)
                except Exception:
                    pass

    def _process_audio_input(self, audio_input: dict) -> torch.Tensor:
        """Process audio input through Whisper encoder and VQAdaptor."""
        whisper_features = audio_input["whisper_input_features"]

        logger.debug("_process_audio_input: shape=%s, dtype=%s",
                     whisper_features.shape, whisper_features.dtype)

        # Run through Whisper encoder
        whisper_output = self.audio_tower(whisper_features)

        # Project whisper output to VQ adaptor input dimension if needed
        if self.whisper_projection is not None:
            whisper_output = self.whisper_projection(whisper_output)

        # Run through VQAdaptor (multi_modal_projector)
        audio_embeds = self.multi_modal_projector(whisper_output)

        logger.debug("_process_audio_input: audio_embeds shape=%s", audio_embeds.shape)
        return audio_embeds

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[torch.Tensor] = None,
        is_multimodal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed input IDs with dual-stream fusion.

        Fusion formula (from reference implementation):
        - For continuous whisper features: (text_emb + whisper_emb) × √2
        - For discrete audio tokens: text_emb + audio_emb (simple addition)
        """
        # 1. Embed text tokens
        text_emb = self.model.model.embed_tokens(input_ids)

        # 2. Handle whisper continuous features with √2 scaling
        if multimodal_embeddings is not None:
            import sys
            # DEBUG: Log multimodal_embeddings details
            logger.debug("input_ids shape: %s", input_ids.shape)
            logger.debug("multimodal_embeddings type: %s", type(multimodal_embeddings))
            if isinstance(multimodal_embeddings, list):
                logger.debug("multimodal_embeddings is list with %s items", len(multimodal_embeddings))
                for i, item in enumerate(multimodal_embeddings):
                    logger.debug("  Item %s: type=%s", i, type(item))
                    if hasattr(item, 'shape'):
                        logger.debug("shape=%s, dtype=%s", item.shape, item.dtype)
                    elif item is None:
                        logger.debug("value=None")
                    else:
                        logger.debug("value=%s", item)
            else:
                logger.debug("multimodal_embeddings shape: %s", multimodal_embeddings.shape)
            if is_multimodal is not None:
                logger.debug("is_multimodal shape: %s", is_multimodal.shape)
                # CUDA graph compatible logging
                if not torch.cuda.is_current_stream_capturing():
                    logger.debug("is_multimodal sum: %s", is_multimodal.sum().item())
            
            # Ensure multimodal_embeddings is a tensor (not a list)
            if isinstance(multimodal_embeddings, list):
                # Filter out None items and check if list is empty
                multimodal_embeddings = [item for item in multimodal_embeddings if item is not None and isinstance(item, torch.Tensor)]
                if len(multimodal_embeddings) == 0:
                    # No valid embeddings, skip multimodal fusion
                    logger.warning("No valid multimodal embeddings after filtering")
                    multimodal_embeddings = None
                else:
                    # Stack list items: [N, seq_len, hidden_size] where N is number of audio items
                    multimodal_embeddings = torch.stack(multimodal_embeddings)

            # Only process multimodal embeddings if we have valid tensors
            if multimodal_embeddings is not None:
                if is_multimodal is not None:
                    # is_multimodal marks which positions in the sequence should use audio features
                    # multimodal_embeddings shape: [num_audio_items, audio_seq_len, hidden_size]
                    # We need to place audio features at positions where is_multimodal is True

                    # For now, assume single audio item (most common case)
                    if multimodal_embeddings.shape[0] == 1:
                        audio_features = multimodal_embeddings[0]  # [audio_seq_len, hidden_size]
                        # CUDA graph compatible: keep as tensor
                        num_audio_positions = is_multimodal.sum()

                        # CUDA graph compatible: convert to int for comparison
                        # During capture, use shape[0] to avoid data-dependent conditional
                        num_audio_int = num_audio_positions.item() if not torch.cuda.is_current_stream_capturing() else audio_features.shape[0]

                        # CUDA graph compatible logging
                        if not torch.cuda.is_current_stream_capturing():
                            logger.debug("embed_input_ids: audio_features shape=%s, num_audio_positions=%s", audio_features.shape, num_audio_int)

                        # Check if audio features match the number of multimodal positions
                        if audio_features.shape[0] == num_audio_int:
                            # Create output tensor starting with text embeddings
                            result_emb = text_emb.clone()

                            # Get multimodal positions
                            multimodal_positions = torch.where(is_multimodal)[0]
                            logger.debug("embed_input_ids: placing audio at positions %s", multimodal_positions.tolist())

                            # REFERENCE IMPLEMENTATION FIX:
                            # Reference implementation ADDS discrete token embeddings + continuous features, then scales
                            # Reference code (modeling_moonshot_kimia.py):
                            #   encoder_input_addwith_discrete_token = (audio_emb + whisper_emb) * sqrt(2)
                            #   audio_emb = audio_emb * (~is_continuous_mask) + encoder_input_addwith_discrete_token * is_continuous_mask
                            #
                            # Previous vllm-omni was REPLACING text_emb with audio_features (WRONG)
                            # Fixed version: ADD text_emb + audio_features, then scale by sqrt(2)

                            # Debug: Check values before fusion
                            discrete_emb = result_emb[multimodal_positions]
                            # CUDA graph compatible logging
                            if not torch.cuda.is_current_stream_capturing():
                                try:
                                    logger.debug("embed_input_ids DEBUG:")
                                    logger.debug("  discrete_emb (BLANK tokens): mean={discrete_emb.mean().item():.6f}, std={discrete_emb.std().item():.6f}, min={discrete_emb.min().item():.6f}, max={discrete_emb.max().item():.6f}")
                                    logger.debug("  audio_features (continuous): mean={audio_features.mean().item():.6f}, std={audio_features.std().item():.6f}, min={audio_features.min().item():.6f}, max={audio_features.max().item():.6f}")
                                except Exception as e:
                                    logger.debug("embed_input_ids DEBUG: Error logging stats: %s", e)

                            # Add discrete token embeddings (BLANK tokens) with continuous features
                            combined_emb = discrete_emb + audio_features

                            # CUDA graph compatible logging
                            if not torch.cuda.is_current_stream_capturing():
                                try:
                                    logger.debug("  combined_emb (before √2): mean={combined_emb.mean().item():.6f}, std={combined_emb.std().item():.6f}")
                                except Exception:
                                    pass

                            # Scale by sqrt(2)
                            combined_emb = combined_emb * (2 ** 0.5)

                            # CUDA graph compatible logging
                            if not torch.cuda.is_current_stream_capturing():
                                try:
                                    logger.debug("  combined_emb (after √2): mean={combined_emb.mean().item():.6f}, std={combined_emb.std().item():.6f}")
                                except Exception:
                                    pass

                            # Place back
                            result_emb[multimodal_positions] = combined_emb

                            text_emb = result_emb
                            logger.debug("embed_input_ids: ✅ Audio features ADDED to discrete tokens and scaled by √2")
                        else:
                            logger.debug("embed_input_ids: ❌ Shape mismatch! audio_features.shape[0]={audio_features.shape[0]} != num_audio_positions={num_audio_positions}")
                            # Fallback: shapes don't match, apply additive fusion
                            whisper_emb = multimodal_embeddings.squeeze(0)
                            scaled_emb = (text_emb + whisper_emb) * (2 ** 0.5)
                            text_emb = torch.where(
                                is_multimodal.unsqueeze(-1),
                                scaled_emb,
                                text_emb
                            )
                    else:
                        # Multiple audio items - not yet implemented
                        raise NotImplementedError("Multiple audio items not yet supported")
                else:
                    # No mask provided, apply √2 scaling to all
                    text_emb = (text_emb + multimodal_embeddings) * (2 ** 0.5)

        # 3. Embed audio tokens from per-slot state (dual token stream fusion)
        # For each request, add the audio token embedding from the previous step
        # This implements the dual token stream: audio_emb + text_emb
        state = getattr(self, "_audio_state", None)
        if not state:
            # No audio state yet (first step or no audio generation)
            inputs_embeds = text_emb
            logger.debug("embed_input_ids: No audio state, using text_emb only")
        else:
            embed_weight = self.model.model.embed_tokens.weight
            num_tokens = text_emb.shape[0]

            logger.debug("embed_input_ids: Audio state exists, applying dual stream fusion for {num_tokens} tokens")

            # Determine batch row indices for each token position
            # In vLLM, input_ids is typically [num_tokens] for prefill or [num_reqs, 1] for decode
            if input_ids.dim() == 1:
                # Prefill or single request: all tokens belong to request 0
                batch_row_indices = [0] * num_tokens
            else:
                # Decode: each row is one request with 1 token
                num_reqs = input_ids.shape[0]
                batch_row_indices = list(range(num_reqs))

            # Start with text embeddings
            inputs_embeds = text_emb.clone()

            # For each position, add the audio embedding from the corresponding request's state
            audio_tokens_added = 0
            for pos in range(num_tokens):
                batch_i = batch_row_indices[pos] if pos < len(batch_row_indices) else 0
                req_state = state.get(batch_i)
                if req_state is None:
                    continue

                audio_out_ids = req_state.get("audio_out_ids")
                if audio_out_ids is None or audio_out_ids.numel() == 0:
                    continue

                # Get the LAST audio token for this request (from previous generation step)
                last_audio_token = audio_out_ids[:, -1:]  # [1, 1]
                # CUDA graph compatible: keep as tensor, use torch.clamp
                last_audio_token_id = last_audio_token.squeeze()  # [1] tensor
                last_audio_token_id = torch.clamp(last_audio_token_id, 0, embed_weight.shape[0] - 1)

                # Embed the audio token and ADD to text embedding (dual stream fusion)
                audio_emb = embed_weight[last_audio_token_id]  # Indexing with tensor is OK
                inputs_embeds[pos] = inputs_embeds[pos] + audio_emb.to(dtype=inputs_embeds.dtype)
                audio_tokens_added += 1

            logger.debug("embed_input_ids: Added {audio_tokens_added} audio token embeddings")

        return inputs_embeds

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute text logits using lm_head."""
        # Use the lm_head from the underlying model
        logits = self.logits_processor(self.model.lm_head, hidden_states)

        # CRITICAL: Mask out audio tokens from text logits
        # Kimi Audio uses a unified vocabulary where:
        # - Text tokens: [0, 152063]
        # - Audio tokens: [152064, 168447]
        # We need to prevent the text sampler from selecting audio tokens
        kimia_token_offset = 152064  # From config.kimia_token_offset
        if logits is not None and logits.shape[-1] > kimia_token_offset:
            # Set audio token logits to -inf so they won't be sampled
            logits[:, kimia_token_offset:] = -float('inf')

        logger.debug("compute_logits: hidden_states shape=%s, logits shape=%s, hidden_mean=%.4f, hidden_std=%.4f",
                     hidden_states.shape,
                     logits.shape if logits is not None else 'None',
                     hidden_states.mean(), hidden_states.std())
        if logits is not None:
            logger.debug("compute_logits: logits_mean=%.4f, logits_std=%.4f, logits_max=%.4f",
                         logits.mean(), logits.std(), logits.max())
            # Log top 5 tokens for debugging
            top5 = torch.topk(logits[0], 5)
            logger.debug("compute_logits: top5 token_ids=%s, top5 logits=%s",
                         top5.indices.tolist(), top5.values.tolist())

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Custom sampler for dual token streaming.

        Implements the reference implementation's dual token stream logic:
        1. Sample both text and audio tokens
        2. Handle text stream termination (replace with BLANK after EOS)
        3. Handle audio delay (first 6 tokens are BLANK)
        4. Store audio tokens per-slot for next step's embedding
        """
        # Initialize per-slot state dicts if needed (lazy initialization)
        if not hasattr(self, "_audio_state"):
            self._audio_state = {}
        if not hasattr(self, "_slot_output_len"):
            self._slot_output_len = {}
        if not hasattr(self, "_text_stream_finished"):
            self._text_stream_finished = {}

        # Initialize stock sampler if needed (following HiggsAudioV2 pattern)
        sampler = getattr(self, "_stock_sampler", None)
        if sampler is None:
            from vllm.v1.sample.sampler import Sampler
            sampler = Sampler()
            self._stock_sampler = sampler

        # Get batch row indices
        num_reqs = logits.shape[0]
        batch_row_indices = list(range(num_reqs))

        # Detect slot reuse (new request in same slot)
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None)
        for batch_i in batch_row_indices:
            current_len = len(output_token_ids[batch_i]) if output_token_ids else 0
            prior_len = self._slot_output_len.get(batch_i, -1)

            # If length decreased, a new request has taken over this slot
            is_new_request = (prior_len > current_len) or (prior_len == -1 and current_len == 0)
            if is_new_request:
                # Evict all stale state for this slot
                self._audio_state.pop(batch_i, None)
                self._text_stream_finished.pop(batch_i, None)
            self._slot_output_len[batch_i] = current_len

        # 1. Sample text token from text logits
        text_sampler_output = sampler(logits=logits, sampling_metadata=sampling_metadata)
        text_tokens = text_sampler_output.sampled_token_ids  # [num_reqs, 1]

        # 2. Get audio logits (stored by forward() in self._pending_audio_logits)
        audio_logits = self._pending_audio_logits
        if audio_logits is None:
            return text_sampler_output

        # 3. Process each request's tokens following reference implementation logic
        for batch_i in batch_row_indices:
            # Get or create state for this slot
            state = self._audio_state.setdefault(
                batch_i,
                {"generation_step": 0, "audio_out_ids": None}
            )

            # Check if text stream already finished for this request
            text_finished = self._text_stream_finished.get(batch_i, False)

            # Get the sampled text token for this request
            if text_tokens.numel() > batch_i:
                # CUDA graph compatible: keep as tensor
                text_token_id = text_tokens[batch_i]
            else:
                continue  # Skip if no token for this request

            # Handle text stream termination (reference: kimia.py line 134-139)
            if text_finished:
                # Text stream already finished, replace token with BLANK
                text_token_id = torch.tensor(self._blank_token_id, device=text_tokens.device, dtype=text_tokens.dtype)
                text_tokens[batch_i] = self._blank_token_id
                logger.debug("Text stream finished, replacing token with BLANK")
            elif text_token_id == self._text_eos_id:
                # Text stream just finished, mark it and keep the EOS token
                # PyTorch can compare tensor == int directly
                self._text_stream_finished[batch_i] = True
                logger.debug("Text stream just finished (EOS token detected)")
            # else: text stream still active, keep the sampled token

            # 4. Handle audio delay and sampling (reference: kimia.py line 143-149)
            step = state["generation_step"]

            if step < self._audio_delay:
                # First 6 tokens: force BLANK (reference: kimia.py line 143-144)
                audio_token_id = self._blank_token_id
            else:
                # Sample from audio logits using argmax (greedy for now)
                audio_logits_filtered = audio_logits[batch_i:batch_i+1, self._token_offset:]  # [1, 16384]
                audio_token_idx = torch.argmax(audio_logits_filtered, dim=-1, keepdim=True)  # [1, 1]
                # CUDA graph compatible: tensor + int = tensor
                audio_token_id = audio_token_idx + self._token_offset

            # 5. Append to cumulative audio history for this slot
            audio_token = torch.tensor([[audio_token_id]], device=logits.device)
            if state["audio_out_ids"] is None:
                state["audio_out_ids"] = audio_token.clone()  # [1, 1]
            else:
                state["audio_out_ids"] = torch.cat(
                    [state["audio_out_ids"], audio_token], dim=-1
                )  # [1, T_so_far]

            # 6. Increment step counter for this slot
            state["generation_step"] = step + 1

            # Debug logging (only for first few steps or periodically)
            if step <= 10 or step % 50 == 0:
                logger.debug("slot %s step %s: text_token=%s, audio_token=%s, text_finished=%s",
                             batch_i, step+1, text_token_id, audio_token_id, text_finished)

        # Store last audio tokens for backward compatibility
        last_audio_tokens = []
        for batch_i in batch_row_indices:
            if batch_i in self._audio_state and self._audio_state[batch_i]["audio_out_ids"] is not None:
                last_token = self._audio_state[batch_i]["audio_out_ids"][:, -1:]
                last_audio_tokens.append(last_token)

        if last_audio_tokens:
            self._last_audio_tokens = torch.cat(last_audio_tokens, dim=0)
        else:
            self._last_audio_tokens = None

        # Return text tokens (drives the engine)
        # Debug: log what we're returning
        if batch_row_indices and text_tokens.numel() > 0:
            logger.debug("sample() returning text_tokens: %s", text_tokens.tolist())
        return text_sampler_output

    def make_omni_output(
        self,
        model_outputs,  # Can be OmniOutput, dict, or tuple (when CUDA graphs are enabled)
        **kwargs,
    ) -> OmniOutput:
        """Package dual-stream output into OmniOutput.

        Includes audio tokens for feedback to next step via next_token_id.
        Uses per-slot audio tokens from _last_audio_tokens.

        Handles three cases:
        1. OmniOutput object (normal eager mode)
        2. dict (some code paths)
        3. tuple (CUDA graph mode - tensors are returned directly)
        """
        # Handle different output formats
        if isinstance(model_outputs, tuple):
            # CUDA graph mode: forward() returns (text_hidden_states, audio_logits)
            # The exact structure depends on what forward() returns
            if len(model_outputs) >= 2:
                text_hidden = model_outputs[0]
                audio_logits = model_outputs[1]
            else:
                text_hidden = model_outputs[0]
                audio_logits = None
        elif isinstance(model_outputs, dict):
            text_hidden = model_outputs.get("text_hidden_states")
            audio_logits = model_outputs.get("audio_logits")
        else:
            # OmniOutput object
            text_hidden = getattr(model_outputs, "text_hidden_states", None)
            audio_logits = getattr(model_outputs, "audio_logits", None)
            if audio_logits is None and hasattr(model_outputs, "multimodal_outputs"):
                audio_logits = model_outputs.multimodal_outputs.get("audio_logits")

        # Get the last audio tokens for each request (per-slot)
        last_audio_tokens = getattr(self, "_last_audio_tokens", None)

        return OmniOutput(
            text_hidden_states=text_hidden,
            multimodal_outputs={
                "audio_logits": audio_logits,
                "audio_tokens": last_audio_tokens,
            },
            next_token_id=last_audio_tokens,  # Critical for feedback!
        )

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        **req_infos,
    ) -> dict:
        """Per-request postprocessing to sync state.

        Returns update dict that gets merged into model_intermediate_buffer.
        Returns per-slot audio and text stream state for each request.
        """
        state = getattr(self, "_audio_state", None)
        text_finished = getattr(self, "_text_stream_finished", None)

        if not state and not text_finished:
            return {}

        # Get the number of requests from req_infos
        num_reqs = req_infos.get("num_reqs", 1)

        per_req_state = {}
        for batch_i in range(num_reqs):
            # Audio state
            if state:
                req_state = state.get(batch_i)
                if req_state and req_state.get("audio_out_ids") is not None:
                    last_audio_token = req_state["audio_out_ids"][:, -1:].item()
                    per_req_state[f"audio_token_{batch_i}"] = last_audio_token
                    per_req_state[f"generation_step_{batch_i}"] = req_state["generation_step"]

            # Text stream termination state
            if text_finished:
                is_finished = text_finished.get(batch_i, False)
                per_req_state[f"text_finished_{batch_i}"] = is_finished

        return per_req_state

    def on_requests_finished(self, finished_req_ids: list[str]) -> None:
        """Reset state for finished requests.

        Note: With per-slot state management, slot reuse is detected inline in sample()
        via _slot_output_len. This method is kept for compatibility but does minimal cleanup.
        """
        # Clear pending logits (will be repopulated on next forward pass)
        self._pending_audio_logits = None
        # Note: We don't clear _audio_state, _slot_output_len, or _text_stream_finished here because
        # slot reuse detection in sample() handles cleanup automatically.
        # This prevents issues where on_requests_finished() is called while
        # other requests are still in flight.

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Load weights from checkpoint."""
        # Separate weights by component
        audio_tower_weights = []
        projector_weights = []
        mimo_layers_weights = []
        mimo_output_weights = []
        mimo_norm_weights = []
        model_weights = []

        for name, tensor in weights:
            # Checkpoint uses different prefixes for different components
            # MIMO layers: "model.mimo_layers.X.*"
            # MIMO norm: "model.mimo_norm.*"
            # MIMO output: "mimo_output.*" (no model. prefix!)
            if name.startswith("model.mimo_layers."):
                # Strip "model.mimo_layers." prefix for our ModuleList structure
                # ModuleList expects keys like "0.self_attn.q_proj.weight"
                new_name = name.replace("model.mimo_layers.", "", 1)
                mimo_layers_weights.append((new_name, tensor))
            elif name.startswith("model.mimo_norm."):
                # Strip "model.mimo_norm." prefix
                new_name = name.replace("model.mimo_norm.", "", 1)
                mimo_norm_weights.append((new_name, tensor))
            elif name.startswith("mimo_output."):
                # Strip "mimo_output." prefix (no model. prefix in checkpoint!)
                new_name = name.replace("mimo_output.", "", 1)
                mimo_output_weights.append((new_name, tensor))
            elif name.startswith("audio_tower."):
                audio_tower_weights.append((name.replace("audio_tower.", "", 1), tensor))
            elif name.startswith("multi_modal_projector."):
                projector_weights.append((name.replace("multi_modal_projector.", "", 1), tensor))
            else:
                # Main model weights (Qwen2 backbone) - keep "model." prefix
                # Qwen2ForCausalLM expects parameter names like "model.layers.X.*"
                model_weights.append((name, tensor))

        # Load audio tower
        if audio_tower_weights:
            self.audio_tower.load_weights(audio_tower_weights)

        # Load projector
        if projector_weights:
            self.multi_modal_projector.load_weights(projector_weights)

        # Load MIMO layers with proper GQA-aware weight fusion
        if mimo_layers_weights:
            logger.debug("Loading %s MIMO layer weights", len(mimo_layers_weights))
            mimo_state_dict = {}

            # Group weights by layer
            layer_weights = {}
            for name, tensor in mimo_layers_weights:
                layer_idx = name.split('.')[0]
                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = {}
                layer_weights[layer_idx][name] = tensor

            # Process each layer
            for layer_idx, weights in layer_weights.items():
                for name, tensor in weights.items():
                    # Map separate gate/up projections to fused gate_up_proj
                    if ".mlp.gate_proj.weight" in name:
                        new_name = name.replace(".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight")
                        up_name = name.replace(".mlp.gate_proj.weight", ".mlp.up_proj.weight")
                        up_tensor = weights.get(up_name)
                        if up_tensor is not None:
                            fused = torch.cat([tensor, up_tensor], dim=0)
                            mimo_state_dict[new_name] = fused
                    elif ".mlp.up_proj.weight" in name:
                        continue  # Already fused with gate_proj
                    # Map separate q/k/v projections to fused qkv_proj (GQA-aware)
                    elif ".self_attn.q_proj.weight" in name:
                        new_name = name.replace(".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight")
                        k_name = name.replace(".self_attn.q_proj.weight", ".self_attn.k_proj.weight")
                        v_name = name.replace(".self_attn.q_proj.weight", ".self_attn.v_proj.weight")
                        k_tensor = weights.get(k_name)
                        v_tensor = weights.get(v_name)
                        if k_tensor is not None and v_tensor is not None:
                            # For GQA, concatenate along output dimension
                            fused = torch.cat([tensor, k_tensor, v_tensor], dim=0)
                            mimo_state_dict[new_name] = fused
                    elif ".self_attn.q_proj.bias" in name:
                        new_name = name.replace(".self_attn.q_proj.bias", ".self_attn.qkv_proj.bias")
                        k_name = name.replace(".self_attn.q_proj.bias", ".self_attn.k_proj.bias")
                        v_name = name.replace(".self_attn.q_proj.bias", ".self_attn.v_proj.bias")
                        k_tensor = weights.get(k_name)
                        v_tensor = weights.get(v_name)
                        if k_tensor is not None and v_tensor is not None:
                            fused = torch.cat([tensor, k_tensor, v_tensor], dim=0)
                            mimo_state_dict[new_name] = fused
                    elif any(x in name for x in [".self_attn.k_proj.", ".self_attn.v_proj."]):
                        continue  # Already fused with q_proj
                    else:
                        mimo_state_dict[name] = tensor

            missing, unexpected = self.mimo_layers.load_state_dict(mimo_state_dict, strict=False)
            if missing:
                logger.warning("Missing MIMO weights: %s...", missing[:5])
            if unexpected:
                logger.warning("Unexpected MIMO weights: %s...", unexpected[:5])
            logger.debug("MIMO layers loaded: %s weights", len(mimo_state_dict))

        # Load audio output head
        if mimo_output_weights:
            logger.debug("Loading %s mimo_output weights", len(mimo_output_weights))
            mimo_output_state_dict = {k: v for k, v in mimo_output_weights}
            missing, unexpected = self.mimo_output.load_state_dict(mimo_output_state_dict, strict=False)
            if missing:
                logger.warning("Missing mimo_output weights: %s", missing)
            if unexpected:
                logger.warning("Unexpected mimo_output weights: %s", unexpected)
            logger.debug("mimo_output loaded successfully")
            # Verify weight shape
            logger.debug("mimo_output.weight shape: %s, mean: %.6f, std: %.6f",
                         self.mimo_output.weight.shape,
                         self.mimo_output.weight.mean(),
                         self.mimo_output.weight.std())

        if mimo_norm_weights:
            logger.debug("Loading %s mimo_norm weights", len(mimo_norm_weights))
            missing, unexpected = self.mimo_norm.load_state_dict(
                {k: v for k, v in mimo_norm_weights},
                strict=False
            )
            if missing:
                logger.warning("Missing mimo_norm weights: %s", missing)
            if unexpected:
                logger.warning("Unexpected mimo_norm weights: %s", unexpected)
            logger.debug("mimo_norm loaded successfully")

        # Load main model (Qwen2 backbone)
        # Pass weights as-is - parameters are named "model.layers.*"
        if model_weights:
            logger.debug("Loading %s main model weights", len(model_weights))
            # Show first few weight names for debugging
            for name, tensor in model_weights[:5]:
                logger.debug("Main model weight: %s shape=%s", name, tensor.shape)
            self.model.load_weights(model_weights)
            logger.debug("Main model weights loaded successfully")

            # Debug: Check if embeddings are loaded
            embed_weight = self.model.model.embed_tokens.weight
            lm_head_weight = self.model.lm_head.weight
            logger.debug("Embeddings loaded: shape=%s, mean=%.6f, std=%.6f",
                         embed_weight.shape, embed_weight.mean(), embed_weight.std())
            logger.debug("LM head loaded: shape=%s, mean=%.6f, std=%.6f",
                         lm_head_weight.shape, lm_head_weight.mean(), lm_head_weight.std())

            # Check a few specific weights to verify they're not random
            logger.debug("Embed weight sample [0,:5]: %s", embed_weight[0, :5].tolist())
            logger.debug("LM head weight sample [0,:5]: %s", lm_head_weight[0, :5].tolist())

            # Check layer 0 weights
            layer0_attn_q = self.model.model.layers[0].self_attn.qkv_proj.weight
            logger.debug("Layer 0 attention qkv weight: shape=%s, mean=%.6f, std=%.6f",
                         layer0_attn_q.shape, layer0_attn_q.mean(), layer0_attn_q.std())
