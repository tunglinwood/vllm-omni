# Copyright 2025 vLLM-Omni Team
"""Stage 0: Kimi Audio LLM with bifurcation for dual output (text + audio)."""

from typing import Optional

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

# Import upstream vllm's KimiAudio processor components
from vllm.model_executor.models.kimi_audio import (
    KimiAudioMultiModalProcessor,
    KimiAudioProcessingInfo,
    KimiAudioDummyInputsBuilder,
)

from vllm_omni.model_executor.models.output_templates import OmniOutput

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


@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioMultiModalProcessor,
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
            KimiAudioWhisperEncoder,
            KimiAudioMultiModalProjector,
        )

        # Audio input processing (reuse from upstream vllm)
        self.audio_tower = KimiAudioWhisperEncoder(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "audio_tower"),
        )

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

        # Dual streaming state (for feeding back both audio and text tokens)
        self._pending_audio_token: Optional[torch.Tensor] = None
        self._pending_audio_logits: Optional[torch.Tensor] = None
        self._generation_step: int = 0

        # Special tokens (from reference implementation)
        self._audio_delay: int = 6  # First 6 audio tokens are BLANK
        self._blank_token_id: int = 18  # kimia_text_blank
        self._text_eos_id: int = 19  # kimia_text_eos
        self._token_offset: int = 152064  # Audio tokens start here

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
        **kwargs,
    ) -> OmniOutput:
        """Forward pass with bifurcation at layer 21."""

        # 1. Embed inputs (reuse upstream fusion)
        if inputs_embeds is not None:
            # Use provided embeddings if available
            inputs_embeds_fused = inputs_embeds
        else:
            inputs_embeds_fused = self.embed_input_ids(input_ids, multimodal_embeddings)

        # 2. Forward through layers 0-21 (first 22 layers)
        hidden_states, residual = self._forward_to_layer_21(input_ids, positions, inputs_embeds_fused)

        # Debug logging (safe formatting)
        try:
            hidden_mean = hidden_states.mean().item() if not torch.isnan(hidden_states.mean()) else float('nan')
            hidden_std = hidden_states.std().item() if not torch.isnan(hidden_states.std()) else float('nan')
            print(f"[KimiAudio] After layer 21: hidden_states shape={hidden_states.shape}, "
                  f"residual={'None' if residual is None else residual.shape}, "
                  f"hidden_stats: mean={hidden_mean:.4f}, std={hidden_std:.4f}")
        except Exception as e:
            print(f"[KimiAudio] After layer 21: shape={hidden_states.shape} (debug error: {e})")

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

        try:
            print(f"[KimiAudio] Text path: hidden_states shape={text_hidden_states.shape}, "
                  f"mean={text_hidden_states.mean().item():.4f}, std={text_hidden_states.std().item():.4f}")
        except:
            print(f"[KimiAudio] Text path: shape={text_hidden_states.shape}")

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

        try:
            print(f"[KimiAudio] Audio path: hidden_states shape={audio_hidden_states.shape}, "
                  f"audio_logits shape={audio_logits.shape}, "
                  f"mean={audio_hidden_states.mean().item():.4f}, std={audio_hidden_states.std().item():.4f}")
        except:
            print(f"[KimiAudio] Audio path: shape={audio_hidden_states.shape}, logits={audio_logits.shape}")

        # Debug: Print top 5 audio logits
        if audio_logits.shape[0] > 0:
            top5_audio = torch.topk(audio_logits[0], 5)
            print(f"[KimiAudio] Audio path: top5 audio token_ids={top5_audio.indices.tolist()}, "
                  f"top5 audio logits={top5_audio.values.tolist()}")
            # Check how many are in audio range
            num_audio_range = (top5_audio.indices >= 152064).sum().item()
            print(f"[KimiAudio] Audio path: {num_audio_range}/5 top tokens are in audio range [152064, 168447]")

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

        # Return as list of 2D tensors, one per batch item
        if audio_embeds.dim() == 3:
            # Unbind batch dimension: [B, T, D] -> list of B tensors [T, D]
            return list(audio_embeds.unbind(dim=0))
        else:
            # Single sample: [T, D] -> wrap in list
            return [audio_embeds]

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[dict]:
        """Parse and validate audio input from kwargs."""
        # Look for audio features in kwargs
        whisper_features = kwargs.get("whisper_input_features", None)
        feature_attention_mask = kwargs.get("feature_attention_mask", None)

        if whisper_features is None:
            return None

        return {
            "whisper_input_features": whisper_features,
            "feature_attention_mask": feature_attention_mask,
        }

    def _process_audio_input(self, audio_input: dict) -> torch.Tensor:
        """Process audio input through Whisper encoder and VQAdaptor."""
        whisper_features = audio_input["whisper_input_features"]

        # Run through Whisper encoder
        whisper_output = self.audio_tower(whisper_features)

        # Project Whisper output to VQ adaptor input dimension if needed
        if self.whisper_projection is not None:
            whisper_output = self.whisper_projection(whisper_output)

        # Run through VQAdaptor (multi_modal_projector)
        audio_embeds = self.multi_modal_projector(whisper_output)

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
            print(f"\n=== DEBUG: embed_input_ids multimodal ===", flush=True)
            print(f"input_ids shape: {input_ids.shape}", flush=True)
            print(f"multimodal_embeddings type: {type(multimodal_embeddings)}", flush=True)
            if isinstance(multimodal_embeddings, list):
                print(f"multimodal_embeddings is list with {len(multimodal_embeddings)} items", flush=True)
                for i, item in enumerate(multimodal_embeddings):
                    print(f"  Item {i}: type={type(item)}, ", end="", flush=True)
                    if hasattr(item, 'shape'):
                        print(f"shape={item.shape}, dtype={item.dtype}", flush=True)
                    elif item is None:
                        print(f"value=None", flush=True)
                    else:
                        print(f"value={item}", flush=True)
            else:
                print(f"multimodal_embeddings shape: {multimodal_embeddings.shape}", flush=True)
            if is_multimodal is not None:
                print(f"is_multimodal shape: {is_multimodal.shape}", flush=True)
                print(f"is_multimodal sum: {is_multimodal.sum().item()}", flush=True)
            print(f"=== END DEBUG ===\n", flush=True)

            # Ensure multimodal_embeddings is a tensor (not a list)
            if isinstance(multimodal_embeddings, list):
                # Filter out None items and check if list is empty
                multimodal_embeddings = [item for item in multimodal_embeddings if item is not None and isinstance(item, torch.Tensor)]
                if len(multimodal_embeddings) == 0:
                    # No valid embeddings, skip multimodal fusion
                    print(f"WARNING: No valid multimodal embeddings after filtering", flush=True)
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
                        num_audio_positions = is_multimodal.sum().item()

                        # Check if audio features match the number of multimodal positions
                        if audio_features.shape[0] == num_audio_positions:
                            # Create output tensor starting with text embeddings
                            result_emb = text_emb.clone()

                            # Place audio features at multimodal positions
                            multimodal_positions = torch.where(is_multimodal)[0]
                            result_emb[multimodal_positions] = audio_features

                            # Apply √2 scaling to multimodal positions
                            result_emb[multimodal_positions] = result_emb[multimodal_positions] * (2 ** 0.5)

                            text_emb = result_emb
                        else:
                            # Fallback: shapes don't match, use original logic with broadcasting
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

        # 3. Embed audio tokens (from previous generation step)
        # Use raw embedding weight to bypass VocabParallelEmbedding restrictions
        if self._pending_audio_token is not None:
            embed_weight = self.model.model.embed_tokens.weight
            # Clamp token IDs to valid range [0, vocab_size-1]
            audio_token_ids = self._pending_audio_token.clamp(0, embed_weight.shape[0] - 1)
            audio_emb = torch.nn.functional.embedding(audio_token_ids, embed_weight)

            # Squeeze extra dimensions if present
            if audio_emb.dim() > 2:
                audio_emb = audio_emb.squeeze(1)

            # CRITICAL: Only add audio embedding to the LAST position (current generation step)
            # text_emb shape: [num_tokens, hidden_dim]
            # audio_emb shape: [num_reqs, hidden_dim] or [1, hidden_dim]
            # We need to add audio_emb to the last token position only
            if text_emb.shape[0] > 1 and audio_emb.shape[0] == 1:
                # Single request: add audio_emb to the last position
                inputs_embeds = text_emb.clone()
                inputs_embeds[-1:] = inputs_embeds[-1:] + audio_emb
            elif text_emb.shape[0] == audio_emb.shape[0]:
                # Batched requests: add each audio_emb to corresponding last position
                # This assumes all requests have the same sequence length
                inputs_embeds = text_emb.clone()
                inputs_embeds[-1:] = inputs_embeds[-1:] + audio_emb[-1:]
            else:
                # Fallback: shapes don't match, skip audio fusion
                inputs_embeds = text_emb
        else:
            inputs_embeds = text_emb

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

        print(f"[KimiAudio] compute_logits: hidden_states shape={hidden_states.shape}, "
              f"logits shape={logits.shape if logits is not None else 'None'}, "
              f"hidden_mean={hidden_states.mean():.4f}, hidden_std={hidden_states.std():.4f}")
        if logits is not None:
            print(f"[KimiAudio] compute_logits: logits_mean={logits.mean():.4f}, logits_std={logits.std():.4f}, "
                  f"logits_max={logits.max():.4f}")
            # Print top 5 tokens for debugging
            top5 = torch.topk(logits[0], 5)
            print(f"[KimiAudio] compute_logits: top5 token_ids={top5.indices.tolist()}, "
                  f"top5 logits={top5.values.tolist()}")

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Custom sampler for dual token streaming.

        Samples both text and audio tokens from their respective logits.
        Audio tokens are stored for next step's embed_input_ids() fusion.
        """
        # Initialize stock sampler if needed (following HiggsAudioV2 pattern)
        sampler = getattr(self, "_stock_sampler", None)
        if sampler is None:
            from vllm.v1.sample.sampler import Sampler
            sampler = Sampler()
            self._stock_sampler = sampler

        # 1. Sample text token from text logits
        # Text logits are already filtered to [0, 152063] in compute_logits()
        text_sampler_output = sampler(logits=logits, sampling_metadata=sampling_metadata)
        text_tokens = text_sampler_output.sampled_token_ids  # [num_reqs, 1]

        # 2. Get audio logits (stored by forward() in self._pending_audio_logits)
        audio_logits = self._pending_audio_logits

        if audio_logits is None:
            # No audio logits available (shouldn't happen in normal operation)
            return text_sampler_output

        # 3. Sample audio token
        if self._generation_step < self._audio_delay:
            # First 6 tokens: force BLANK (token 18)
            audio_tokens = torch.full_like(text_tokens, self._blank_token_id)
        else:
            # Sample from audio logits using argmax (greedy for now)
            # audio_logits shape: [num_reqs, 168448] (full vocab)
            # We need to extract only the audio portion [152064, 168447]
            audio_logits_filtered = audio_logits[:, self._token_offset:]  # [num_reqs, 16384]
            audio_token_indices = torch.argmax(audio_logits_filtered, dim=-1, keepdim=True)  # [num_reqs, 1]
            # Add offset to get actual audio token IDs [152064, 168447]
            audio_tokens = audio_token_indices + self._token_offset

        # 4. Store audio token for next step's embed_input_ids()
        self._pending_audio_token = audio_tokens

        # 5. Increment generation step
        self._generation_step += 1

        # 6. Debug logging
        if self._generation_step <= 10 or self._generation_step % 50 == 0:
            print(f"[KimiAudio] sample step {self._generation_step}: "
                  f"text_token={text_tokens[0].item() if text_tokens.numel() > 0 else 'N/A'}, "
                  f"audio_token={audio_tokens[0].item() if audio_tokens.numel() > 0 else 'N/A'}")

        # 7. Return text tokens (drives the engine)
        return text_sampler_output

    def make_omni_output(
        self,
        model_outputs: dict,
        **kwargs,
    ) -> OmniOutput:
        """Package dual-stream output into OmniOutput.

        Includes audio tokens for feedback to next step via next_token_id.
        """
        text_hidden = model_outputs.get("text_hidden_states")
        audio_logits = model_outputs.get("audio_logits")

        return OmniOutput(
            text_hidden_states=text_hidden,
            multimodal_outputs={
                "audio_logits": audio_logits,
                "audio_tokens": self._pending_audio_token,
            },
            next_token_id=self._pending_audio_token,  # Critical for feedback!
        )

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        **req_infos,
    ) -> dict:
        """Per-request postprocessing to sync state.

        Returns update dict that gets merged into model_intermediate_buffer.
        """
        return {
            "audio_token": self._pending_audio_token,
            "generation_step": self._generation_step,
        }

    def on_requests_finished(self, finished_req_ids: list[str]) -> None:
        """Reset state for finished requests.

        Prevents cross-request contamination.
        """
        # Reset dual streaming state
        self._pending_audio_token = None
        self._pending_audio_logits = None
        self._generation_step = 0

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
            print(f"[KimiAudio] Loading {len(mimo_layers_weights)} MIMO layer weights")
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
                print(f"[KimiAudio] WARNING: Missing MIMO weights: {missing[:5]}...")
            if unexpected:
                print(f"[KimiAudio] WARNING: Unexpected MIMO weights: {unexpected[:5]}...")
            print(f"[KimiAudio] MIMO layers loaded: {len(mimo_state_dict)} weights")

        # Load audio output head
        if mimo_output_weights:
            print(f"[KimiAudio] Loading {len(mimo_output_weights)} mimo_output weights")
            mimo_output_state_dict = {k: v for k, v in mimo_output_weights}
            missing, unexpected = self.mimo_output.load_state_dict(mimo_output_state_dict, strict=False)
            if missing:
                print(f"[KimiAudio] WARNING: Missing mimo_output weights: {missing}")
            if unexpected:
                print(f"[KimiAudio] WARNING: Unexpected mimo_output weights: {unexpected}")
            print(f"[KimiAudio] mimo_output loaded successfully")
            # Verify weight shape
            print(f"[KimiAudio] mimo_output.weight shape: {self.mimo_output.weight.shape}, "
                  f"mean: {self.mimo_output.weight.mean():.6f}, std: {self.mimo_output.weight.std():.6f}")

        if mimo_norm_weights:
            print(f"[KimiAudio] Loading {len(mimo_norm_weights)} mimo_norm weights")
            missing, unexpected = self.mimo_norm.load_state_dict(
                {k: v for k, v in mimo_norm_weights},
                strict=False
            )
            if missing:
                print(f"[KimiAudio] WARNING: Missing mimo_norm weights: {missing}")
            if unexpected:
                print(f"[KimiAudio] WARNING: Unexpected mimo_norm weights: {unexpected}")
            print(f"[KimiAudio] mimo_norm loaded successfully")

        # Load main model (Qwen2 backbone)
        # Pass weights as-is - parameters are named "model.layers.*"
        if model_weights:
            print(f"[KimiAudio] Loading {len(model_weights)} main model weights")
            # Show first few weight names for debugging
            for name, tensor in model_weights[:5]:
                print(f"[KimiAudio] Main model weight: {name} shape={tensor.shape}")
            self.model.load_weights(model_weights)
            print(f"[KimiAudio] Main model weights loaded successfully")

            # Debug: Check if embeddings are loaded
            embed_weight = self.model.model.embed_tokens.weight
            lm_head_weight = self.model.lm_head.weight
            print(f"[KimiAudio] Embeddings loaded: shape={embed_weight.shape}, "
                  f"mean={embed_weight.mean():.6f}, std={embed_weight.std():.6f}")
            print(f"[KimiAudio] LM head loaded: shape={lm_head_weight.shape}, "
                  f"mean={lm_head_weight.mean():.6f}, std={lm_head_weight.std():.6f}")

            # Check a few specific weights to verify they're not random
            print(f"[KimiAudio] Embed weight sample [0,:5]: {embed_weight[0, :5].tolist()}")
            print(f"[KimiAudio] LM head weight sample [0,:5]: {lm_head_weight[0, :5].tolist()}")

            # Check layer 0 weights
            layer0_attn_q = self.model.model.layers[0].self_attn.qkv_proj.weight
            print(f"[KimiAudio] Layer 0 attention qkv weight: shape={layer0_attn_q.shape}, "
                  f"mean={layer0_attn_q.mean():.6f}, std={layer0_attn_q.std():.6f}")
