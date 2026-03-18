# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kimi-Audio Talker model for TTS generation.

Extends vLLM's KimiAudioForConditionalGeneration (ASR backbone) with TTS-specific MIMO layers.
This follows the vLLM-Omni pattern: reuse vLLM backbone, add modality-specific extensions.

Architecture:
- Backbone: 28 shared layers from vLLM's KimiAudio (handles text understanding)
- MIMO layers 0-5: Audio generation transformer (bifurcates from final backbone output)
- mimo_output: Audio token logits (vocab 152064-168447)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_kimi_audio_tts import KimiAudioTalkerConfig

logger = init_logger(__name__)


# =============================================================================
# Reusable components imported from vLLM's Kimi-Audio ASR implementation
# =============================================================================
from vllm.model_executor.models.kimi_audio import (
    KimiAudioWhisperEncoder,
    KimiAudioMultiModalProjector,
    KimiAudioProcessingInfo,
    KimiAudioDummyInputsBuilder,
    KimiAudioMultiModalDataParser,
    KimiAudioMultiModalProcessor,
    _get_feat_extract_output_lengths,
)

# Kimi-Audio constants (define locally to avoid import issues)
KIMIA_WHISPER_SUBFOLDER = "whisper-large-v3"

from vllm.tokenizers.kimi_audio import KimiAudioTokenizer

from vllm.transformers_utils.processors.kimi_audio import (
    KimiAudioProcessor,
)

# =============================================================================
# TTS-specific MIMO decoder layer
# =============================================================================


class KimiAudioMIMODecoderLayer(nn.Module):
    """MIMO decoder layer for audio generation path.
    
    Matches backbone decoder layer architecture:
    - GQA attention (28 heads Q, 4 heads K/V)
    - Same dimensions and activation
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_position_embeddings: int,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_heads = num_key_value_heads
        
        # Separate Q/K/V projections (matches checkpoint structure)
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=True,
            return_bias=False,
            gather_output=True,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=True,
            return_bias=False,
            gather_output=True,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=True,
            return_bias=False,
            gather_output=True,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
        )
        
        # MLP - separate gate/up projections (matches checkpoint)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            intermediate_size,
            bias=False,
            return_bias=False,
            gather_output=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            intermediate_size,
            bias=False,
            return_bias=False,
            gather_output=True,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
        )
        
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters={'rope_theta': rope_theta},
            dual_chunk_attention_config=None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for MIMO decoder layer."""
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        
        # Handle both 2D (profile) and 3D formats
        if hidden_states.dim() == 2:
            # Profile run: [batch*seq, hidden]
            # Linear layers may return tensor or (tensor, None)
            result = self.q_proj(hidden_states)
            query_states = result[0] if isinstance(result, tuple) else result
            result = self.o_proj(query_states)
            output = result[0] if isinstance(result, tuple) else result
            return residual + output
        
        # Normal inference: [batch, seq, hidden]
        bsz, q_len, _ = hidden_states.size()
        
        # Separate Q/K/V projections
        result = self.q_proj(hidden_states)
        query_states = result[0] if isinstance(result, tuple) else result
        result = self.k_proj(hidden_states)
        key_states = result[0] if isinstance(result, tuple) else result
        result = self.v_proj(hidden_states)
        value_states = result[0] if isinstance(result, tuple) else result
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, position_ids)
        from vllm.model_executor.layers.rotary_embedding import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Simple attention (no KV cache for TTS)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        result = self.o_proj(attn_output)
        attn_output = result[0] if isinstance(result, tuple) else result
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        result = self.gate_proj(hidden_states)
        gate = result[0] if isinstance(result, tuple) else result
        gate = torch.nn.functional.silu(gate)
        result = self.up_proj(hidden_states)
        up = result[0] if isinstance(result, tuple) else result
        result = self.down_proj(gate * up)
        hidden_states = result[0] if isinstance(result, tuple) else result
        
        return residual + hidden_states


# =============================================================================
# Kimi-Audio TTS Talker - extends vLLM's ASR with TTS capability
# =============================================================================


class KimiAudioTalkerForConditionalGeneration(nn.Module):
    """Kimi-Audio TTS Talker.
    
    Extends vLLM's KimiAudioForConditionalGeneration (ASR) with TTS-specific MIMO layers.
    
    Reuses from vLLM:
    - KimiAudioWhisperEncoder (for future audio input support)
    - KimiAudioMultiModalProjector (VQ-Adaptor)
    - KimiAudioProcessor, KimiAudioTokenizer (input processing)
    - Weight mappings and architecture constants
    
    Adds in vLLM-Omni:
    - MIMO layers 0-5 (audio generation)
    - mimo_norm and mimo_output (TTS output head)
    """
    
    # Use same weight mappings as vLLM's ASR + add TTS-specific mappings
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Backbone (shared with ASR) - same as vLLM's kimi_audio.py
            "model.layers.": "language_model.model.layers.",
            "model.embed_tokens.": "language_model.model.embed_tokens.",
            "model.norm.": "language_model.model.norm.",
            "lm_head.": "language_model.lm_head.",
            # VQ-Adaptor (audio projector) - same as vLLM's kimi_audio.py
            "model.vq_adaptor.layers.0.": "multi_modal_projector.vq_adaptor_layers_0.",
            "model.vq_adaptor.layers.3.": "multi_modal_projector.vq_adaptor_layers_3.",
            "model.vq_adaptor.layers.4.": "multi_modal_projector.vq_adaptor_layers_4.",
            # MIMO layers (TTS-specific) - vLLM-Omni extension
            "model.mimo_layers.": "mimo_layers.",
            "model.mimo_norm.": "mimo_norm.",
            "model.mimo_output.": "mimo_output.",
        },
        orig_to_new_substr={
            # Map MLP submodule to direct projections for MIMO layers
            "mimo_layers..mlp.": "mimo_layers.",
        }
    )
    
    # Explicitly mark as text generation model for vLLM registry
    is_text_generation_model: bool = True
    
    # Flag for vLLM-Omni runner to extract multimodal outputs
    have_multimodal_outputs: bool = True
    
    # For generation scheduler, we handle sampling in forward() directly
    has_preprocess: bool = False
    has_postprocess: bool = False
    
    # MTP attributes for AR scheduler (minimal stub for Kimi-Audio)
    mtp_hidden_size: int = 3584  # Same as hidden_size
    talker_mtp_output_key: str = "audio_codes"
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model_path = vllm_config.model_config.model
        
        # Initialize vLLM backbone using Qwen2ForCausalLM (handles weight merging correctly)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config.with_hf_config(
                self.config, architectures=["Qwen2ForCausalLM"]
            ),
            prefix=maybe_prefix(prefix, "language_model"),
        )
        
        # VQ-Adaptor (audio projector) - for ASR compatibility
        # Reuses vLLM's KimiAudioMultiModalProjector structure
        self.multi_modal_projector = KimiAudioMultiModalProjector(
            whisper_dim=getattr(self.config, 'kimia_adaptor_input_dim', 5120),
            llm_dim=self.config.hidden_size,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )
        
        # TTS-specific MIMO layers (6 layers for audio generation)
        # Use config attributes directly from HF config
        self.mimo_layers = nn.ModuleList([
            KimiAudioMIMODecoderLayer(
                hidden_size=self.config.hidden_size,
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                intermediate_size=self.config.intermediate_size,
                rms_norm_eps=self.config.rms_norm_eps,
                rope_theta=self.config.rope_theta,
                max_position_embeddings=self.config.max_position_embeddings,
                prefix=f"{prefix}.mimo_layers.{i}",
            )
            for i in range(6)
        ])
        self.mimo_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        
        # Audio output projection (full vocab, audio tokens are 152064-168447)
        self.mimo_output = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
            return_bias=False,
            gather_output=True,
        )
        
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        
        # Set MTP hidden size from config
        self.mtp_hidden_size = int(self.config.hidden_size)
        
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input IDs using the backbone's embedding layer."""
        return self.language_model.model.embed_tokens(input_ids)
    
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MTP (Multi-Token Prediction) for AR scheduler.
        
        Minimal stub for Kimi-Audio (single codebook).
        Predicts next audio tokens given previous hidden state.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            input_embeds: Input embeddings [batch, seq_len, hidden]
            last_talker_hidden: Previous step hidden states [batch, hidden]
            text_step: Text step indicator tensor
            
        Returns:
            tuple: (inputs_embeds_for_next_step, audio_codes)
                - inputs_embeds: [batch, hidden] for next MTP step
                - audio_codes: [batch, 1] predicted audio codes (single codebook)
        """
        # Project last hidden state to audio space using MIMO output
        audio_logits, _ = self.mimo_output(last_talker_hidden)
        
        # Sample audio tokens (argmax for MTP fast-path)
        audio_token_ids = torch.argmax(audio_logits, dim=-1)  # [batch]
        
        # Format as single codebook [batch, 1]
        audio_codes = audio_token_ids.unsqueeze(1)
        
        # Get embeddings for next MTP step
        next_embeds = self.embed_input_ids(input_ids)
        if next_embeds.dim() == 3:
            # Take last token's embedding
            next_embeds = next_embeds[:, -1, :]  # [batch, hidden]
        
        return next_embeds, audio_codes
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Forward pass for TTS generation.
        
        Args:
            input_ids: Input token IDs
            positions: Position IDs
            intermediate_tensors: For pipeline parallelism
            inputs_embeds: Optional pre-computed embeddings
            
        Returns:
            Audio hidden states for token prediction
        """
        # Run backbone (all 28 layers)
        # Note: We don't pass **kwargs to language_model to avoid unexpected arguments
        backbone_output = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        
        # Get hidden states from backbone
        if isinstance(backbone_output, dict):
            hidden_states = backbone_output.get('hidden_states')
        elif isinstance(backbone_output, torch.Tensor):
            hidden_states = backbone_output
        else:
            hidden_states = backbone_output[0] if isinstance(backbone_output, (tuple, list)) else backbone_output
        
        # AUDIO PATH: Run through MIMO layers (0-5)
        mimo_hidden_states = hidden_states.clone()
        
        # Generate position IDs for MIMO layers if not provided
        if mimo_hidden_states.dim() == 3:
            bsz, seq_len, _ = mimo_hidden_states.shape
            if positions is None or positions.shape[-1] != seq_len:
                positions = torch.arange(seq_len, device=mimo_hidden_states.device).unsqueeze(0).expand(bsz, -1)
        else:
            # 2D case (profile runs): [batch*seq, hidden]
            # Create dummy position_ids
            seq_len = mimo_hidden_states.shape[0]
            positions = torch.arange(seq_len, device=mimo_hidden_states.device).unsqueeze(0)
        
        # Run through MIMO layers
        for mimo_layer in self.mimo_layers:
            mimo_hidden_states = mimo_layer(mimo_hidden_states, positions)
        
        # Apply final norm
        audio_hidden_states = self.mimo_norm(mimo_hidden_states)
        
        # Return audio hidden states for AR worker to sample
        # The make_omni_output() method will create OmniOutput with audio codes
        logger.info(f"[KimiAudio-TTS] forward: returning audio_hidden_states shape={audio_hidden_states.shape}")
        return audio_hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """Compute logits for audio tokens."""
        if hidden_states is None:
            return None
        
        # logits_processor expects (lm_head, hidden_states)
        logits = self.logits_processor(self.mimo_output, hidden_states)
        
        return logits
    
    def postprocess(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Postprocess to sample audio tokens from hidden states.
        
        Called by vLLM-Omni runner after model forward to extract audio codes.
        
        Args:
            hidden_states: Audio hidden states from MIMO layers [batch*seq, hidden]
            **kwargs: Additional information from intermediate buffer
            
        Returns:
            Dict with audio_codes to store in intermediate buffer
        """
        logger.info(f"[KimiAudio-TTS] postprocess called, hidden_states shape: {hidden_states.shape if hidden_states is not None else None}, kwargs keys: {list(kwargs.keys())}")
        
        update_dict = {}
        
        # Sample audio tokens from hidden states using mimo_output
        if hasattr(self, 'mimo_output') and hidden_states is not None:
            # Compute logits
            logits, _ = self.mimo_output(hidden_states)
            logger.info(f"[KimiAudio-TTS] logits shape: {logits.shape}")
            
            # Sample from logits (simple argmax for now, can add temperature/top-k later)
            # logits shape: [batch*seq, vocab_size]
            audio_token_ids = torch.argmax(logits, dim=-1)  # [batch*seq]
            logger.info(f"[KimiAudio-TTS] audio_token_ids shape: {audio_token_ids.shape}, min: {audio_token_ids.min()}, max: {audio_token_ids.max()}")
            
            # Reshape to match expected format [seq_len] for single request
            if audio_token_ids.dim() == 1:
                audio_codes = audio_token_ids.unsqueeze(0)  # [1, seq_len]
            else:
                audio_codes = audio_token_ids
            
            update_dict["audio_codes"] = audio_codes
            update_dict["audio_hidden_states"] = hidden_states
            logger.info(f"[KimiAudio-TTS] audio_codes shape: {audio_codes.shape}")
        else:
            logger.warning(f"[KimiAudio-TTS] postprocess: mimo_output={hasattr(self, 'mimo_output')}, hidden_states={hidden_states is not None}")
        
        return update_dict
    
    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Create OmniOutput from audio hidden states.
        
        Args:
            model_outputs: Audio hidden states from forward() [batch*seq, hidden]
            **kwargs: May contain model_intermediate_buffer with audio_codes
            
        Returns:
            OmniOutput with audio_codes
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        
        # First try to get audio codes from intermediate buffer (populated by postprocess)
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []
        
        audio_codes_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            ac = info.get("audio_codes")
            if isinstance(ac, torch.Tensor):
                audio_codes_list.append(ac)
        
        if audio_codes_list:
            audio_codes = torch.cat(audio_codes_list, dim=0)
            logger.info(f"[KimiAudio-TTS] make_omni_output: audio_codes shape={audio_codes.shape} (from buffer)")
            return OmniOutput(
                text_hidden_states=model_outputs,
                multimodal_outputs={"audio_codes": audio_codes},
            )
        
        # Fallback: sample audio codes directly from hidden states
        # This is needed for AR worker which doesn't call postprocess()
        if isinstance(model_outputs, torch.Tensor) and hasattr(self, 'mimo_output'):
            # Compute logits and sample
            logits_result = self.mimo_output(model_outputs)
            logits = logits_result[0] if isinstance(logits_result, tuple) else logits_result
            audio_token_ids = torch.argmax(logits, dim=-1)  # [batch*seq]
            
            # Reshape to [1, seq_len] for single request
            if audio_token_ids.dim() == 1:
                audio_codes = audio_token_ids.unsqueeze(0)
            else:
                audio_codes = audio_token_ids
            
            logger.info(f"[KimiAudio-TTS] make_omni_output: audio_codes shape={audio_codes.shape} (sampled)")
            return OmniOutput(
                text_hidden_states=model_outputs,
                multimodal_outputs={"audio_codes": audio_codes},
            )
        
        # Last resort: empty OmniOutput
        logger.warning("[KimiAudio-TTS] make_omni_output: creating empty OmniOutput")
        return OmniOutput(
            text_hidden_states=model_outputs if isinstance(model_outputs, torch.Tensor) else None,
            multimodal_outputs={},
        )
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader.
        
        Loads:
        - Backbone weights (28 layers) via vLLM's Qwen2 pattern (handles merged gate_up_proj)
        - VQ-Adaptor weights (for audio feature projection, if present)
        - MIMO layers (TTS-specific) - handles separate gate/up projections
        - mimo_norm and mimo_output
        """
        # Transform weights: handle MIMO layer structure
        def transform_weights(ws):
            for name, tensor in ws:
                # Map checkpoint structure to our implementation
                if name.startswith("model.mimo_layers."):
                    # Remove .self_attn. and .mlp. submodules
                    name = name.replace(".self_attn.", ".")
                    name = name.replace(".mlp.", ".")
                yield name, tensor
        
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(transform_weights(weights), mapper=self.hf_to_vllm_mapper)
        
        logger.info(f"Loaded {len(loaded)} weights for KimiAudioTalker (including MIMO layers)")
        return loaded
