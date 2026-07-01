# Copyright 2025 vLLM-Omni Team
"""Top-level dispatcher for Kimi Audio model."""

import os
from typing import Optional

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.sequence import IntermediateTensors
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.kimi_audio.custom_processor import (
    CustomKimiAudioMultiModalProcessor,
)
from vllm.model_executor.models.kimi_audio import (
    KimiAudioProcessingInfo,
    KimiAudioDummyInputsBuilder,
)


@MULTIMODAL_REGISTRY.register_processor(
    CustomKimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder,
)
class KimiAudioForConditionalGeneration(nn.Module):
    """Top-level model that dispatches to stage 0 (LLM) or stage 1 (detokenizer).

    The stage is determined by the MODEL_STAGE environment variable:
    - "fused_llm" (default): Stage 0 - LLM with dual output heads
    - "audio_detokenizer": Stage 1 - Flow-matching detokenizer + vocoder
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix

        # Check model_stage env var (set by stage runner)
        model_stage = os.environ.get("MODEL_STAGE", "fused_llm")

        if model_stage == "fused_llm":
            # Stage 0: LLM with dual output heads
            from .kimi_audio_llm import KimiAudioLLMForConditionalGeneration
            self.model = KimiAudioLLMForConditionalGeneration(
                vllm_config=vllm_config, prefix=prefix
            )
        elif model_stage == "audio_detokenizer":
            # Stage 1: Flow-matching detokenizer + vocoder
            from .kimi_audio_detokenizer import KimiAudioDetokenizerForConditionalGeneration
            self.model = KimiAudioDetokenizerForConditionalGeneration(
                vllm_config=vllm_config, prefix=prefix
            )
        else:
            raise ValueError(f"Unknown MODEL_STAGE: {model_stage}")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        multimodal_embeddings: Optional[torch.Tensor] = None,
    ) -> OmniOutput:
        """Forward pass - delegates to the appropriate stage model."""
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            multimodal_embeddings=multimodal_embeddings,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: any,
    ) -> Optional[torch.Tensor]:
        """Compute logits - delegates to the appropriate stage model."""
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states, sampling_metadata)
        return None

    def embed_multimodal(self, **kwargs) -> Optional[list[torch.Tensor]]:
        """Process multimodal inputs - delegates to the appropriate stage model."""
        if hasattr(self.model, "embed_multimodal"):
            return self.model.embed_multimodal(**kwargs)
        return None

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Load weights - delegates to the appropriate stage model."""
        self.model.load_weights(weights)
