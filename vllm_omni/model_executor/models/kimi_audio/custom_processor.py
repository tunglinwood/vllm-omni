# Copyright 2025 vLLM-Omni Team
"""Custom multi-modal processor for Kimi Audio that passes audio_tokens."""

from typing import Any

from transformers import BatchFeature
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.models.kimi_audio import (
    KimiAudioMultiModalProcessor as BaseKimiAudioMultiModalProcessor,
    KimiAudioProcessingInfo,
    KimiAudioDummyInputsBuilder,
)

logger = init_logger(__name__)


class CustomKimiAudioMultiModalProcessor(BaseKimiAudioMultiModalProcessor):
    """Custom processor that passes audio_tokens to the model."""

    def _get_mm_fields_config(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
    ) -> dict[str, MultiModalFieldConfig]:
        """Override to add audio_tokens field."""
        logger.debug("_get_mm_fields_config called with mm_data keys: %s", list(mm_data.keys()))

        # Get base config from parent
        base_config = super()._get_mm_fields_config(prompt, mm_data)

        logger.debug("Base config keys: %s", list(base_config.keys()))

        # Add audio_tokens field if present
        if "audio_tokens" in mm_data:
            base_config["audio_tokens"] = MultiModalFieldConfig(
                batched=False,
                modalities=("audio",),
            )
            logger.debug("Added audio_tokens to mm_fields_config")

        return base_config

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        mm_kwargs: dict[str, Any],
        tok_kwargs: dict[str, Any],
    ) -> BatchFeature:
        """Override to pass audio_tokens through."""
        logger.warning("_call_hf_processor called with mm_data keys: %s", list(mm_data.keys()))
        logger.warning("mm_data types: %s", {k: type(v) for k, v in mm_data.items()})

        # Call parent processor to get whisper features
        hf_inputs = super()._call_hf_processor(
            prompt, mm_data, mm_kwargs, tok_kwargs
        )

        logger.warning("After parent processor, hf_inputs keys: %s", list(hf_inputs.keys()))
        for key in hf_inputs.keys():
            val = hf_inputs[key]
            if hasattr(val, 'shape'):
                logger.warning("  %s: shape=%s, dtype=%s", key, val.shape, val.dtype)
            else:
                logger.warning("  %s: type=%s", key, type(val))

        # Pass through audio_tokens if present
        if "audio_tokens" in mm_data:
            hf_inputs["audio_tokens"] = mm_data["audio_tokens"]
            logger.warning("Added audio_tokens to hf_inputs: %d tokens", len(mm_data['audio_tokens']))
        else:
            logger.warning("No audio_tokens in mm_data")

        return hf_inputs
