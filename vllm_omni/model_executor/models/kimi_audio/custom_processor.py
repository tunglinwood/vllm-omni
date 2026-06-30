# Copyright 2025 vLLM-Omni Team
"""Custom multi-modal processor for Kimi Audio that passes audio_tokens."""

from typing import Any

from transformers import BatchFeature
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.kimi_audio import (
    KimiAudioMultiModalProcessor as BaseKimiAudioMultiModalProcessor,
    KimiAudioProcessingInfo,
    KimiAudioDummyInputsBuilder,
)


class CustomKimiAudioMultiModalProcessor(BaseKimiAudioMultiModalProcessor):
    """Custom processor that passes audio_tokens to the model."""

    def _get_mm_fields_config(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
    ) -> dict[str, MultiModalFieldConfig]:
        """Override to add audio_tokens field."""
        print(f"[CustomProcessor] _get_mm_fields_config called with mm_data keys: {list(mm_data.keys())}", flush=True)

        # Get base config from parent
        base_config = super()._get_mm_fields_config(prompt, mm_data)

        print(f"[CustomProcessor] Base config keys: {list(base_config.keys())}", flush=True)

        # Add audio_tokens field if present
        if "audio_tokens" in mm_data:
            base_config["audio_tokens"] = MultiModalFieldConfig(
                batched=False,
                modalities=("audio",),
            )
            print(f"[CustomProcessor] ✅ Added audio_tokens to mm_fields_config", flush=True)

        return base_config

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        mm_kwargs: dict[str, Any],
        tok_kwargs: dict[str, Any],
    ) -> BatchFeature:
        """Override to pass audio_tokens through."""
        print(f"[CustomProcessor] _call_hf_processor called with mm_data keys: {list(mm_data.keys())}", flush=True)

        # Call parent processor to get whisper features
        hf_inputs = super()._call_hf_processor(
            prompt, mm_data, mm_kwargs, tok_kwargs
        )

        print(f"[CustomProcessor] After parent processor, hf_inputs keys: {list(hf_inputs.keys())}", flush=True)

        # Pass through audio_tokens if present
        if "audio_tokens" in mm_data:
            hf_inputs["audio_tokens"] = mm_data["audio_tokens"]
            print(f"[CustomProcessor] ✅ Added audio_tokens to hf_inputs: {len(mm_data['audio_tokens'])} tokens", flush=True)
        else:
            print(f"[CustomProcessor] No audio_tokens in mm_data", flush=True)

        return hf_inputs
