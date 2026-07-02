# Copyright 2025 vLLM-Omni Team
"""Custom multi-modal processor for Kimi Audio that passes audio_tokens."""

import sys
from typing import Any, Sequence

from transformers import BatchFeature
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import PromptReplacement
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.models.kimi_audio import (
    KimiAudioMultiModalProcessor as BaseKimiAudioMultiModalProcessor,
    KimiAudioProcessingInfo,
    KimiAudioDummyInputsBuilder,
)
from vllm.transformers_utils.processors.kimi_audio import KimiAudioProcessor

logger = init_logger(__name__)


class CustomKimiAudioMultiModalProcessor(BaseKimiAudioMultiModalProcessor):
    """Custom processor that passes audio_tokens through."""

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: dict[str, Any],
    ) -> dict[str, MultiModalFieldConfig]:
        """Override to add audio_tokens field."""
        logger.warning("_get_mm_fields_config called with hf_inputs keys: %s", list(hf_inputs.keys()))

        # Get base config from parent
        base_config = super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)

        logger.warning("Base config keys: %s", list(base_config.keys()))

        # Add audio_tokens field if present
        if "audio_tokens" in hf_inputs:
            base_config["audio_tokens"] = MultiModalFieldConfig(
                batched=False,
                modalities=("audio",),
            )
            logger.warning("Added audio_tokens to mm_fields_config")

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
        # Diagnostic: dump each mm_data value
        for k, v in mm_data.items():
            v_repr = repr(v)[:300]
            logger.warning("mm_data[%r] = %s", k, v_repr)

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
                logger.warning("  %s: type=%s repr=%s", key, type(val), repr(val)[:200])

        # Pass through audio_tokens if present
        if "audio_tokens" in mm_data:
            hf_inputs["audio_tokens"] = mm_data["audio_tokens"]
            logger.warning("Added audio_tokens to hf_inputs: %d tokens", len(mm_data['audio_tokens']))
        else:
            logger.warning("No audio_tokens in mm_data")

        # Pass through raw audio arrays for discrete tokenization
        # The HF processor consumes mm_data['audio'] (numpy arrays) and returns whisper features
        # But we need the raw audio back for Glm4Tokenizer - stash it in hf_inputs
        audios = mm_data.get("audios", mm_data.get("audio", None))
        if audios is not None:
            hf_inputs["_raw_audio_for_tokenizer"] = audios
            logger.warning("Stashed raw audio in hf_inputs['_raw_audio_for_tokenizer'], type=%s, len=%s",
                           type(audios).__name__, len(audios) if hasattr(audios, '__len__') else 'N/A')

        return hf_inputs

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ) -> Sequence[PromptReplacement]:
        """Override to add diagnostic logging for prompt replacement."""
        print(f"[_get_prompt_updates] CALLED", file=sys.stderr, flush=True)
        # Diagnostic: dump out_mm_kwargs contents
        out_mm_data = out_mm_kwargs.get_data()
        print(f"[_get_prompt_updates] out_mm_data keys={list(out_mm_data.keys())}", file=sys.stderr, flush=True)
        for k, v in out_mm_data.items():
            if hasattr(v, 'shape'):
                print(f"[_get_prompt_updates]   {k}: shape={tuple(v.shape)}, dtype={v.dtype}", file=sys.stderr, flush=True)
            else:
                print(f"[_get_prompt_updates]   {k}: type={type(v).__name__}", file=sys.stderr, flush=True)

        whisper_features = out_mm_data.get("whisper_input_features")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")

        # Determine number of mel frames from whisper features if available,
        # else fall back to feature_attention_mask.
        if whisper_features is not None and hasattr(whisper_features, "shape"):
            # whisper_features: [B, num_mel_bins, num_frames]
            num_frames = int(whisper_features.shape[-1])
            print(f"[_get_prompt_updates] Using whisper_features num_frames={num_frames}", file=sys.stderr, flush=True)
        elif feature_attention_mask is not None:
            # feature_attention_mask: [B, num_frames]
            num_frames = int(feature_attention_mask.sum(-1).item())
            print(f"[_get_prompt_updates] Using feature_attention_mask sum={num_frames}", file=sys.stderr, flush=True)
        else:
            num_frames = 376  # fallback default
            print(f"[_get_prompt_updates] No features/mask, fallback num_frames={num_frames}", file=sys.stderr, flush=True)

        # Reference implementation formula (from KimiAudioCustomWhisperEncoder):
        #   L = num_frames * hop_length (160)
        #   token_len = (L - 1) // (hop_length * 8) + 1
        #   target_length = token_len * 4
        #
        # Simplified: target_length = ((num_frames - 1) // 8 + 1) * 4
        #
        # The custom Whisper encoder pads/slices to this length after the
        # standard stride-2 conv (which produces num_frames // 2 outputs).
        # The vLLM `_get_feat_extract_output_lengths` formula gives different
        # (incorrect for Kimi) values, so we use the reference formula here.
        audio_output_lens_value = ((num_frames - 1) // 8 + 1) * 4
        audio_output_lengths = [audio_output_lens_value]

        print(f"[_get_prompt_updates] whisper_features={'present' if whisper_features is not None else 'MISSING'}, feature_attention_mask={'present' if feature_attention_mask is not None else 'MISSING'}, num_frames={num_frames}, audio_output_lengths={audio_output_lengths}", file=sys.stderr, flush=True)

        def get_replacement_kimiaudio(item_idx: int):
            num_features = (
                audio_output_lengths[item_idx]
                if item_idx < len(audio_output_lengths)
                else 376
            )
            if num_features == 0:
                num_features = 376
            print(f"[_get_prompt_updates] get_replacement_kimiaudio({item_idx}) -> {num_features} blank tokens", file=sys.stderr, flush=True)
            return [KimiAudioProcessor.KIMIA_TEXT_BLANK] * num_features

        # Use the token ID as target
        replacements = [
            PromptReplacement(
                modality="audio",
                target=[KimiAudioProcessor.KIMIA_TEXT_BLANK],
                replacement=get_replacement_kimiaudio,
            ),
        ]
        print(f"[_get_prompt_updates] returning {len(replacements)} PromptReplacements, target={replacements[0].target}", file=sys.stderr, flush=True)
        return replacements
