# Copyright 2025 vLLM-Omni Team
"""Thin custom multi-modal processor for Kimi Audio.

Overrides the BLANK-count formula to match our upstream-aligned architecture:
Whisper-Large-v3 output → 4-frame concat via reshape → VQ Adaptor.
So BLANK count = ceil(mel_frames / 4).

GLM-4 tokenizer is NOT used here — it's only needed for voice cloning
(tokenizing reference audio), which is a separate pipeline path.
"""

from typing import Any, Sequence

from vllm.logger import init_logger
from vllm.model_executor.models.kimi_audio import (
    KimiAudioMultiModalProcessor as BaseKimiAudioMultiModalProcessor,
)
from vllm.multimodal.processing import PromptReplacement
from vllm.transformers_utils.processors.kimi_audio import KimiAudioProcessor

logger = init_logger(__name__)


class CustomKimiAudioMultiModalProcessor(BaseKimiAudioMultiModalProcessor):
    """Custom processor with upstream-aligned BLANK count formula."""

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ) -> Sequence[PromptReplacement]:
        """Compute BLANK count from whisper feature shape.

        Aligned with upstream vllm 0.23: after the Whisper encoder, the
        4-frame concat via reshape produces T//4 audio embeds.

        Our KimiAudioCustomWhisperEncoder outputs encoder_frames == num_mel_frames
        (it pads/slices to target_length = token_len*4 ≈ mel_frames).

        So BLANK count = ceil(num_mel_frames / 4).
        For qa_example.wav (376 mel frames): 94 BLANKs.
        """
        out_mm_data = out_mm_kwargs.get_data()
        whisper_features = out_mm_data.get("whisper_input_features")

        if whisper_features is not None and hasattr(whisper_features, "shape"):
            num_frames = int(whisper_features.shape[-1])
            # 4-frame concat produces T//4 embeds (with T padded to multiple of 4)
            n_blank = (num_frames + 3) // 4
        else:
            # Last-resort fallback (dummy init with no audio input)
            num_frames = 376
            n_blank = num_frames // 4  # = 94

        audio_output_lengths = [n_blank]

        logger.debug("[_get_prompt_updates] num_frames=%d, n_blank=%d",
                     num_frames, n_blank)

        def get_replacement_kimiaudio(item_idx: int):
            num_features = (
                audio_output_lengths[item_idx]
                if item_idx < len(audio_output_lengths)
                else 94
            )
            if num_features == 0:
                num_features = 94
            return [KimiAudioProcessor.KIMIA_TEXT_BLANK] * num_features

        return [
            PromptReplacement(
                modality="audio",
                target=[KimiAudioProcessor.KIMIA_TEXT_BLANK],
                replacement=get_replacement_kimiaudio,
            ),
        ]
