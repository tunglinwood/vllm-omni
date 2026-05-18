# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimia Audio (Kimi-Audio) TTS model support.

Provides model classes for Kimi-Audio TTS pipeline:
- KimiaAudioFusedForConditionalGeneration: AR stage (text -> audio codes)
- KimiaAudioCode2WavForConditionalGeneration: code2wav stage (audio codes -> waveform)
"""

from vllm_omni.model_executor.models.kimia_audio.kimia_audio_ar_stage import (
    KimiaAudioFusedForConditionalGeneration,
)
from vllm_omni.model_executor.models.kimia_audio.kimia_audio_code2wav import (
    KimiaAudioCode2WavForConditionalGeneration,
)

__all__ = [
    "KimiaAudioFusedForConditionalGeneration",
    "KimiaAudioCode2WavForConditionalGeneration",
    "MODEL_STAGE",
    "TTS_MODEL_TYPE",
]

MODEL_STAGE = "fused_thinker_talker"
"""Model stage identifier for the Kimi-Audio fused thinker-talker AR stage."""

TTS_MODEL_TYPE = "kimia_audio"
"""TTS model type identifier for Kimi-Audio serving."""
