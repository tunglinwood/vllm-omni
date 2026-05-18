# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for Kimia Audio: fused -> code2wav.

Extracts audio codes from Stage 0 (fused_thinker_talker) output and
packages them as input for Stage 1 (code2wav detokenizer).

Kimi-Audio produces single-token audio codes per step (not multi-channel
RVQ like MiMoAudio), so no column-major flattening is needed at this level.
The code2wav detokenizer handles the ODE-based waveform generation internally.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
    _validate_stage_inputs,
)

logger = init_logger(__name__)


def fused2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: collect fused stage outputs, then pass audio codes to code2wav.

    Args:
        stage_list: List of stage outputs from upstream stages.
        engine_input_source: Indices of upstream stages to consume.
        prompt: Original prompt (unused currently).
        requires_multimodal_data: Whether multimodal data is required (unused).

    Returns:
        List of OmniTokensPrompt containing audio codes for the code2wav stage.
    """
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []

    for talker_output in talker_outputs:
        if not talker_output.finished:
            # Non-async decode runs once after talker has accumulated
            # the final code sequence.
            continue

        output = talker_output.outputs[0]
        audio_codes = output.multimodal_output.get("audio_codes")

        logger.info(
            "DIAG fused2code2wav: talker_output.finished=%s, audio_codes type=%s, "
            "audio_codes shape=%s",
            talker_output.finished,
            type(audio_codes).__name__,
            list(audio_codes.shape) if isinstance(audio_codes, torch.Tensor) else None,
        )

        if audio_codes is None or (isinstance(audio_codes, torch.Tensor) and audio_codes.numel() == 0):
            logger.warning("No audio_codes found in fused stage output, skipping.")
            continue

        # Convert audio codes to flat list of token IDs for code2wav stage
        # audio_codes may be a bare tensor (current) or list-wrapped (legacy):
        #   {"audio_codes": tensor(...)}  or  {"audio_codes": [tensor(...)]}
        if isinstance(audio_codes, list):
            # Unwrap: list-wrapped payload from fused stage
            if len(audio_codes) > 0 and isinstance(audio_codes[0], torch.Tensor):
                audio_codes = audio_codes[0]
                codec_codes = audio_codes.to(torch.long).reshape(-1).cpu().tolist()
            else:
                codec_codes = audio_codes
        elif isinstance(audio_codes, torch.Tensor):
            audio_codes = audio_codes.to(torch.long)
            # Flatten to 1D: code2wav expects flat token IDs
            codec_codes = audio_codes.reshape(-1).cpu().tolist()
        else:
            logger.warning(
                "Unexpected audio_codes type: %s, skipping.",
                type(audio_codes),
            )
            continue

        if not codec_codes:
            continue

        # Guard against oversized code sequences that would crash code2wav.
        # This can happen if audio codes accumulate incorrectly across steps.
        MAX_AUDIO_CODES = 8192  # max_model_len of code2wav stage
        if len(codec_codes) > MAX_AUDIO_CODES:
            logger.warning(
                "Audio codes (%d) exceed max_model_len (%d), truncating to last %d.",
                len(codec_codes),
                MAX_AUDIO_CODES,
                MAX_AUDIO_CODES,
            )
            codec_codes = codec_codes[-MAX_AUDIO_CODES:]

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )

    return code2wav_inputs
