# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for Kimia Audio: fused -> code2wav.

Extracts audio codes from Stage 0 (fused_thinker_talker) output and
packages them as input for Stage 1 (code2wav detokenizer).

Kimi-Audio produces single-token audio codes per step (not multi-channel
RVQ like MiMoAudio), so no column-major flattening is needed at this level.
The code2wav detokenizer handles the ODE-based waveform generation internally.

Audio codes are injected into multimodal_outputs via the model runner's
sample_tokens() method (from model._audio_codes). They arrive as a tensor
in multimodal_output["audio_codes"].
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

TOKEN_OFFSET = 152064  # kimia_token_offset from config


def fused2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any = None,
) -> list[Any]:
    """Non-async: collect fused stage outputs, then pass audio codes to code2wav.

    Args:
        source_outputs: List of output objects from upstream stages.
        prompt: Original prompt (unused currently).
        requires_multimodal_data: Whether multimodal data is required (unused).
        streaming_context: Streaming context (unused).

    Returns:
        List of OmniTokensPrompt containing audio codes for the code2wav stage.
    """
    code2wav_inputs: list[OmniTokensPrompt] = []

    for source_output in source_outputs:
        if not source_output.finished:
            continue

        output = source_output.outputs[0]

        # First try multimodal_output["audio_codes"] (injected by model runner)
        mm_output = output.multimodal_output if hasattr(output, 'multimodal_output') else None
        audio_codes_tensor = None

        if isinstance(mm_output, dict):
            audio_codes_tensor = mm_output.get("audio_codes")
        elif mm_output is not None:
            if hasattr(mm_output, "get"):
                audio_codes_tensor = mm_output.get("audio_codes")

        if audio_codes_tensor is None or (isinstance(audio_codes_tensor, torch.Tensor) and audio_codes_tensor.numel() == 0):
            # Fallback: try extracting from token_ids (tokens >= TOKEN_OFFSET)
            token_ids = list(output.token_ids) if output.token_ids else []
            audio_codes_from_tokens = [tid - TOKEN_OFFSET for tid in token_ids if tid >= TOKEN_OFFSET]

            if audio_codes_from_tokens:
                codec_codes = audio_codes_from_tokens
            else:
                logger.warning(
                    "No audio codes found. multimodal_output keys=%s, "
                    "max_token=%d, token_ids=%s",
                    list(mm_output.keys()) if isinstance(mm_output, dict) else "N/A",
                    max(token_ids) if token_ids else 0,
                    token_ids[:20] if token_ids else [],
                )
                continue
        elif isinstance(audio_codes_tensor, torch.Tensor):
            audio_codes_tensor = audio_codes_tensor.to(torch.long)
            codec_codes = audio_codes_tensor.reshape(-1).cpu().tolist()
        else:
            codec_codes = list(audio_codes_tensor)

        if not codec_codes:
            continue

        # Guard against oversized code sequences that would crash code2wav.
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
