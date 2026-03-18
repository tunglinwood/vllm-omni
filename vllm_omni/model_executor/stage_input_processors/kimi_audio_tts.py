# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processor for Kimi-Audio TTS: Talker → Code2Wav transition."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    max_ic_for_chunk_size,
)

logger = init_logger(__name__)


def talker2detokenizer(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: collect all talker codes, then pass to detokenizer at once.
    
    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for talker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required
        
    Returns:
        List of OmniTokensPrompt for detokenizer stage
    """
    from vllm_omni.inputs.data import OmniTokensPrompt
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _validate_stage_inputs
    
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    detokenizer_inputs: list[OmniTokensPrompt] = []
    
    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        
        # Extract audio codes from talker output
        # audio_codes shape: [num_frames,] or [num_frames, num_groups]
        # OmniOutput uses multimodal_outputs (plural)
        multimodal_outputs = getattr(output, 'multimodal_outputs', None)
        if not isinstance(multimodal_outputs, dict):
            multimodal_outputs = {}
        
        audio_codes = multimodal_outputs.get("audio_codes")
        
        if audio_codes is None:
            logger.warning(f"No audio_codes found in talker output. multimodal_outputs type={type(multimodal_outputs)}, keys={multimodal_outputs.keys() if isinstance(multimodal_outputs, dict) else 'N/A'}")
            continue
        
        audio_codes = audio_codes.to(torch.long)
        
        # Filter zero-padded frames (EOS/invalid steps)
        if audio_codes.ndim == 2:
            valid_mask = audio_codes.any(dim=1)
            audio_codes = audio_codes[valid_mask]
        
        # Get ref_code if present (for ICL mode)
        ref_code = multimodal_outputs.get("ref_code") if isinstance(multimodal_outputs, dict) else None
        ref_code_len = 0
        
        if isinstance(ref_code, list):
            ref_code = ref_code[0] if ref_code else None
        
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
            ref_code = ref_code.to(torch.long).cpu().contiguous()
            ref_code_len = int(ref_code.shape[0])
            audio_codes = torch.cat([ref_code.to(audio_codes.device), audio_codes], dim=0)
        
        # Flatten codes for detokenizer input
        # Detokenizer expects flat sequence of audio token IDs
        codec_codes = audio_codes.cpu().reshape(-1).tolist()
        
        additional_information = {"left_context_size": [ref_code_len]} if ref_code_len > 0 else None
        
        from vllm_omni.inputs.data import OmniTokensPrompt
        detokenizer_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information,
            )
        )
    
    return detokenizer_inputs


def _extract_last_frame(pooling_output: dict[str, Any]) -> torch.Tensor | None:
    """Extract last frame of audio codes for streaming."""
    audio_codes = pooling_output.get("audio_codes")
    
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    
    if audio_codes.ndim == 1:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        return frame.to(torch.long).reshape(-1)
    
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        return frame.to(torch.long).reshape(-1)
    
    raise ValueError(f"Invalid audio_codes shape for Kimi-Audio async_chunk: {tuple(audio_codes.shape)}")


def talker2detokenizer_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async chunk version: convert talker pooling_output to detokenizer payload.
    
    Simplified version that returns all accumulated audio codes when finished.
    
    Args:
        transfer_manager: Transfer manager with connector and request state
        pooling_output: Output from talker stage
        request: Current request object
        is_finished: Whether the request is finished
        
    Returns:
        Payload dict with code_predictor_codes or None if waiting for more chunks
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    
    logger.info(f"[AsyncChunk] request_id={request_id}, pooling_output type={type(pooling_output)}, finished={finished}")
    if isinstance(pooling_output, dict):
        logger.info(f"[AsyncChunk] pooling_output keys={pooling_output.keys()}")
    
    # Initialize request payload storage
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    
    # Extract audio codes from pooling output
    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        
        if frame is not None:
            codec_codes = frame.cpu().tolist()
            transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
            logger.info(f"[AsyncChunk] Appended {len(codec_codes)} codec codes")
        
        # Store ref_code for first chunk (ICL mode)
        ref_code = pooling_output.get("ref_code")
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
            if request_payload.get(request_id) is None:
                request_payload[request_id] = ref_code.to(torch.long).cpu().contiguous()
    
    elif not finished:
        # Some steps may not produce pooling_output.
        pass
    
    # For generation scheduler: flush accumulated codes after each chunk
    # (don't wait for finished signal which may not come)
    length = len(transfer_manager.code_prompt_token_ids.get(request_id, []))
    logger.info(f"[AsyncChunk] Processing chunk, total chunks={length}, finished={finished}")
    
    if length <= 0:
        # No data yet, wait for more chunks
        if not finished:
            return None
        return {
            "code_predictor_codes": [],
            "finished": torch.tensor(True, dtype=torch.bool),
        }
    
    # Flatten all chunks
    all_codes = []
    for chunk in transfer_manager.code_prompt_token_ids[request_id]:
        all_codes.extend(chunk)
    
    # Prepend ref_code if present
    ref_code = request_payload.pop(request_id, None)
    if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
        ref_frames = ref_code.tolist()
        all_codes = ref_frames + all_codes
        logger.info(f"[AsyncChunk] Prepended {len(ref_frames)} ref_code frames")
    
    logger.info(f"[AsyncChunk] Returning {len(all_codes)} total code_predictor_codes")
    
    # Clear accumulated codes after sending (prevent duplicates on next call)
    transfer_manager.code_prompt_token_ids[request_id] = []
    
    return {
        "code_predictor_codes": all_codes,
        "left_context_size": 0,
        "finished": torch.tensor(finished, dtype=torch.bool),
    }
