# Copyright 2025 vLLM-Omni Team
"""Stage input processor for Kimi Audio async chunk streaming."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)

logger = init_logger(__name__)

# Chunk size: 50 tokens = 1 second at 50 tokens/sec
CODEC_CHUNK_FRAMES = 50
CODEC_LEFT_CONTEXT_FRAMES = 0  # No overlap for flow-matching


def llm2detokenizer_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict,
    request: Any,
    is_finished: bool,
) -> OmniPayloadStruct | None:
    """Convert LLM audio tokens to detokenizer input (async chunk streaming).

    This function is called by the SharedMemoryConnector to transfer audio tokens
    from Stage 0 (LLM) to Stage 1 (detokenizer).

    Args:
        transfer_manager: Manages state across chunks (accumulated tokens per request)
        multimodal_output: Dict with "audio_tokens" key from Stage 0 (sampled token IDs)
        request: Current request object
        is_finished: Whether this is the final chunk

    Returns:
        OmniPayloadStruct with audio tokens, or None if not ready to send
    """
    # Extract audio tokens from multimodal_output (already sampled token IDs)
    audio_token_ids = multimodal_output.get("audio_tokens")  # [B, 1] or [B]
    if audio_token_ids is None:
        logger.warning("No audio_tokens in multimodal_output, keys: %s", list(multimodal_output.keys()))
        return None

    # Debug: Print audio token stats
    logger.info(f"[KimiAudio Stage Transfer] Received audio_tokens: shape={audio_token_ids.shape}, "
                f"min={audio_token_ids.min()}, max={audio_token_ids.max()}, mean={audio_token_ids.float().mean():.2f}")

    # Handle both 1D [B] and 2D [B, 1] shapes
    if audio_token_ids.dim() == 1:
        # [B] -> [B, 1]
        audio_token_ids = audio_token_ids.unsqueeze(1)

    # CRITICAL: Filter audio tokens by kimia_token_offset (152064)
    # Kimi Audio uses a unified vocabulary where:
    # - Text tokens: [0, 152063]
    # - Audio tokens: [152064, 168447] (16384 audio tokens)
    # The audio detokenizer expects tokens in range [0, 16383]
    kimia_token_offset = 152064  # From config.kimia_token_offset

    # Filter to keep only audio tokens (>= offset)
    audio_mask = audio_token_ids >= kimia_token_offset
    num_audio_tokens = audio_mask.sum().item()
    logger.info(f"[KimiAudio Stage Transfer] Audio token filtering: {num_audio_tokens} tokens >= {kimia_token_offset}")

    audio_token_ids = audio_token_ids[audio_mask]  # Flatten to 1D

    # Subtract offset to get token IDs in range [0, 16383]
    if audio_token_ids.numel() > 0:
        audio_token_ids = audio_token_ids - kimia_token_offset
        # Reshape to [1, L] for compatibility
        audio_token_ids = audio_token_ids.unsqueeze(0)
        logger.info(f"[KimiAudio Stage Transfer] After filtering: {audio_token_ids.shape}, min={audio_token_ids.min()}, max={audio_token_ids.max()}")
    else:
        # No valid audio tokens - return empty tensor
        audio_token_ids = torch.zeros((1, 0), dtype=audio_token_ids.dtype, device=audio_token_ids.device)
        logger.warning(f"[KimiAudio Stage Transfer] No audio tokens found! All tokens were in text range.")

    # Important: Only keep the last token's IDs (for generation, not prefill)
    # During prefill, B can be > 1 (one for each prompt token)
    # During generation, B = 1 (one for the generated token)
    # We only want the last token (the one we just generated)
    if audio_token_ids.shape[0] > 1:
        # Take only the last token (last row)
        audio_token_ids = audio_token_ids[-1:, :]  # [1, L]

    # Accumulate tokens per request
    request_id = request.request_id
    if not hasattr(transfer_manager, "audio_tokens"):
        transfer_manager.audio_tokens = {}

    if request_id not in transfer_manager.audio_tokens:
        transfer_manager.audio_tokens[request_id] = []

    transfer_manager.audio_tokens[request_id].append(audio_token_ids)

    # Debug: log shapes
    logger.info(f"[KimiAudio Stage Transfer] Request {request_id}: "
                f"audio_token_ids shape={audio_token_ids.shape}, "
                f"num_accumulated={len(transfer_manager.audio_tokens[request_id])}")

    # Chunk gating: send when we have enough tokens or finished
    accumulated = torch.cat(transfer_manager.audio_tokens[request_id], dim=1)
    chunk_size = CODEC_CHUNK_FRAMES  # 50

    if accumulated.shape[1] >= chunk_size or is_finished:
        # Take last chunk_size tokens
        if accumulated.shape[1] >= chunk_size:
            chunk = accumulated[:, -chunk_size:]
        else:
            chunk = accumulated

        # Build payload
        payload = OmniPayloadStruct(
            codes=CodesStruct(audio=chunk.reshape(-1)),
            meta=MetaStruct(
                finished=torch.tensor(is_finished, dtype=torch.bool),
                codec_chunk_frames=chunk_size,
                codec_left_context_frames=CODEC_LEFT_CONTEXT_FRAMES,
            ),
        )

        # Clear accumulated (keep last chunk for overlap if needed)
        if is_finished:
            del transfer_manager.audio_tokens[request_id]
        else:
            transfer_manager.audio_tokens[request_id] = [chunk]

        return payload

    return None  # Not ready to send yet


def llm2detokenizer(
    transfer_manager: Any,
    multimodal_output: dict,
    request: Any,
    is_finished: bool,
) -> OmniPayloadStruct | None:
    """Convert LLM audio logits to detokenizer input (sync mode).

    Args:
        transfer_manager: Manages state across chunks
        multimodal_output: Dict with "audio_logits" key from Stage 0
        request: Current request object
        is_finished: Whether this is the final chunk

    Returns:
        OmniPayloadStruct with audio tokens, or None
    """
    # For sync mode, just process all tokens at once
    return llm2detokenizer_async_chunk(
        transfer_manager, multimodal_output, request, is_finished
    )


def llm2detokenizer_token_only(
    multimodal_output: dict,
) -> torch.Tensor | None:
    """Extract audio token IDs only (for sync process_input).

    Args:
        multimodal_output: Dict with "audio_tokens" key from Stage 0

    Returns:
        Audio token IDs tensor, or None
    """
    audio_token_ids = multimodal_output.get("audio_tokens")
    if audio_token_ids is None:
        return None

    # Handle both 1D [B] and 2D [B, 1] shapes
    if audio_token_ids.dim() == 1:
        audio_token_ids = audio_token_ids.unsqueeze(1)

    # CRITICAL: Filter audio tokens by kimia_token_offset (152064)
    # Kimi Audio uses a unified vocabulary where:
    # - Text tokens: [0, 152063]
    # - Audio tokens: [152064, 168447] (16384 audio tokens)
    # The audio detokenizer expects tokens in range [0, 16383]
    kimia_token_offset = 152064  # From config.kimia_token_offset

    # Filter to keep only audio tokens (>= offset)
    audio_mask = audio_token_ids >= kimia_token_offset
    audio_token_ids = audio_token_ids[audio_mask]  # Flatten to 1D

    # Subtract offset to get token IDs in range [0, 16383]
    if audio_token_ids.numel() > 0:
        audio_token_ids = audio_token_ids - kimia_token_offset
        # Reshape to [1, L] for compatibility
        audio_token_ids = audio_token_ids.unsqueeze(0)
    else:
        # No valid audio tokens - return empty tensor
        audio_token_ids = torch.zeros((1, 0), dtype=audio_token_ids.dtype, device=audio_token_ids.device)

    # Important: Only keep the last token's IDs (for generation, not prefill)
    if audio_token_ids.shape[0] > 1:
        audio_token_ids = audio_token_ids[-1:, :]  # [1, L]

    return audio_token_ids


def llm2detokenizer_full_payload(
    transfer_manager: Any,
    multimodal_output: dict,
    request: Any,
    is_finished: bool,
) -> OmniPayloadStruct | None:
    """Convert LLM audio tokens to full detokenizer payload (for next_stage_input).

    Args:
        transfer_manager: Manages state across chunks
        multimodal_output: Dict with "audio_tokens" key from Stage 0
        request: Current request object
        is_finished: Whether this is the final chunk

    Returns:
        OmniPayloadStruct with audio tokens, or None
    """
    # For now, same as async_chunk
    return llm2detokenizer_async_chunk(
        transfer_manager, multimodal_output, request, is_finished
    )
