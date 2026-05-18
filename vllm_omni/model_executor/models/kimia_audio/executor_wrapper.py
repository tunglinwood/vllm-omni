# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom executor wrapper for Kimi-Audio dual-stream AR generation.

This module provides a wrapper around vLLM's standard executor that patches
the worker's `sample_tokens` method to append audio codes to the sampled
token output. This enables dual-stream (text + audio) token feedback in
vLLM's autoregressive generation loop.

The injection mechanism works as follows:
1. The model's `forward()` stores `audio_codes` on the model instance
2. The patched `sample_tokens` retrieves audio codes and appends them to
   each request's sampled token IDs
3. vLLM's scheduler feeds back both text and audio tokens in the next step
4. The model unpacks mixed tokens and combines embeddings
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _apply_kimia_audio_patch(worker: Any) -> None:
    """Apply the Kimi-Audio audio code injection patch to a worker.

    Called via collective_rpc on each worker. Patches model_runner.sample_tokens
    to append audio codes to sampled output.
    """
    model_runner = worker.model_runner
    if model_runner is None:
        logger.warning("Cannot patch sample_tokens: model_runner is None")
        return

    model = getattr(model_runner, "model", None)
    if model is None:
        logger.warning("Cannot patch sample_tokens: model is None")
        return

    if not hasattr(model, "_audio_codes"):
        logger.info("Not a Kimi-Audio model, skipping audio injection patch.")
        return

    config = getattr(model, "config", None)
    if config is None:
        logger.warning("Cannot patch sample_tokens: model config is None")
        return

    token_offset = int(getattr(config, "kimia_token_offset", 152064))

    original_sample_tokens = model_runner.sample_tokens

    def patched_sample_tokens(grammar_output: Any = None) -> Any:
        """Wrapped sample_tokens that appends audio codes to output."""
        output = original_sample_tokens(grammar_output)

        if output is None:
            return output

        audio_codes = getattr(model, "_audio_codes", None)
        if audio_codes is None:
            return output

        try:
            if hasattr(audio_codes, "reshape"):
                codes_flat = audio_codes.reshape(-1).cpu().tolist()
            elif isinstance(audio_codes, list):
                codes_flat = [c for code_list in audio_codes for c in code_list]
            else:
                codes_flat = list(audio_codes)

            if not codes_flat:
                return output

            sampled_token_ids = getattr(output, "sampled_token_ids", None)
            if sampled_token_ids is None:
                return output

            if len(sampled_token_ids) == 1 and len(codes_flat) >= 1:
                sampled_token_ids[0].append(codes_flat[0] + token_offset)
            elif len(sampled_token_ids) > 1:
                for i in range(min(len(sampled_token_ids), len(codes_flat))):
                    sampled_token_ids[i].append(codes_flat[i] + token_offset)

        except Exception:
            logger.warning(
                "Failed to inject audio codes into sampled output.",
                exc_info=True,
            )

        return output

    model_runner.sample_tokens = patched_sample_tokens
    logger.info(
        "Patched sample_tokens for Kimi-Audio (token_offset=%d).",
        token_offset,
    )


class KimiaAudioExecutorWrapper:
    """Wraps a standard vLLM executor to inject audio codes into sampled output.

    This wrapper intercepts the `sample_tokens` call and appends generated
    audio codes to the sampled token IDs, enabling dual-stream AR generation.

    The wrapper works by:
    1. Creating the underlying executor
    2. Using collective_rpc to install the audio patch on each worker
    3. The worker patches model_runner.sample_tokens to append audio codes

    Usage:
        ```python
        vllm_config, executor_class = build_vllm_config(...)
        if is_kimia_audio:
            executor_class = KimiaAudioExecutorWrapper(executor_class)
        ```
    """

    def __init__(self, executor_class: type) -> None:
        self._executor_class = executor_class

    def __call__(self, vllm_config: VllmConfig) -> Any:
        """Create the wrapped executor and set up audio code injection."""
        executor = self._executor_class(vllm_config)

        # After executor init, patch workers via collective_rpc.
        # The callable is cloudpickled and executed on each worker.
        executor.collective_rpc(_apply_kimia_audio_patch)
        logger.info("Kimi-Audio audio code injection patch installed.")

        return executor
