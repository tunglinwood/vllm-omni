# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference Kimi-Audio detokenizer wrapper for vLLM-Omni.

This module wraps the reference Kimi-Audio StreamingSemanticFMWrapper
detokenizer for use in the vLLM-Omni code2wav stage. It provides identical
generation quality to the reference implementation, including the
look_ahead_tokens mechanism that vLLM's native rewrite lacks.

The wrapper generates mel spectrograms via the reference DiT + ODE solver,
then returns them for vocoding by the vLLM vocoder (HiFi-GAN).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Add reference code to path
_REFERENCE_PATH = "/root/workspace/Kimi-Audio"
if _REFERENCE_PATH not in sys.path:
    sys.path.insert(0, _REFERENCE_PATH)

from kimia_infer.models.detokenizer.semantic_fm_prefix_streaming import (  # noqa: E402
    StreamingSemanticFMWrapper,
)


class ReferenceDetokenizerWrapper:
    """Thin wrapper around StreamingSemanticFMWrapper for vLLM integration.

    Generates mel spectrograms via the reference DiT + ODE solver with
    look_ahead_tokens support. Mel is returned for vocoding by the vLLM
    vocoder (HiFi-GAN).
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        max_kv_cache_tokens: int = 900,
        ode_steps: int = 30,
        chunk_size: int = 30,
        look_ahead_tokens: int = 12,
    ):
        self.device = device
        self.ode_steps = ode_steps
        self.chunk_size = chunk_size
        self.look_ahead_tokens = look_ahead_tokens

        config_path = os.path.join(model_path, "audio_detokenizer", "config.yaml")
        ckpt_path = os.path.join(model_path, "audio_detokenizer", "model.pt")

        logger.info(
            "Loading reference detokenizer from %s (config=%s)",
            model_path,
            config_path,
        )

        self.detokenizer = StreamingSemanticFMWrapper.from_pretrained(
            model_config=config_path,
            ckpt_path=ckpt_path,
            device=device,
            max_prompt_chunk=2,
            max_kv_cache_tokens=max_kv_cache_tokens,
            use_cfg=False,
            use_cfg_rescale=False,
            cfg_init=1.0,
            cfg_scale=4.0,
            cfg_schedule="linear",
        )
        logger.info("Reference detokenizer loaded successfully")

    @torch.no_grad()
    def generate_streaming(
        self,
        codes: torch.Tensor,
        vocoder: Any = None,
        n_steps: int | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Generate waveform from audio codes using reference detokenizer.

        Generates mel via reference DiT, then vocodes with the provided vocoder.

        Args:
            codes: audio codes [B, T] — only batch_size=1 supported
            vocoder: vLLM HiFi-GAN vocoder module
            n_steps: ODE steps per chunk (default: self.ode_steps)
            chunk_size: chunk size in tokens (default: self.chunk_size)

        Returns:
            waveform [B, T_audio] if vocoder provided, else mel [B, 80, T]
        """
        mel = self.generate_mel(codes, n_steps=n_steps, chunk_size=chunk_size)

        if vocoder is not None:
            waveform = vocoder(mel)
            return waveform

        return mel

    @torch.no_grad()
    def generate_mel(
        self,
        codes: torch.Tensor,
        n_steps: int | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Generate mel spectrogram from audio codes using reference detokenizer.

        Args:
            codes: audio codes [B, T]
            n_steps: ODE steps per chunk
            chunk_size: chunk size in tokens

        Returns:
            mel spectrogram [B, 80, T]
        """
        if n_steps is None:
            n_steps = self.ode_steps
        if chunk_size is None:
            chunk_size = self.chunk_size

        assert codes.dim() == 2 and codes.shape[0] == 1, "Only batch_size=1 supported"
        codes = codes.squeeze(0)  # [T]

        self.detokenizer.clear_all_states()
        self.detokenizer.start_position_id = 0

        mel = self.detokenizer.infer_mel(
            semantic_tokens=codes,
            ode_steps=n_steps,
            chunk_size=chunk_size,
        )

        return mel.unsqueeze(0)  # [B, 80, T]
