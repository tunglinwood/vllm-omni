# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whisper feature extractor for Kimi-Audio speech-to-speech.

Encapsulates Whisper encoder + VQAdaptor for runtime feature extraction
from user audio input. The model was trained with Whisper features
(use_whisper_feature: True), so extracting them at request time is
essential for accurate audio generation.

Pipeline: audio waveform -> Whisper encoder -> 4x downsample -> VQAdaptor -> [1, seq, 3584]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

if TYPE_CHECKING:
    from transformers import WhisperFeatureExtractor, WhisperModel

logger = init_logger(__name__)


class VQAdaptor(nn.Module):
    """Projects Whisper features (5120-dim) to Kimi-Audio hidden space (3584-dim).

    Architecture: Linear[5120->3584] -> SiLU -> Linear[3584->3584] -> LayerNorm[3584]
    """

    def __init__(self, input_dim: int = 5120, hidden_dim: int = 3584):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class WhisperFeatureExtractor:
    """Lazy-loaded Whisper encoder + VQAdaptor for runtime audio feature extraction.

    Usage:
        extractor = WhisperFeatureExtractor(whisper_path, vq_shard_path)
        whisper_emb = extractor.extract(audio_waveform, sample_rate=16000)
        # whisper_emb: [1, seq_len, 3584]
    """

    def __init__(
        self,
        whisper_model_path: str,
        vq_adaptor_shard_path: str,
        device: str = "auto",
    ):
        self._whisper_path = whisper_model_path
        self._vq_shard_path = vq_adaptor_shard_path
        self._device = device
        self._whisper_model: WhisperModel | None = None
        self._feature_extractor: WhisperFeatureExtractor | None = None
        self._vq_adaptor: VQAdaptor | None = None

    @property
    def whisper_model(self) -> WhisperModel:
        """Lazy-load Whisper model on first access."""
        if self._whisper_model is None:
            self._init_whisper()
        return self._whisper_model

    @property
    def vq_adaptor(self) -> VQAdaptor:
        """Lazy-load VQAdaptor on first access."""
        if self._vq_adaptor is None:
            self._init_whisper()
        return self._vq_adaptor

    def _init_whisper(self) -> None:
        """Initialize Whisper encoder + VQAdaptor (lazy, called once)."""
        from safetensors.torch import load_file
        from transformers import WhisperFeatureExtractor as HFWhisperFeatureExtractor
        from transformers import WhisperModel

        logger.info("Loading Whisper encoder from %s...", self._whisper_path)
        whisper_model = WhisperModel.from_pretrained(self._whisper_path)
        whisper_model.eval()

        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device

        whisper_model.encoder.to(torch.bfloat16)
        if device == "cuda":
            whisper_model.encoder = whisper_model.encoder.cuda()

        self._whisper_model = whisper_model
        self._whisper_device = device
        logger.info(
            "Loaded Whisper encoder on %s with %d params",
            device,
            sum(p.numel() for p in whisper_model.encoder.parameters()),
        )

        # Load VQAdaptor
        logger.info("Loading VQAdaptor from %s...", self._vq_shard_path)
        ckpt = load_file(self._vq_shard_path)
        vq_adaptor = VQAdaptor(input_dim=5120, hidden_dim=3584)

        # VQAdaptor weights in checkpoint use nn.Sequential indexing:
        #   layers.0 = first Linear, layers.3 = second Linear, layers.4 = LayerNorm
        vq_adaptor.load_state_dict({
            "layers.0.weight": ckpt["model.vq_adaptor.layers.0.weight"],
            "layers.0.bias": ckpt["model.vq_adaptor.layers.0.bias"],
            "layers.2.weight": ckpt["model.vq_adaptor.layers.3.weight"],
            "layers.2.bias": ckpt["model.vq_adaptor.layers.3.bias"],
            "layers.3.weight": ckpt["model.vq_adaptor.layers.4.weight"],
            "layers.3.bias": ckpt["model.vq_adaptor.layers.4.bias"],
        })
        vq_adaptor.eval()
        vq_adaptor.to(torch.bfloat16)
        if device == "cuda":
            vq_adaptor = vq_adaptor.cuda()

        self._vq_adaptor = vq_adaptor
        logger.info("Loaded VQAdaptor weights")

        # Feature extractor (mel spectrogram)
        self._feature_extractor = HFWhisperFeatureExtractor.from_pretrained(
            self._whisper_path
        )

    def _preprocess_audio(
        self, audio: np.ndarray | torch.Tensor, sample_rate: int = 16000
    ) -> torch.Tensor:
        """Convert audio to Whisper input features [B, 128, seq]."""
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy().flatten()
        else:
            audio = audio.flatten()

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        inputs = self._feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        return inputs.input_features  # [1, 128, seq]

    def _downsample_4x(self, whisper_features: torch.Tensor) -> torch.Tensor:
        """Downsample Whisper output from 50Hz to 12.5Hz (4x).

        Concatenates every 4 consecutive frames into one:
        [B, 1200, 1280] -> [B, 300, 5120]
        """
        batch_size, seq_len, hidden_dim = whisper_features.shape
        trunc_len = (seq_len // 4) * 4
        if trunc_len != seq_len:
            whisper_truncated = whisper_features[:, :trunc_len, :]
        else:
            whisper_truncated = whisper_features

        # Reshape: group every 4 frames and concat their features
        return whisper_truncated.view(
            batch_size, trunc_len // 4, hidden_dim * 4
        )  # [B, seq//4, 5120]

    @torch.no_grad()
    def extract(
        self,
        audio: np.ndarray | torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Extract Whisper features from audio waveform.

        Args:
            audio: Audio waveform (numpy array or torch tensor).
            sample_rate: Sample rate of the audio.

        Returns:
            Whisper embedding tensor [1, seq_len, 3584] — projected through
            VQAdaptor, ready for fusion with text embeddings.
        """
        # Preprocess audio to mel spectrogram features
        # Ensure feature extractor is initialized (lazy init)
        if self._feature_extractor is None:
            self._init_whisper()
        audio_flat = audio.flatten()
        num_samples = len(audio_flat)
        input_features = self._preprocess_audio(audio, sample_rate)
        input_features = input_features.to(
            device=self._whisper_device, dtype=torch.bfloat16
        )

        # Run Whisper encoder
        encoder_outputs = self._whisper_model.encoder(input_features)
        whisper_features = encoder_outputs.last_hidden_state  # [1, seq, 1280]

        # Truncate encoder output to actual audio length, matching the reference
        # Kimi-Audio Whisper encoder behavior (whisper_Lv3/whisper.py:195-218).
        # The HF WhisperFeatureExtractor pads audio to 30s (480000 samples),
        # producing 1500 encoder frames. The reference truncates back based on
        # the actual audio length: token_len = (L - 1) // (160 * 8) + 1, then
        # keeps only token_len * 4 encoder frames before 4x downsampling.
        token_len = (num_samples - 1) // (160 * 8) + 1
        max_encoder_frames = token_len * 4
        actual_encoder_frames = whisper_features.shape[1]
        if max_encoder_frames < actual_encoder_frames:
            whisper_features = whisper_features[:, :max_encoder_frames, :]

        # 4x downsampling
        whisper_downsampled = self._downsample_4x(whisper_features)  # [1, seq//4, 5120]

        # VQAdaptor projection
        whisper_emb = self._vq_adaptor(whisper_downsampled)  # [1, seq//4, 3584]

        logger.info(
            "Extracted Whisper features: input %d samples -> %d encoder frames "
            "(truncated to %d) -> output %s",
            num_samples,
            actual_encoder_frames,
            min(max_encoder_frames, actual_encoder_frames),
            list(whisper_emb.shape),
        )

        return whisper_emb  # [1, seq_len, 3584]
