# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving helpers for Kimi-Audio S2S Whisper feature extraction.

This module provides functions to extract Whisper encoder features from
audio content in chat messages, used for speech-to-speech conditioning.
"""

from __future__ import annotations

import asyncio
import base64
import glob
import os
import sys
from io import BytesIO

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def has_audio_content(messages: list) -> bool:
    """Check if any message contains audio_url or video_url content."""
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("audio_url", "video_url"):
                    return True
    return False


async def extract_whisper_for_s2s(
    messages: list,
    model_dir: str,
    extract_audio_from_video_async: callable | None = None,
) -> tuple[torch.Tensor | None, dict | None]:
    """Extract Whisper features from audio_url items in chat messages.

    Used for Kimi Audio speech-to-speech: user input audio is processed
    through Whisper encoder + VQAdaptor to produce conditioning features
    that the model was trained with.

    Args:
        messages: Chat completion messages
        model_dir: Path to the model directory (contains whisper-large-v3/
            and model safetensors shards)
        extract_audio_from_video_async: Optional async function to extract
            audio from video URLs. If None, video URLs are skipped.

    Returns:
        Tuple of (whisper_emb [1, seq_len, 3584], raw_audio_info).
        raw_audio_info is {"audio_array": np.ndarray, "sample_rate": int}.
    """
    import soundfile as sf

    audio_arrays: list[np.ndarray] = []
    sample_rates: list[int] = []

    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if not isinstance(content, list):
            continue

        # Direct audio_url items
        for part in content:
            if isinstance(part, dict) and part.get("type") == "audio_url":
                audio_url = part.get("audio_url", {}).get("url", "")
                if audio_url.startswith("data:audio/"):
                    b64_data = audio_url.split(",", 1)[-1]
                    audio_bytes = base64.b64decode(b64_data)
                    audio_np, sr = sf.read(BytesIO(audio_bytes))
                    audio_arrays.append(audio_np)
                    sample_rates.append(sr)

        # Audio extracted from video URLs
        if extract_audio_from_video_async is not None:
            video_urls = [
                part.get("video_url", {}).get("url")
                for part in content
                if isinstance(part, dict) and part.get("type") == "video_url" and part.get("video_url", {}).get("url")
            ]
            if video_urls:
                audios = await asyncio.gather(*(extract_audio_from_video_async(u) for u in video_urls))
                for audio_np, sr in audios:
                    audio_arrays.append(audio_np)
                    sample_rates.append(int(sr))

    if not audio_arrays:
        return None, None

    # Concatenate all audio segments
    if len(audio_arrays) > 1:
        combined_audio = np.concatenate(audio_arrays)
        combined_sr = sample_rates[0]
    else:
        combined_audio = audio_arrays[0]
        combined_sr = sample_rates[0]

    # Resample to 16kHz for Whisper if needed (e.g., S2S output at 24kHz)
    WHISPER_SR = 16000
    if combined_sr != WHISPER_SR:
        import librosa
        logger.info(
            "Resampling input audio from %dHz to %dHz for Whisper",
            combined_sr, WHISPER_SR,
        )
        combined_audio = librosa.resample(
            combined_audio, orig_sr=combined_sr, target_sr=WHISPER_SR
        )
        combined_sr = WHISPER_SR

    whisper_path = os.path.join(model_dir, "whisper-large-v3")

    # Try reference Kimi-Audio WhisperEncoder first
    ref_encoder = None
    if os.path.isdir(whisper_path):
        try:
            # Look for reference implementation relative to model dir
            # or via KIMIA_AUDIO_REF_PATH env var
            ref_path = os.environ.get("KIMIA_AUDIO_REF_PATH", "")
            if ref_path and ref_path not in sys.path:
                sys.path.insert(0, ref_path)

            from kimia_infer.models.tokenizer.whisper_Lv3.whisper import (
                WhisperEncoder,
            )

            ref_encoder = WhisperEncoder(whisper_path, mel_batch_size=20)
            ref_encoder = ref_encoder.cuda()
            ref_encoder = ref_encoder.bfloat16()
            ref_encoder.eval()
            logger.info("Reference WhisperEncoder loaded for Whisper extraction")
        except Exception as e:
            logger.warning(
                "Failed to load reference WhisperEncoder, falling back to vLLM extractor: %s", e,
            )
            ref_encoder = None

    if ref_encoder is not None:
        audio_tensor = torch.from_numpy(combined_audio).float().unsqueeze(0).cuda()
        with torch.no_grad():
            encoder_output = ref_encoder(audio_tensor)

        # 4x downsample
        batch_size, seq_len, hidden_dim = encoder_output.shape
        trunc_len = (seq_len // 4) * 4
        if trunc_len != seq_len:
            encoder_output = encoder_output[:, :trunc_len, :]
        downsampled = encoder_output.view(batch_size, trunc_len // 4, hidden_dim * 4)

        # VQAdaptor projection
        from safetensors.torch import load_file

        from vllm_omni.model_executor.models.kimia_audio.whisper_feature_extractor import (
            VQAdaptor,
        )

        vq_shard_path = _find_vq_adaptor_shard(model_dir)
        if vq_shard_path is None:
            logger.warning("VQAdaptor shard not found for Whisper feature extraction")
            return None, None

        ckpt = load_file(vq_shard_path)
        vq_adaptor = VQAdaptor(input_dim=5120, hidden_dim=3584)
        vq_adaptor.load_state_dict({
            "layers.0.weight": ckpt["model.vq_adaptor.layers.0.weight"],
            "layers.0.bias": ckpt["model.vq_adaptor.layers.0.bias"],
            "layers.2.weight": ckpt["model.vq_adaptor.layers.3.weight"],
            "layers.2.bias": ckpt["model.vq_adaptor.layers.3.bias"],
            "layers.3.weight": ckpt["model.vq_adaptor.layers.4.weight"],
            "layers.3.bias": ckpt["model.vq_adaptor.layers.4.bias"],
        })
        vq_adaptor.eval()
        vq_adaptor.cuda()

        whisper_emb = vq_adaptor(downsampled.float())
        whisper_raw = downsampled.float()  # [1, seq//4, 5120] pre-VQAdaptor
    else:
        # Fallback: vLLM's HF-based Whisper extractor
        from vllm_omni.model_executor.models.kimia_audio.whisper_feature_extractor import (
            WhisperFeatureExtractor as RuntimeExtractor,
        )

        vq_shard_path = _find_vq_adaptor_shard(model_dir)
        if vq_shard_path is None:
            logger.warning("VQAdaptor shard not found for Whisper feature extraction")
            return None, None

        extractor = RuntimeExtractor(
            whisper_model_path=whisper_path,
            vq_adaptor_shard_path=vq_shard_path,
        )
        whisper_emb = extractor.extract(combined_audio, sample_rate=combined_sr)
        # Fallback path doesn't produce raw 5120-dim features separately
        whisper_raw = None

    logger.info(
        "Extracted Whisper features from input audio: %d samples -> shape %s",
        len(combined_audio),
        list(whisper_emb.shape),
    )
    raw_audio_info = {
        "audio_array": combined_audio.tolist(),
        "sample_rate": int(combined_sr),
    }
    if whisper_raw is not None:
        raw_audio_info["whisper_raw"] = whisper_raw.detach().to("cpu").contiguous()
    return whisper_emb, raw_audio_info


def _find_vq_adaptor_shard(model_dir: str) -> str | None:
    """Find the safetensors shard containing the VQAdaptor weights."""
    shards = glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors"))
    for shard in sorted(shards):
        from safetensors.torch import load_file

        ckpt = load_file(shard)
        if any("vq_adaptor" in k for k in ckpt.keys()):
            return shard
    return None
