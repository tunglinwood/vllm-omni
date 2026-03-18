#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Simple test for Kimi-Audio TTS with AsyncOmni (non-streaming)."""

import asyncio
import os

import numpy as np
import wave

from vllm_omni import AsyncOmni
from vllm_omni.inputs.data import OmniTokensPrompt


def save_wav(waveform: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Save waveform to WAV file."""
    if waveform.ndim == 2:
        waveform = waveform.squeeze()
    waveform = np.clip(waveform, -1.0, 1.0)
    waveform_int16 = (waveform * 32767).astype(np.int16)
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform_int16.tobytes())


async def main():
    model_path = "/data1/moonshotai/Kimi-Audio-7B-Instruct"
    stage_configs_path = "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml"
    output_dir = "/root/learning/vllm-omni/examples/offline_inference/kimi_audio_tts/output_test"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing AsyncOmni...")
    omni = AsyncOmni(
        model=model_path,
        stage_configs_path=stage_configs_path,
        enforce_eager=True,
    )
    print("Engine ready!")
    
    text = "你好"
    print(f"Generating TTS for: {text}")
    
    request = OmniTokensPrompt(
        prompt_token_ids=[0] * 128,
        multi_modal_data=None,
        additional_information={
            "text": [text],
            "task_type": ["tts"],
        },
    )
    
    audio_chunks = []
    sample_rate = 24000
    
    async for stage_output in omni.generate(request, request_id="test"):
        mm = stage_output.request_output.outputs[0].multimodal_output
        if stage_output.finished:
            print("Generation finished!")
            audio_data = mm.get("model_outputs")
            if audio_data:
                chunk = audio_data[0].cpu().numpy()
                audio_chunks.append(chunk)
                print(f"Final chunk: {len(chunk)} samples")
        else:
            audio_data = mm.get("model_outputs")
            if audio_data:
                chunk = audio_data[0].cpu().numpy()
                audio_chunks.append(chunk)
                print(f"Chunk received: {len(chunk)} samples")
    
    if audio_chunks:
        waveform = np.concatenate(audio_chunks)
        output_path = os.path.join(output_dir, "test_output.wav")
        save_wav(waveform, sample_rate, output_path)
        print(f"Saved: {output_path} ({len(waveform)/sample_rate:.2f}s)")
    else:
        print("No audio generated!")
    
    omni.close()


if __name__ == "__main__":
    asyncio.run(main())
