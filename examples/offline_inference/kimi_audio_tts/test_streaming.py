#!/usr/bin/env python3
"""Test Kimi-Audio TTS with async_chunk streaming mode."""

import asyncio
import os
import time

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


async def test_streaming():
    model_path = "/data1/moonshotai/Kimi-Audio-7B-Instruct"
    # Use ASYNC-CHUNK config for streaming
    stage_configs_path = "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml"
    output_dir = "/root/learning/vllm-omni/examples/offline_inference/kimi_audio_tts/output_streaming"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Streaming Test (kimi_audio_tts.yaml - async_chunk: true)")
    print("=" * 80)
    
    t0 = time.time()
    print(f"\n[T+{time.time()-t0:.1f}s] Initializing AsyncOmni engine...")
    
    omni = AsyncOmni(
        model=model_path,
        stage_configs_path=stage_configs_path,
        enforce_eager=True,
    )
    
    print(f"[T+{time.time()-t0:.1f}s] ✓ AsyncOmni engine initialized")
    print(f"  - Stages: {len(omni.stage_list)}")
    print(f"  - Async chunk: {omni.async_chunk}")
    
    # Create request
    print(f"\n[T+{time.time()-t0:.1f}s] Creating request...")
    text = "你好"
    request = OmniTokensPrompt(
        prompt_token_ids=[0] * 128,
        multi_modal_data=None,
        additional_information={
            "text": [text],
            "task_type": ["tts"],
        },
    )
    print(f"✓ Request created for: '{text}'")
    
    # Generate with streaming
    print(f"\n[T+{time.time()-t0:.1f}s] Starting streaming generation...")
    
    audio_chunks = []
    sample_rate = 24000
    chunk_count = 0
    t_gen_start = time.time()
    
    try:
        async for stage_output in omni.generate(request, request_id="stream-test"):
            chunk_count += 1
            
            if stage_output.finished:
                t_gen_end = time.time()
                print(f"\n✓ Generation finished! (chunk {chunk_count}, {(t_gen_end - t_gen_start):.1f}s)")
                
                mm = stage_output.request_output.outputs[0].multimodal_output
                audio_data = mm.get("model_outputs")
                
                if audio_data:
                    chunk = audio_data[0].cpu().numpy()
                    audio_chunks.append(chunk)
                    print(f"  Final chunk: {len(chunk)} samples")
            else:
                mm = stage_output.request_output.outputs[0].multimodal_output
                audio_data = mm.get("model_outputs")
                
                if audio_data:
                    chunk = audio_data[0].cpu().numpy()
                    audio_chunks.append(chunk)
                    print(f"  Chunk {chunk_count}: {len(chunk)} samples")
        
        # Save audio
        if audio_chunks:
            waveform = np.concatenate(audio_chunks)
            duration = len(waveform) / sample_rate
            print(f"\n✓ Generated {duration:.2f}s of audio ({len(waveform)} samples)")
            
            output_path = os.path.join(output_dir, "test_streaming.wav")
            save_wav(waveform, sample_rate, output_path)
            print(f"✓ Saved: {output_path}")
            print(f"\n🎉 SUCCESS! Streaming TTS working!")
        else:
            print("\n✗ No audio generated")
            
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n[T+{time.time()-t0:.1f}s] Cleaning up...")
        omni.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(test_streaming())
