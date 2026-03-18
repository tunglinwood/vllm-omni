#!/usr/bin/env python3
"""Test Kimi-Audio TTS with non-async-chunk mode."""

import os
import time

import numpy as np
import wave

from vllm_omni import Omni
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


def test_non_async():
    model_path = "/data1/moonshotai/Kimi-Audio-7B-Instruct"
    # Use NON-ASYNC-CHUNK config
    stage_configs_path = "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/kimi_audio_tts_no_async_chunk.yaml"
    output_dir = "/root/learning/vllm-omni/examples/offline_inference/kimi_audio_tts/output_non_async"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Non-Async-Chunk Test (kimi_audio_tts_no_async_chunk.yaml)")
    print("=" * 80)
    
    t0 = time.time()
    print(f"\n[T+{time.time()-t0:.1f}s] Initializing Omni engine...")
    
    omni = Omni(
        model=model_path,
        stage_configs_path=stage_configs_path,
        enforce_eager=True,
    )
    
    print(f"[T+{time.time()-t0:.1f}s] ✓ Omni engine initialized")
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
    
    # Generate
    print(f"\n[T+{time.time()-t0:.1f}s] Starting generation...")
    
    try:
        outputs = omni.generate([request])
        
        if outputs and len(outputs) > 0:
            output = outputs[0]
            if output.outputs and len(output.outputs) > 0:
                mm = output.outputs[0].multimodal_output
                audio_data = mm.get("model_outputs")
                sample_rates = mm.get("sr", [24000])
                
                if audio_data:
                    waveform = audio_data[0].cpu().numpy()
                    sample_rate = int(sample_rates[0].item()) if sample_rates else 24000
                    
                    print(f"✓ Generated {len(waveform)/sample_rate:.2f}s of audio")
                    
                    # Save
                    output_path = os.path.join(output_dir, "test_non_async.wav")
                    save_wav(waveform, sample_rate, output_path)
                    print(f"✓ Saved: {output_path}")
                    print(f"\n🎉 SUCCESS! End-to-end TTS working!")
                else:
                    print("✗ No audio data in output")
            else:
                print("✗ No outputs in response")
        else:
            print("✗ No outputs generated")
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n[T+{time.time()-t0:.1f}s] Cleaning up...")
        omni.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    test_non_async()
