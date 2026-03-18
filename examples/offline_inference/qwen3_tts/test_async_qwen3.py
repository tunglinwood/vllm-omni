#!/usr/bin/env python3
"""Test Qwen3-TTS with AsyncOmni to verify system-level functionality."""

import asyncio
import os
import time

import numpy as np
import soundfile as sf

from vllm_omni import AsyncOmni
from vllm_omni.inputs.data import OmniTokensPrompt


async def test_qwen3_tts():
    # Use Qwen3-TTS CustomVoice model
    model_path = "/data1/modelscope-cache/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    stage_configs_path = "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml"
    output_dir = "/root/learning/vllm-omni/examples/offline_inference/qwen3_tts/output_async_test"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Qwen3-TTS AsyncOmni Test")
    print("=" * 80)
    
    t0 = time.time()
    print(f"\n[T+{time.time()-t0:.1f}s] Initializing AsyncOmni engine...")
    
    try:
        omni = AsyncOmni(
            model=model_path,
            stage_configs_path=stage_configs_path,
            enforce_eager=True,
        )
        
        print(f"[T+{time.time()-t0:.1f}s] ✓ AsyncOmni engine initialized")
        print(f"  - Stages: {len(omni.stage_list)}")
        print(f"  - Async chunk: {omni.async_chunk}")
        
        # Create request for CustomVoice
        print(f"\n[T+{time.time()-t0:.1f}s] Creating request...")
        text = "Hello, this is a test."
        request = OmniTokensPrompt(
            prompt_token_ids=[0] * 128,
            multi_modal_data=None,
            additional_information={
                "task_type": ["CustomVoice"],
                "text": [text],
                "language": ["English"],
                "speaker": ["Vivian"],
                "instruct": [""],
                "max_new_tokens": [512],
            },
        )
        print(f"✓ Request created for: '{text}'")
        
        # Generate
        print(f"\n[T+{time.time()-t0:.1f}s] Starting generation...")
        
        audio_chunks = []
        sample_rate = 24000
        chunk_count = 0
        
        async for stage_output in omni.generate(request, request_id="qwen3-test"):
            chunk_count += 1
            
            if stage_output.finished:
                print(f"\n✓ Generation finished! (chunk {chunk_count})")
                
                mm = stage_output.request_output.outputs[0].multimodal_output
                audio_data = mm.get("audio")
                
                if audio_data:
                    chunk = audio_data[0].cpu().numpy()
                    audio_chunks.append(chunk)
                    print(f"  Final chunk: {len(chunk)} samples")
            else:
                mm = stage_output.request_output.outputs[0].multimodal_output
                audio_data = mm.get("audio")
                
                if audio_data:
                    chunk = audio_data[0].cpu().numpy()
                    audio_chunks.append(chunk)
                    print(f"  Chunk {chunk_count}: {len(chunk)} samples")
        
        # Save audio
        if audio_chunks:
            waveform = np.concatenate(audio_chunks)
            duration = len(waveform) / sample_rate
            print(f"\n✓ Generated {duration:.2f}s of audio ({len(waveform)} samples)")
            
            output_path = os.path.join(output_dir, "test_qwen3_async.wav")
            sf.write(output_path, waveform, sample_rate)
            print(f"✓ Saved: {output_path}")
            print(f"\n🎉 SUCCESS! Qwen3-TTS AsyncOmni working!")
        else:
            print("\n✗ No audio generated")
            
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n[T+{time.time()-t0:.1f}s] Cleaning up...")
        if 'omni' in locals():
            omni.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(test_qwen3_tts())
