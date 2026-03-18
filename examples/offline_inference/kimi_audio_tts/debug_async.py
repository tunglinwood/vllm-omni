#!/usr/bin/env python3
"""Debug test for AsyncOmni with Kimi-Audio."""

import asyncio
import sys

from vllm_omni import AsyncOmni
from vllm_omni.inputs.data import OmniTokensPrompt


async def test_async_omni():
    model_path = "/data1/moonshotai/Kimi-Audio-7B-Instruct"
    stage_configs_path = "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml"
    
    print("=" * 60)
    print("AsyncOmni Kimi-Audio TTS Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Config: {stage_configs_path}")
    print()
    
    print("Step 1: Initializing AsyncOmni engine...")
    try:
        omni = AsyncOmni(
            model=model_path,
            stage_configs_path=stage_configs_path,
            enforce_eager=True,
        )
        print("✓ Engine initialized successfully!")
        print(f"  - Stages: {len(omni.stage_list)}")
        print(f"  - Async chunk: {omni.async_chunk}")
        print()
    except Exception as e:
        print(f"✗ Engine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Step 2: Creating request...")
    text = "你好"
    request = OmniTokensPrompt(
        prompt_token_ids=[0] * 128,
        multi_modal_data=None,
        additional_information={
            "text": [text],
            "task_type": ["tts"],
        },
    )
    print(f"✓ Request created for text: '{text}'")
    print()
    
    print("Step 3: Starting generation...")
    chunk_count = 0
    audio_chunks = []
    
    try:
        async for stage_output in omni.generate(request, request_id="test-001"):
            chunk_count += 1
            print(f"  Chunk {chunk_count}: finished={stage_output.finished}")
            
            mm = stage_output.request_output.outputs[0].multimodal_output
            audio_data = mm.get("model_outputs")
            
            if audio_data:
                chunk = audio_data[0].cpu().numpy()
                audio_chunks.append(chunk)
                print(f"    → Audio chunk: {len(chunk)} samples")
            
            if chunk_count > 10:  # Safety limit
                print("  [Breaking due to chunk limit]")
                break
        
        print()
        print(f"Step 4: Generation complete!")
        print(f"  - Total chunks: {chunk_count}")
        print(f"  - Total samples: {sum(len(c) for c in audio_chunks)}")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print()
        print("Step 5: Cleaning up...")
        omni.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(test_async_omni())
