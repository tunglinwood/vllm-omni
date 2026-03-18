#!/usr/bin/env python3
"""Debug test with detailed logging for AsyncOmni."""

import asyncio
import logging
import sys

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

from vllm_omni import AsyncOmni
from vllm_omni.inputs.data import OmniTokensPrompt


async def test_with_debug():
    model_path = "/data1/moonshotai/Kimi-Audio-7B-Instruct"
    stage_configs_path = "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml"
    
    print("=" * 80)
    print("DEBUG: AsyncOmni Kimi-Audio TTS Test")
    print("=" * 80)
    
    # Step 1: Initialize
    print("\n[STEP 1] Initializing AsyncOmni...")
    omni = AsyncOmni(
        model=model_path,
        stage_configs_path=stage_configs_path,
        enforce_eager=True,
    )
    print(f"✓ Engine initialized")
    print(f"  - Stages: {len(omni.stage_list)}")
    print(f"  - Async chunk: {omni.async_chunk}")
    print(f"  - Output handler: {omni.output_handler}")
    
    # Step 2: Check stage status
    print("\n[STEP 2] Checking stage status...")
    for i, stage in enumerate(omni.stage_list):
        print(f"  Stage {i}:")
        print(f"    - Type: {stage.stage_type}")
        print(f"    - Final output: {getattr(stage, 'final_output', False)}")
        print(f"    - Final output type: {getattr(stage, 'final_output_type', None)}")
        print(f"    - Engine input source: {getattr(stage, 'engine_input_source', None)}")
        print(f"    - Custom process func: {getattr(stage, 'custom_process_input_func', None)}")
    
    # Step 3: Create request
    print("\n[STEP 3] Creating request...")
    text = "你好"
    request = OmniTokensPrompt(
        prompt_token_ids=[0] * 128,
        multi_modal_data=None,
        additional_information={
            "text": [text],
            "task_type": ["tts"],
        },
    )
    print(f"✓ Request created")
    
    # Step 4: Start generation with timeout
    print("\n[STEP 4] Starting generation (timeout=60s)...")
    chunk_count = 0
    
    try:
        async with asyncio.timeout(60):
            async for stage_output in omni.generate(request, request_id="debug-001"):
                chunk_count += 1
                print(f"\n✓ Chunk {chunk_count} received!")
                print(f"  - Stage ID: {stage_output.stage_id}")
                print(f"  - Finished: {stage_output.finished}")
                
                if hasattr(stage_output, 'request_output') and stage_output.request_output:
                    mm = stage_output.request_output.outputs[0].multimodal_output
                    audio_data = mm.get("model_outputs")
                    if audio_data:
                        chunk = audio_data[0].cpu().numpy()
                        print(f"  - Audio samples: {len(chunk)}")
                
                if chunk_count >= 5:
                    print("\n[Breaking after 5 chunks]")
                    break
        
        print(f"\n✓ Generation completed successfully!")
        print(f"  Total chunks: {chunk_count}")
        
    except asyncio.TimeoutError:
        print(f"\n✗ TIMEOUT after 60 seconds!")
        print(f"  Chunks received: {chunk_count}")
        
        # Debug: Check output handler status
        print(f"\n[DEBUG] Output handler status:")
        print(f"  - Handler task: {omni.output_handler}")
        print(f"  - Done: {omni.output_handler.done() if omni.output_handler else 'N/A'}")
        print(f"  - Cancelled: {omni.output_handler.cancelled() if omni.output_handler else 'N/A'}")
        
        # Check stage output queues
        print(f"\n[DEBUG] Stage output queues:")
        for i, stage in enumerate(omni.stage_list):
            out_q = stage._out_q
            if out_q:
                try:
                    size = out_q.qsize()
                    print(f"  - Stage {i} output queue size: {size}")
                except:
                    print(f"  - Stage {i} output queue: unknown")
            else:
                print(f"  - Stage {i} output queue: None")
        
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n[STEP 5] Cleaning up...")
        omni.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(test_with_debug())
