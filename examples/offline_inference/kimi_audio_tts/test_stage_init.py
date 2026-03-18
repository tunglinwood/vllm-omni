#!/usr/bin/env python3
"""Debug test with explicit stage initialization logging."""

import asyncio
import logging
import sys
import time

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

from vllm_omni import AsyncOmni
from vllm_omni.inputs.data import OmniTokensPrompt


async def test_stage_init():
    model_path = "/data1/moonshotai/Kimi-Audio-7B-Instruct"
    stage_configs_path = "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml"
    
    print("=" * 80)
    print("DEBUG: Stage Initialization Test")
    print("=" * 80)
    
    t0 = time.time()
    print(f"\n[T+{time.time()-t0:.1f}s] Starting AsyncOmni initialization...")
    
    omni = AsyncOmni(
        model=model_path,
        stage_configs_path=stage_configs_path,
        enforce_eager=True,
    )
    
    print(f"[T+{time.time()-t0:.1f}s] ✓ AsyncOmni initialized")
    print(f"  - Stages: {len(omni.stage_list)}")
    print(f"  - Async chunk: {omni.async_chunk}")
    print(f"  - Output handler: {omni.output_handler}")
    
    # Check each stage
    for i, stage in enumerate(omni.stage_list):
        print(f"\n[T+{time.time()-t0:.1f}s] Stage {i} details:")
        print(f"  - Type: {stage.stage_type}")
        print(f"  - Final output: {getattr(stage, 'final_output', False)}")
        print(f"  - Engine input source: {getattr(stage, 'engine_input_source', None)}")
        
        # Check if stage has output queue
        out_q = getattr(stage, '_out_q', None)
        if out_q:
            try:
                size = out_q.qsize()
                print(f"  - Output queue size: {size}")
            except:
                print(f"  - Output queue: exists (size unknown)")
        else:
            print(f"  - Output queue: None")
        
        # Try to collect any pending messages
        result = stage.try_collect()
        if result:
            print(f"  - Pending message: {result.get('type', 'unknown')}")
        else:
            print(f"  - No pending messages")
    
    print(f"\n[T+{time.time()-t0:.1f}s] Test complete")
    omni.close()


if __name__ == "__main__":
    asyncio.run(test_stage_init())
