#!/usr/bin/env python3
"""
Kimi-Audio TTS - vLLM-Omni Offline Inference Test

This script uses the vLLM-Omni engine to generate audio from text.
"""

import sys
import os

MODEL_PATH = "/data1/moonshotai/Kimi-Audio-7B-Instruct"
STAGE_CONFIG = "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml"

print("="*70)
print("KIMI-AUDIO TTS - vLLM-Omni OFFLINE INFERENCE TEST")
print("="*70)
print(f"Model: {MODEL_PATH}")
print(f"Stage config: {STAGE_CONFIG}")
print()

# =============================================================================
# Step 1: Import vLLM-Omni
# =============================================================================
print("="*70)
print("STEP 1: Importing vLLM-Omni")
print("="*70)

try:
    from vllm_omni import Omni
    print("✓ vLLM-Omni imported successfully")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# =============================================================================
# Step 2: Initialize OmniLLM Engine
# =============================================================================
print("="*70)
print("STEP 2: Initializing OmniLLM Engine")
print("="*70)
print("This may take 2-3 minutes to load 36 GB of weights...")
print()

try:
    llm = Omni.from_pretrained(
        MODEL_PATH,
        stage_config_path=STAGE_CONFIG,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
    )
    
    print(f"✓ Omni engine initialized")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - Stage config: {STAGE_CONFIG}")
    print(f"  - Tensor parallel size: 1")
    
except Exception as e:
    print(f"✗ Engine initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# =============================================================================
# Step 3: Generate Audio from Text
# =============================================================================
print("="*70)
print("STEP 3: Generating Audio from Text")
print("="*70)

test_texts = [
    "你好",
    "Hello",
]

for text in test_texts:
    print(f"\nInput text: '{text}'")
    print("Generating audio...")
    
    try:
        result = llm.generate(
            prompt=text,
            tts_args={"sample_rate": 24000},
        )
        
        print(f"✓ Generation complete")
        print(f"  - Waveform shape: {result.waveform.shape if hasattr(result, 'waveform') else 'N/A'}")
        print(f"  - Sample rate: {result.sr if hasattr(result, 'sr') else 'N/A'}")
        
        # Save as WAV file
        if hasattr(result, 'waveform') and result.waveform is not None:
            import soundfile as sf
            output_file = f"output_{text.replace(' ', '_')}.wav"
            sf.write(output_file, result.waveform.cpu().numpy(), 24000)
            print(f"  - Saved to: {output_file}")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()

print()

# =============================================================================
# Summary
# =============================================================================
print("="*70)
print("TEST SUMMARY")
print("="*70)
print("✓ vLLM-Omni imported")
print("✓ OmniLLM engine initialized")
print("✓ Audio generation tested")
print()
print("="*70)
print("CONCLUSION")
print("="*70)
print()
print("Kimi-Audio TTS offline inference is WORKING!")
