#!/usr/bin/env python3
"""Minimal test for Kimi-Audio TTS offline inference."""

import sys
import torch

print("=" * 60)
print("Kimi-Audio TTS - Offline Inference Test")
print("=" * 60)

# Step 1: Test imports
print("\n[1/5] Testing imports...")
try:
    from vllm_omni.model_executor.models.kimi_audio_tts import (
        KimiAudioTalkerForConditionalGeneration,
        KimiAudioCode2Wav,
    )
    from vllm_omni.model_executor.models.kimi_audio_tts.configuration_kimi_audio_tts import (
        KimiAudioTalkerConfig,
    )
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Step 2: Test configuration
print("\n[2/5] Testing configuration...")
try:
    config = KimiAudioTalkerConfig()
    print(f"   - hidden_size: {config.hidden_size}")
    print(f"   - num_hidden_layers: {config.num_hidden_layers}")
    print(f"   - num_attention_heads: {config.num_attention_heads}/{config.num_key_value_heads}")
    print(f"   - mimo_layers: {config.mimo_layers}")
    print(f"   - vocab_size: {config.vocab_size}")
    print(f"   - audio_token_offset: {config.audio_token_offset}")
    print("✅ Configuration loaded")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# Step 3: Test model path
print("\n[3/5] Checking model path...")
model_path = "/data1/moonshotai/Kimi-Audio-7B-Instruct"
import os
if os.path.exists(model_path):
    print(f"   ✅ Model found at: {model_path}")
    # Check for audio detokenizer
    detokenizer_path = os.path.join(model_path, "audio_detokenizer", "model.pt")
    if os.path.exists(detokenizer_path):
        print(f"   ✅ Audio detokenizer found: {detokenizer_path}")
    else:
        print(f"   ⚠️  Audio detokenizer not found: {detokenizer_path}")
else:
    print(f"   ❌ Model not found: {model_path}")
    sys.exit(1)

# Step 4: Test stage config
print("\n[4/5] Checking stage configuration...")
stage_config_path = "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml"
if os.path.exists(stage_config_path):
    print(f"   ✅ Stage config found: {stage_config_path}")
    with open(stage_config_path, 'r') as f:
        content = f.read()
        if "async_chunk: true" in content:
            print("   ✅ Async chunk enabled")
        if "kimi_audio_talker" in content:
            print("   ✅ Talker stage configured")
        if "kimi_audio_code2wav" in content:
            print("   ✅ Code2Wav stage configured")
else:
    print(f"   ❌ Stage config not found: {stage_config_path}")
    sys.exit(1)

# Step 5: Test offline inference script
print("\n[5/5] Checking offline inference script...")
inference_script = "/root/learning/vllm-omni/examples/offline_inference/kimi_audio_tts/end2end.py"
if os.path.exists(inference_script):
    print(f"   ✅ Inference script found: {inference_script}")
    print("\n" + "=" * 60)
    print("✅ ALL CHECKS PASSED!")
    print("=" * 60)
    print("\nTo run offline inference:")
    print(f"  cd /root/learning/vllm-omni/examples/offline_inference/kimi_audio_tts")
    print(f"  python end2end.py \\")
    print(f"    --model-path {model_path} \\")
    print(f"    --text 'Hello, this is Kimi-Audio TTS test.' \\")
    print(f"    --output-dir output_audio")
    print("\nOr use the shell script:")
    print(f"  bash run_single_prompt.sh")
    print("=" * 60)
else:
    print(f"   ❌ Inference script not found: {inference_script}")
    sys.exit(1)
