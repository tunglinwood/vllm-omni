#!/usr/bin/env python3
"""Test Qwen3-TTS audio generation with SYNC Omni."""

if __name__ == '__main__':
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniTokensPrompt
    import numpy as np
    import soundfile as sf
    import time

    model_path = '/data1/modelscope-cache/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
    stage_configs_path = '/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml'
    output_path = '/tmp/qwen3_test_audio.wav'

    print("=" * 80)
    print("Qwen3-TTS SYNC Omni Audio Generation Test")
    print("=" * 80)

    t0 = time.time()
    print(f"\n[T+{time.time()-t0:.1f}s] Initializing Omni engine...")
    omni = Omni(
        model=model_path,
        stage_configs_path=stage_configs_path,
        enforce_eager=True,
    )
    print(f"[T+{time.time()-t0:.1f}s] ✓ Omni initialized")

    # Create CustomVoice request
    print(f"\n[T+{time.time()-t0:.1f}s] Creating CustomVoice request...")
    request = OmniTokensPrompt(
        prompt_token_ids=[0] * 128,
        multi_modal_data=None,
        additional_information={
            'task_type': ['CustomVoice'],
            'text': ['Hello, this is a test of Qwen3 TTS.'],
            'language': ['English'],
            'speaker': ['Vivian'],
            'instruct': [''],
            'max_new_tokens': [256],
        },
    )
    print(f"✓ Request created")

    # Generate
    print(f"\n[T+{time.time()-t0:.1f}s] Starting generation...")
    outputs = omni.generate([request])
    print(f"[T+{time.time()-t0:.1f}s] ✓ Generation completed")

    # Check output
    if outputs and outputs[0].outputs:
        mm = outputs[0].outputs[0].multimodal_output
        print(f"\nMultimodal output keys: {list(mm.keys())}")
        
        # Try different keys
        for key in ['audio', 'model_outputs', 'waveform']:
            if key in mm:
                audio_data = mm[key]
                if isinstance(audio_data, list) and len(audio_data) > 0:
                    audio = audio_data[0].cpu().numpy()
                    print(f"✓ Found audio in '{key}': {len(audio)} samples")
                    
                    # Get sample rate
                    sr = 24000
                    if 'sr' in mm:
                        sr = int(mm['sr'][0].item()) if isinstance(mm['sr'], list) else int(mm['sr'].item())
                    
                    # Save
                    sf.write(output_path, audio, sr)
                    print(f"✓ Saved to {output_path}")
                    print(f"  Duration: {len(audio)/sr:.2f}s at {sr}Hz")
                    break
        else:
            print("✗ No audio data found in any expected key")
    else:
        print("✗ No outputs generated")

    omni.close()
    print(f"\n[T+{time.time()-t0:.1f}s] ✓ Test complete")
