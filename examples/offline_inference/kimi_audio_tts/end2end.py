#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline inference script for Kimi-Audio TTS."""

import argparse
import os
import wave
from pathlib import Path

import numpy as np
import torch

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniTokensPrompt


def parse_args():
    parser = argparse.ArgumentParser(description="Kimi-Audio TTS Offline Inference")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data1/moonshotai/Kimi-Audio-7B-Instruct",
        help="Path to Kimi-Audio model checkpoint",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of Kimi-Audio text-to-speech synthesis.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to file containing text prompts (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_audio",
        help="Output directory for generated audio",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage configuration YAML",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming mode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def save_wav(waveform: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Save waveform to WAV file."""
    # Ensure waveform is in correct format
    if waveform.ndim == 2:
        waveform = waveform.squeeze()
    
    # Normalize to 16-bit range
    waveform = np.clip(waveform, -1.0, 1.0)
    waveform_int16 = (waveform * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform_int16.tobytes())


def load_prompts(text: str | None, text_file: str | None) -> list[str]:
    """Load text prompts from arguments or file."""
    prompts = []
    
    if text_file is not None:
        with open(text_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif text is not None:
        prompts = [text]
    else:
        prompts = ["Hello, this is a test of Kimi-Audio TTS."]
    
    return prompts


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(args.text, args.text_file)
    
    print(f"Kimi-Audio TTS Offline Inference")
    print(f"Model: {args.model_path}")
    print(f"Prompts: {len(prompts)}")
    print(f"Output: {output_dir}")
    print()
    
    # Initialize engine
    print("Initializing OmniEngine...")
    
    # Prepare stage config
    stage_configs_path = args.stage_configs_path
    if stage_configs_path is None:
        # Use default config - use absolute path
        stage_configs_path = "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml"
    
    # Create engine
    engine = Omni(
        model=args.model_path,
        stage_configs_path=stage_configs_path,
        enforce_eager=False,
        gpu_memory_utilization=0.5,
    )
    
    print("Engine initialized successfully")
    print()
    
    # Process prompts
    for i, prompt_text in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: {prompt_text[:50]}...")
        
        # Create request with placeholder token IDs
        # The AR scheduler requires prompt_token_ids to have the correct length
        # even though they will be replaced by embeddings during preprocessing
        estimated_len = max(64, len(prompt_text) * 4)  # Rough estimate
        request = OmniTokensPrompt(
            prompt_token_ids=[0] * estimated_len,
            multi_modal_data=None,
            additional_information={
                "text": [prompt_text],
                "task_type": ["tts"],
            },
        )
        
        # Generate
        if args.streaming:
            # Streaming mode
            print("  Streaming mode...")
            audio_chunks = []
            sample_rate = 24000
            
            for output in engine.generate([request], streaming=True):
                if output.outputs and output.outputs[0].multimodal_output:
                    audio_data = output.outputs[0].multimodal_output.get("model_outputs")
                    if audio_data:
                        audio_chunks.append(audio_data[0].cpu().numpy())
            
            # Concatenate chunks
            if audio_chunks:
                waveform = np.concatenate(audio_chunks)
            else:
                waveform = np.zeros(0)
        else:
            # Non-streaming mode
            outputs = engine.generate([request])
            
            if outputs and outputs[0].outputs:
                multimodal = outputs[0].outputs[0].multimodal_output
                audio_data = multimodal.get("model_outputs", [])
                sample_rates = multimodal.get("sr", [24000])
                
                if audio_data and len(audio_data) > 0:
                    waveform = audio_data[0].cpu().numpy()
                    sample_rate = int(sample_rates[0].item()) if sample_rates else 24000
                else:
                    print("  Warning: No audio data in output")
                    waveform = np.zeros(0)
                    sample_rate = 24000
            else:
                print("  Warning: No outputs generated")
                waveform = np.zeros(0)
                sample_rate = 24000
        
        # Save audio
        output_path = output_dir / f"output_{i:04d}.wav"
        save_wav(waveform, sample_rate, str(output_path))
        print(f"  Saved: {output_path} ({len(waveform)/sample_rate:.2f}s)")
    
    print()
    print(f"Generation complete! Output files in {output_dir}")


if __name__ == "__main__":
    main()
