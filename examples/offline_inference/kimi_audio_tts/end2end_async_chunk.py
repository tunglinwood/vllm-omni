#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline inference with async_chunk enabled via AsyncOmni.

This script demonstrates streaming TTS generation with Kimi-Audio using
async_chunk semantics: downstream stages (Talker, Code2Wav) start
processing as soon as the first chunk of audio codes is available.

Usage:
    python end2end_async_chunk.py --text "你好" --output-dir output_audio

Or with a text file:
    python end2end_async_chunk.py --text-file prompts.txt --output-dir output_audio
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

import numpy as np
import wave

from vllm_omni import AsyncOmni
from vllm_omni.inputs.data import OmniTokensPrompt


def parse_args():
    parser = argparse.ArgumentParser(description="Kimi-Audio TTS with AsyncOmni (Streaming Mode)")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data1/moonshotai/Kimi-Audio-7B-Instruct",
        help="Path to Kimi-Audio model checkpoint",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="你好",
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
        default="output_audio_async_chunk",
        help="Output directory for generated audio",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage configuration YAML (async_chunk mode)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for Stage-0 (Talker)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling for Stage-0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
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
        prompts = ["你好"]
    
    return prompts


def _default_async_chunk_stage_configs_path() -> str | None:
    """Best-effort default stage config for running Kimi-Audio with async_chunk."""
    import os
    
    # Try common locations
    candidates = [
        os.path.join(
            os.path.dirname(__file__),
            "../../../vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml",
        ),
        "/root/learning/vllm-omni/vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml",
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    return None


async def main(args):
    """Run offline inference with AsyncOmni, logging each audio chunk as it arrives."""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(args.text, args.text_file)
    
    print(f"Kimi-Audio TTS Offline Inference (AsyncOmni - Streaming Mode)")
    print(f"Model: {args.model_path}")
    print(f"Prompts: {len(prompts)}")
    print(f"Output: {output_dir}")
    print()
    
    # Prepare stage config
    stage_configs_path = args.stage_configs_path
    if stage_configs_path is None:
        stage_configs_path = _default_async_chunk_stage_configs_path()
        if stage_configs_path is None:
            raise RuntimeError(
                "Stage config not found. Please specify --stage-configs-path "
                "or ensure kimi_audio_tts.yaml is in the default location."
            )
    
    print(f"[Info] Creating AsyncOmni with stage_configs_path={stage_configs_path}")
    print(f"[Info] This will use async_chunk mode for streaming TTS generation")
    print()
    
    # Initialize AsyncOmni engine
    print("Initializing AsyncOmni engine...")
    t0 = time.perf_counter()
    
    omni = AsyncOmni(
        model=args.model_path,
        stage_configs_path=stage_configs_path,
        enforce_eager=True,  # More stable for debugging
    )
    
    t1 = time.perf_counter()
    print(f"Engine initialized in {t1 - t0:.2f} seconds")
    print()
    
    # Process each prompt
    for i, prompt_text in enumerate(prompts):
        request_id = f"{i}_{prompt_text[:20]}"
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
        
        # Streaming generation
        t_start = time.perf_counter()
        t_prev = t_start
        chunk_idx = 0
        audio_chunks = []
        sample_rate = 24000  # Kimi-Audio outputs 24kHz audio
        
        print(f"  Streaming audio chunks...")
        
        async for stage_output in omni.generate(request, request_id=request_id):
            mm = stage_output.request_output.outputs[0].multimodal_output
            
            if not stage_output.finished:
                # Intermediate chunk received
                audio_data = mm.get("model_outputs")
                if audio_data:
                    chunk_tensor = audio_data[0] if isinstance(audio_data, list) else audio_data
                    audio_chunks.append(chunk_tensor.cpu().numpy())
                    
                    t_now = time.perf_counter()
                    dt_ms = (t_now - t_prev) * 1000
                    ttfa_ms = (t_now - t_start) * 1000
                    
                    if chunk_idx == 0:
                        print(f"    Chunk {chunk_idx}: TTFA={ttfa_ms:.1f}ms, samples={len(chunk_tensor)}")
                    else:
                        print(f"    Chunk {chunk_idx}: inter-chunk={dt_ms:.1f}ms, samples={len(chunk_tensor)}")
                    
                    t_prev = t_now
                    chunk_idx += 1
            else:
                # Final chunk received
                t_end = time.perf_counter()
                total_ms = (t_end - t_start) * 1000
                
                # Get final audio data
                audio_data = mm.get("model_outputs")
                if audio_data:
                    chunk_tensor = audio_data[0] if isinstance(audio_data, list) else audio_data
                    audio_chunks.append(chunk_tensor.cpu().numpy())
                
                print(f"    Done: total={total_ms:.1f}ms, chunks={chunk_idx}")
        
        # Concatenate all chunks
        if audio_chunks:
            waveform = np.concatenate(audio_chunks)
            duration_sec = len(waveform) / sample_rate
            print(f"  Generated {duration_sec:.2f}s of audio ({len(waveform)} samples)")
        else:
            print(f"  Warning: No audio data generated")
            waveform = np.zeros(0)
        
        # Save audio
        output_path = output_dir / f"output_{i:04d}.wav"
        save_wav(waveform, sample_rate, str(output_path))
        print(f"  Saved: {output_path}")
        print()
    
    print(f"Generation complete! Output files in {output_dir}")
    
    # Cleanup
    omni.close()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
