#!/usr/bin/env python3
"""
Test audio quality by:
1. Sending qa_example.wav to vllm-omni (GPU 6) to generate audio response
2. Using vllm (GPU 7) to transcribe the generated audio
3. Comparing transcription quality
"""

import base64
import requests
import json
import tempfile
import os

# Configuration
VLLM_OMNI_URL = "http://localhost:8091/v1/chat/completions"  # GPU 6
VLLM_ASR_URL = "http://localhost:8092/v1/chat/completions"   # GPU 7
MODEL_PATH = "/data1/moonshotai/Kimi-Audio-7B-Instruct"
AUDIO_INPUT = "/root/learning/Kimi-Audio/test_audios/qa_example.wav"

def load_audio_as_base64(audio_path):
    """Load audio file and convert to base64."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

def save_base64_to_file(base64_data, output_path):
    """Save base64 audio data to file."""
    audio_bytes = base64.b64decode(base64_data)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    return output_path

def step1_generate_audio_with_vllm_omni():
    """Send qa_example.wav to vllm-omni and get audio response."""
    print("=" * 80)
    print("STEP 1: Generate audio with vllm-omni (GPU 6)")
    print("=" * 80)

    audio_base64 = load_audio_as_base64(AUDIO_INPUT)
    print(f"Input audio: {AUDIO_INPUT}")
    print(f"Input size: {os.path.getsize(AUDIO_INPUT)} bytes")

    payload = {
        "model": MODEL_PATH,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Can you count from 1 to 10?"
                    }
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }

    print(f"\nSending request to {VLLM_OMNI_URL}...")
    response = requests.post(VLLM_OMNI_URL, json=payload, timeout=120)

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None, None

    result = response.json()

    # Extract text and audio from response
    text_content = result["choices"][0]["message"]["content"]
    audio_data = result["choices"][0]["message"].get("audio", {})

    print(f"\n✅ Text response from vllm-omni:")
    print(f"   {text_content}")

    if audio_data and "data" in audio_data:
        audio_base64_out = audio_data["data"]
        print(f"\n✅ Audio output generated (size: {len(audio_base64_out)} bytes base64)")

        # Save to temp file
        output_path = "/tmp/vllm_omni_output.wav"
        save_base64_to_file(audio_base64_out, output_path)
        print(f"   Saved to: {output_path}")
        print(f"   File size: {os.path.getsize(output_path)} bytes")

        return text_content, output_path
    else:
        print("\n❌ No audio output in response")
        return text_content, None

def step2_transcribe_with_vllm(audio_path):
    """Use vllm (GPU 7) to transcribe the audio."""
    print("\n" + "=" * 80)
    print("STEP 2: Transcribe audio with vllm (GPU 7)")
    print("=" * 80)

    if not audio_path or not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return None

    audio_base64 = load_audio_as_base64(audio_path)
    print(f"Input audio: {audio_path}")
    print(f"Input size: {os.path.getsize(audio_path)} bytes")

    # Send audio to vllm with ASR prompt
    payload = {
        "model": MODEL_PATH,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Please transcribe this audio exactly as spoken."
                    }
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.0  # Greedy for accurate transcription
    }

    print(f"\nSending request to {VLLM_ASR_URL}...")
    response = requests.post(VLLM_ASR_URL, json=payload, timeout=120)

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

    result = response.json()
    transcription = result["choices"][0]["message"]["content"]

    print(f"\n✅ Transcription from vllm (GPU 7):")
    print(f"   {transcription}")

    return transcription

def step3_compare_results(original_text, transcription):
    """Compare the original response with transcription."""
    print("\n" + "=" * 80)
    print("STEP 3: Quality Comparison")
    print("=" * 80)

    print(f"\nOriginal text (from vllm-omni):")
    print(f"   {original_text}")

    print(f"\nTranscription (from vllm GPU 7):")
    print(f"   {transcription}")

    # Simple similarity check
    if original_text and transcription:
        # Remove extra spaces for comparison
        orig_clean = " ".join(original_text.split())
        trans_clean = " ".join(transcription.split())

        if orig_clean == trans_clean:
            print("\n✅ PERFECT MATCH: Transcription matches original exactly!")
        else:
            print("\n⚠️  MISMATCH: Transcription differs from original")
            print(f"\nOriginal (cleaned):  {orig_clean}")
            print(f"Transcription:       {trans_clean}")

            # Calculate simple word overlap
            orig_words = set(orig_clean.lower().split())
            trans_words = set(trans_clean.lower().split())
            overlap = len(orig_words & trans_words)
            total = len(orig_words | trans_words)
            similarity = overlap / total if total > 0 else 0
            print(f"\nWord overlap similarity: {similarity:.2%}")

def main():
    print("\n🎵 Kimi Audio Quality Test")
    print("Testing audio output quality by transcribing with separate vllm instance\n")

    # Step 1: Generate audio with vllm-omni
    original_text, audio_path = step1_generate_audio_with_vllm_omni()

    if not audio_path:
        print("\n❌ Failed to generate audio, stopping test")
        return

    # Step 2: Transcribe with vllm
    transcription = step2_transcribe_with_vllm(audio_path)

    # Step 3: Compare
    if transcription:
        step3_compare_results(original_text, transcription)

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
