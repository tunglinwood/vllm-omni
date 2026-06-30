#!/usr/bin/env python3
"""Test Kimi Audio with audio input to verify the fix."""

import base64
import requests
import json
import sys

# Read the audio file
audio_file = "/root/learning/Kimi-Audio/test_audios/qa_example.wav"
with open(audio_file, "rb") as f:
    audio_data = f.read()

# Encode to base64
audio_base64 = base64.b64encode(audio_data).decode("utf-8")

# Prepare the request
url = "http://localhost:8091/v1/chat/completions"

payload = {
    "model": "/data1/moonshotai/Kimi-Audio-7B-Instruct",
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
                }
            ]
        }
    ],
    "max_tokens": 2048,
    "temperature": 0.0,
    "stream": False
}

print(f"Sending request with audio file: {audio_file}")
print(f"Audio size: {len(audio_data)} bytes")
print(f"Request URL: {url}")

try:
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()
    print("\n=== Response ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Extract the response text
    if "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0]["message"]
        content = message.get("content", "")
        print("\n=== Text Output ===")
        print(content)

        # Check if there's audio output
        if message.get("audio"):
            print("\n=== Audio Output ===")
            print(f"Audio data present: {len(message['audio'].get('data', ''))} characters")
        else:
            print("\n=== No Audio Output ===")

except requests.exceptions.RequestException as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
