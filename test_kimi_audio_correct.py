#!/usr/bin/env python3
"""Test Kimi Audio with the correct format from reference implementation."""

import base64
import requests
import json

# Read the audio file
audio_path = "/root/learning/Kimi-Audio/test_audios/qa_example.wav"
with open(audio_path, "rb") as f:
    audio_bytes = f.read()

audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

# Use the format from the reference implementation
url = "http://localhost:8091/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# Format 1: Audio-only input (let the model respond to the audio)
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
    "max_tokens": 500,
    "temperature": 0.7
}

print("=" * 80)
print("Test 1: Audio-only input (no text prompt)")
print("=" * 80)
print(f"Audio: {audio_path}")
print(f"Expected response: '当然可以，这很简单。一二三四五六七八九十。'")
print(f"Translation: 'Of course, it's very simple. One two three four five six seven eight nine ten.'")
print()

response = requests.post(url, headers=headers, json=payload, timeout=60)

if response.status_code == 200:
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    tokens = result.get("usage", {}).get("completion_tokens", 0)

    print("✅ Response:")
    print(content)
    print()
    print(f"Tokens: {tokens}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)

print()
print("=" * 80)
