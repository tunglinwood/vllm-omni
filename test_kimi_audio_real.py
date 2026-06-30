#!/usr/bin/env python3
"""Test Kimi Audio with the actual qa_example.wav file."""

import base64
import requests
import json

# Read the actual audio file
audio_path = "/root/learning/Kimi-Audio/test_audios/qa_example.wav"
with open(audio_path, "rb") as f:
    audio_bytes = f.read()

# Convert to base64
audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

# Prepare the request
url = "http://localhost:8091/v1/chat/completions"
headers = {"Content-Type": "application/json"}

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

print("Sending request with qa_example.wav...")
print(f"Audio size: {len(audio_bytes)} bytes")

response = requests.post(url, headers=headers, json=payload, timeout=60)

if response.status_code == 200:
    result = response.json()
    print("\n=== Response ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Extract the text content
    if "choices" in result and len(result["choices"]) > 0:
        content = result["choices"][0]["message"]["content"]
        print("\n=== Text Response ===")
        print(content)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
