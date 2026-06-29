#!/usr/bin/env python3
"""Debug script to check if audio input reaches the server."""

import requests
import base64
import json

# Load test audio
with open("/root/learning/Kimi-Audio/test_audios/qa_example.wav", "rb") as f:
    audio_data = f.read()

# Encode as base64
audio_b64 = base64.b64encode(audio_data).decode("utf-8")

# Test request with audio input
payload = {
    "model": "/data1/moonshotai/Kimi-Audio-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{audio_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": "What does this audio say?"
                }
            ]
        }
    ],
    "max_tokens": 50,
    "temperature": 0.7,
    "stream": False
}

print("Sending request to Kimi Audio server...")
print(f"Audio size: {len(audio_data)} bytes")

try:
    response = requests.post(
        "http://localhost:8091/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30
    )

    print(f"\nResponse status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")

    if response.status_code == 200:
        result = response.json()
        print("\n=== Response ===")
        print(json.dumps(result, indent=2)[:1000])
    else:
        print(f"\nError response:")
        print(response.text[:500])

except requests.exceptions.Timeout:
    print("\nRequest timed out after 30 seconds")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
