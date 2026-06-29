#!/usr/bin/env python3
"""Test Kimi Audio server with audio input."""

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
    "max_tokens": 500,
    "temperature": 0.7,
    "stream": False
}

print("Sending request to Kimi Audio server...")
print(f"Audio size: {len(audio_data)} bytes")
print(f"Request payload: {json.dumps(payload, indent=2)[:500]}...")

try:
    response = requests.post(
        "http://localhost:8091/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60
    )

    print(f"\nResponse status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\n=== Response ===")
        print(json.dumps(result, indent=2))

        # Extract text response
        if "choices" in result and len(result["choices"]) > 0:
            text = result["choices"][0]["message"]["content"]
            print("\n=== Text Output ===")
            print(text)
    else:
        print(f"\nError response:")
        print(response.text)

except requests.exceptions.Timeout:
    print("\nRequest timed out after 60 seconds")
except Exception as e:
    print(f"\nError: {e}")
