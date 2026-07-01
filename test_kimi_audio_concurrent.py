#!/usr/bin/env python3
"""Test Kimi Audio with concurrent requests using qa_example.wav."""

import base64
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Read the actual audio file once
audio_path = "/root/learning/Kimi-Audio/test_audios/qa_example.wav"
with open(audio_path, "rb") as f:
    audio_bytes = f.read()

# Convert to base64 once
audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

def send_request(request_id: int, prompt: str) -> dict:
    """Send a single request and return the result."""
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
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }

    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"] if result.get("choices") else ""
            return {
                "request_id": request_id,
                "status": "success",
                "content": content,
                "elapsed": elapsed,
                "tokens": result.get("usage", {}).get("completion_tokens", 0)
            }
        else:
            return {
                "request_id": request_id,
                "status": "error",
                "error": f"HTTP {response.status_code}: {response.text}",
                "elapsed": elapsed
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "request_id": request_id,
            "status": "exception",
            "error": str(e),
            "elapsed": elapsed
        }

def main():
    # Define concurrent requests with different prompts
    prompts = [
        "Can you count from 1 to 10?",
        "What is 2 plus 2?",
        "Say hello in Chinese.",
        "What color is the sky?",
        "Name three fruits."
    ]

    num_requests = len(prompts)
    print(f"=" * 70)
    print(f"Concurrent Request Test: {num_requests} requests")
    print(f"Audio file: {audio_path}")
    print(f"Audio size: {len(audio_bytes)} bytes")
    print(f"=" * 70)
    print()

    # Send all requests concurrently
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [
            executor.submit(send_request, i, prompts[i])
            for i in range(num_requests)
        ]

        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    total_time = time.time() - start_time

    # Sort results by request_id for consistent ordering
    results.sort(key=lambda x: x["request_id"])

    # Print results
    print("Results:")
    print("-" * 70)

    success_count = 0
    total_tokens = 0

    for result in results:
        req_id = result["request_id"]
        status = result["status"]
        elapsed = result["elapsed"]

        if status == "success":
            success_count += 1
            tokens = result.get("tokens", 0)
            total_tokens += tokens
            content = result["content"]

            # Truncate long responses for display
            if len(content) > 100:
                content = content[:100] + "..."

            print(f"[Request {req_id}] ✅ SUCCESS ({elapsed:.2f}s, {tokens} tokens)")
            print(f"  Prompt: {prompts[req_id]}")
            print(f"  Response: {content}")
        else:
            print(f"[Request {req_id}] ❌ {status.upper()} ({elapsed:.2f}s)")
            print(f"  Prompt: {prompts[req_id]}")
            print(f"  Error: {result.get('error', 'Unknown error')}")
        print()

    # Summary
    print("=" * 70)
    print("Summary:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful: {success_count}/{num_requests}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    if success_count > 0:
        print(f"  Average time per request: {total_time/num_requests:.2f}s")
        print(f"  Throughput: {num_requests/total_time:.2f} req/s")
        print(f"  Token throughput: {total_tokens/total_time:.2f} tokens/s")
    print("=" * 70)

if __name__ == "__main__":
    main()
