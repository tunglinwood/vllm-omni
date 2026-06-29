#!/usr/bin/env python3
"""Test to see what prompt is being generated."""

from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/data1/moonshotai/Kimi-Audio-7B-Instruct",
    trust_remote_code=True
)

# Test messages
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "data:audio/wav;base64,dGVzdA=="
                }
            },
            {
                "type": "text",
                "text": "What does this audio say?"
            }
        ]
    }
]

# Apply chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("Generated prompt:")
print(prompt)
print("\nPrompt tokens:")
tokens = tokenizer.encode(prompt)
print(tokens)
print(f"\nNumber of tokens: {len(tokens)}")

# Check if special tokens are present
special_tokens = [
    "<|im_kimia_user_msg_start|>",
    "<|im_media_begin|>",
    "<|im_kimia_text_blank|>",
    "<|im_media_end|>",
    "<|im_msg_end|>",
    "<|im_kimia_assistant_msg_start|>"
]

print("\nSpecial tokens in prompt:")
for token in special_tokens:
    if token in prompt:
        print(f"  ✅ {token}")
    else:
        print(f"  ❌ {token} (MISSING)")
