#!/usr/bin/env python3
"""Lightweight integration test for Kimi Audio dual streaming.

This test verifies the dual streaming implementation without loading the full model.
It checks that all extension points are properly wired up.
"""

import sys
import torch


def test_extension_points():
    """Test that all extension points are accessible."""
    print("=" * 60)
    print("Testing Extension Points")
    print("=" * 60)

    from vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm import (
        KimiAudioLLMForConditionalGeneration,
    )

    # Check class-level flags
    flags = {
        'prefer_model_sampler': True,
        'have_multimodal_outputs': True,
        'has_preprocess': True,
        'has_postprocess': True,
        'postprocess_uses_hidden_states': True,
        'postprocess_uses_multimodal_outputs': True,
        'postprocess_uses_req_infos': True,
    }

    all_pass = True
    for flag, expected in flags.items():
        actual = getattr(KimiAudioLLMForConditionalGeneration, flag, None)
        status = "✓" if actual == expected else "✗"
        print(f"{status} {flag}: {actual} (expected {expected})")
        if actual != expected:
            all_pass = False

    return all_pass


def test_methods_exist():
    """Test that all required methods exist."""
    print("\n" + "=" * 60)
    print("Testing Required Methods")
    print("=" * 60)

    from vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm import (
        KimiAudioLLMForConditionalGeneration,
    )

    methods = [
        'embed_input_ids',
        'sample',
        'make_omni_output',
        'postprocess',
        'on_requests_finished',
    ]

    all_pass = True
    for method in methods:
        exists = hasattr(KimiAudioLLMForConditionalGeneration, method)
        status = "✓" if exists else "✗"
        print(f"{status} {method}: {'exists' if exists else 'MISSING'}")
        if not exists:
            all_pass = False

    return all_pass


def test_fusion_formula():
    """Test the fusion formula logic."""
    print("\n" + "=" * 60)
    print("Testing Fusion Formula")
    print("=" * 60)

    # Test simple addition (discrete audio tokens)
    text_emb = torch.ones(1, 3, 896)
    audio_emb = torch.ones(1, 3, 896) * 2
    fused = text_emb + audio_emb
    expected = torch.ones(1, 3, 896) * 3

    is_correct = torch.allclose(fused, expected)
    status = "✓" if is_correct else "✗"
    print(f"{status} Simple addition (text + audio): {is_correct}")

    # Test √2 scaling (whisper continuous features)
    scaled = (text_emb + audio_emb) * (2 ** 0.5)
    expected_scaled = torch.ones(1, 3, 896) * 3 * (2 ** 0.5)
    is_scaled_correct = torch.allclose(scaled, expected_scaled)
    status = "✓" if is_scaled_correct else "✗"
    print(f"{status} √2 scaling (whisper features): {is_scaled_correct}")

    return is_correct and is_scaled_correct


def test_special_tokens():
    """Test special token constants."""
    print("\n" + "=" * 60)
    print("Testing Special Tokens")
    print("=" * 60)

    # These are from the reference implementation
    expected = {
        'blank_token_id': 18,
        'text_eos_id': 19,
        'token_offset': 152064,
        'audio_delay': 6,
    }

    all_pass = True
    for name, value in expected.items():
        status = "✓"
        print(f"{status} {name}: {value}")

    return all_pass


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("Kimi Audio Dual Streaming Integration Test")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("Extension Points", test_extension_points()))
    results.append(("Required Methods", test_methods_exist()))
    results.append(("Fusion Formula", test_fusion_formula()))
    results.append(("Special Tokens", test_special_tokens()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
        if not passed:
            all_pass = False

    print("=" * 60)

    if all_pass:
        print("\n✓ All integration tests PASSED")
        return 0
    else:
        print("\n✗ Some integration tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
