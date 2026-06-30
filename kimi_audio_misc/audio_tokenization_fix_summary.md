# Kimi Audio Tokenization Fix - Implementation Summary

## Date: 2026-06-30
## Status: ✅ IMPLEMENTED - Testing in Progress

---

## Root Cause (Previously Identified)

The model was generating EOS immediately because:
1. **Model expects**: Discrete audio tokens from Glm4Tokenizer (e.g., [152293, 152294, ...])
2. **vllm-omni provided**: BLANK tokens (151666) instead
3. **Result**: Model doesn't recognize BLANK pattern → generates EOS with high confidence (logit=12.25)

---

## Solution Implemented

### Changes Made to `kimi_audio_llm.py`

#### 1. Added Glm4Tokenizer Initialization (lines ~287-310)
```python
# NEW: Audio tokenizer for discrete audio tokens
try:
    from transformers import AutoTokenizer
    print("[KimiAudio] Loading Glm4Tokenizer for discrete audio tokenization...", flush=True)
    self.audio_tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/glm-4-voice-tokenizer",
        trust_remote_code=True
    )
    # Move to GPU if possible
    if hasattr(self.audio_tokenizer, 'to'):
        import torch
        self.audio_tokenizer = self.audio_tokenizer.to(torch.cuda.current_device())
    print("[KimiAudio] ✅ Glm4Tokenizer loaded successfully", flush=True)
except Exception as e:
    print(f"[KimiAudio] ❌ WARNING: Failed to load Glm4Tokenizer: {e}", flush=True)
    self.audio_tokenizer = None

# Store audio tokens for current request
self._current_audio_tokens: Optional[list[int]] = None
```

#### 2. Added Audio Tokenization Method (lines ~539-598)
```python
def _tokenize_audio_to_discrete_tokens(self, raw_audio: Any) -> Optional[list[int]]:
    """Tokenize audio into discrete tokens using Glm4Tokenizer."""
    if self.audio_tokenizer is None:
        return None

    try:
        # Convert raw audio to file path if needed
        # ... (handles numpy arrays, tensors, and file paths)

        # Tokenize audio using Glm4Tokenizer
        wav_tokens = self.audio_tokenizer.tokenize(audio_path=audio_path)

        # Add offset to get actual token IDs
        wav_tokens = wav_tokens + self._token_offset  # 152064

        # Convert to list
        wav_tokens_list = wav_tokens.squeeze(0).cpu().numpy().tolist()

        return wav_tokens_list
    except Exception as e:
        print(f"[KimiAudio] ❌ ERROR: Failed to tokenize audio: {e}", flush=True)
        return None
```

#### 3. Modified `_parse_and_validate_audio_input` (lines ~457-490)
```python
# NEW: Also tokenize audio into discrete tokens
discrete_tokens = self._tokenize_audio_to_discrete_tokens(raw_audio)
if discrete_tokens is not None:
    self._current_audio_tokens = discrete_tokens
    print(f"[KimiAudio] ✅ Stored {len(discrete_tokens)} discrete audio tokens", flush=True)
else:
    print(f"[KimiAudio] ⚠️ Could not tokenize audio, will use BLANK tokens", flush=True)
    self._current_audio_tokens = None
```

#### 4. Modified `forward()` to Replace BLANK Tokens (lines ~306-340)
```python
# NEW: Replace BLANK tokens in input_ids with actual audio tokens
if self._current_audio_tokens is not None and input_ids is not None:
    try:
        # Find positions where input_ids contains BLANK tokens
        blank_mask = (input_ids == self._blank_token_id)
        num_blank_positions = blank_mask.sum().item()

        if num_blank_positions > 0:
            print(f"[KimiAudio] forward: Found {num_blank_positions} BLANK tokens in input_ids", flush=True)

            # Check if we have the right number of audio tokens
            if len(self._current_audio_tokens) == num_blank_positions:
                # Replace BLANK tokens with actual audio tokens
                audio_tokens_tensor = torch.tensor(
                    self._current_audio_tokens,
                    dtype=input_ids.dtype,
                    device=input_ids.device
                )
                input_ids = input_ids.clone()  # Don't modify original
                input_ids[blank_mask] = audio_tokens_tensor
                print(f"[KimiAudio] ✅ Replaced BLANK tokens with discrete audio tokens", flush=True)
                print(f"[KimiAudio] First 5 audio tokens: {self._current_audio_tokens[:5]}", flush=True)
            else:
                print(f"[KimiAudio] ⚠️ Mismatch: {num_blank_positions} BLANK positions but {len(self._current_audio_tokens)} audio tokens", flush=True)
    except Exception as e:
        print(f"[KimiAudio] ❌ ERROR replacing BLANK tokens: {e}", flush=True)
```

---

## Expected Behavior After Fix

### Before Fix (BROKEN)
```
Input IDs: [BOS, USER_START, MEDIA_BEGIN, BLANK×188, MEDIA_END, text..., MSG_END, ASSISTANT_START]
                                         ↑
                                    BLANK tokens (151666)

Result: Model generates EOS immediately (logit=12.25)
Text output: "" (empty)
```

### After Fix (EXPECTED)
```
Input IDs: [BOS, USER_START, MEDIA_BEGIN, 152293, 152294, ..., 168158, MEDIA_END, text..., MSG_END, ASSISTANT_START]
                                         ↑
                                    Actual audio tokens from Glm4Tokenizer

Result: Model recognizes audio context
Text output: "Sure, I can count from 1 to 10..." (meaningful text)
Audio output: Generated (dual stream)
```

---

## Test Results

### Server Startup
```
✅ Glm4Tokenizer loaded successfully
✅ Model weights loaded (36 shards)
✅ FlashInfer autotuning complete
✅ Server started on port 8091
```

### Pending Test
```bash
python test_kimi_audio_input.py
```

Expected logs:
```
[KimiAudio] Tokenizing audio: /tmp/tmpXXX.wav
[KimiAudio] ✅ Tokenized audio into 188 discrete tokens
[KimiAudio] First 5 tokens: [152293, 152294, 152301, ...]
[KimiAudio] ✅ Stored 188 discrete audio tokens
[KimiAudio] forward: Found 188 BLANK tokens in input_ids
[KimiAudio] forward: Will replace with 188 discrete audio tokens
[KimiAudio] ✅ Replaced BLANK tokens with discrete audio tokens
[KimiAudio] First 5 audio tokens: [152293, 152294, 152301, ...]
```

Expected output:
- Text: Meaningful response (not empty)
- Audio: Generated (dual stream working)
- finish_reason: "stop" (but after generating content, not immediately)

---

## Verification Checklist

- [ ] Glm4Tokenizer loads successfully ✅ (confirmed in logs)
- [ ] Audio tokenization works (pending test)
- [ ] BLANK tokens are replaced with audio tokens (pending test)
- [ ] Model generates meaningful text (pending test)
- [ ] Audio output is also generated (pending test)
- [ ] Dual stream inference works (pending test)

---

## Key Insights

1. **Audio tokenization is critical**: The model was trained with discrete audio tokens, not continuous features alone
2. **Dual input required**: Model needs BOTH discrete tokens (in input_ids) AND continuous features (in multimodal_embeddings)
3. **Token offset**: Glm4Tokenizer returns base tokens, need to add offset (152064) to get actual token IDs
4. **Timing**: Tokenization happens in preprocessing, replacement happens in forward()

---

## Next Steps

1. Run `python test_kimi_audio_input.py` and verify:
   - Audio is tokenized into discrete tokens
   - BLANK tokens are replaced
   - Text output is meaningful
   - Audio output is generated

2. If successful:
   - Test with multiple audio files
   - Test batched inference
   - Verify audio quality

3. If issues remain:
   - Check token count mismatch (188 BLANK positions vs actual audio tokens)
   - Verify audio tokenizer is working correctly
   - Check if there are other architectural differences

---

## Files Modified

1. `vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py`
   - Added Glm4Tokenizer initialization (~25 lines)
   - Added `_tokenize_audio_to_discrete_tokens` method (~60 lines)
   - Modified `_parse_and_validate_audio_input` (~15 lines)
   - Modified `forward()` to replace BLANK tokens (~35 lines)
   - Total: ~135 lines added/modified

---

## Conclusion

The fix has been implemented and the server is starting up. The key change is that we now tokenize input audio into discrete tokens using Glm4Tokenizer and replace the BLANK tokens in input_ids with these actual audio tokens before passing them to the model.

This should resolve the immediate EOS generation issue and allow the model to generate meaningful text output while also producing audio output (dual stream).

**Status**: Testing in progress...
