# Kimi Audio Dual Stream Investigation - Final Report

## Executive Summary

**Status**: 🔴 **CRITICAL ISSUE REMAINS**

The audio feature extraction and fusion pipeline is now working correctly, matching the reference implementation. However, the model still generates EOS (end-of-sequence) token immediately at step 1, producing empty text output.

---

## What's Working ✅

### 1. Audio Feature Extraction Pipeline
```
Input: Mel spectrogram [1, 128, 370]
  ↓ HF Processor
Whisper features: mean=-0.322, std=0.418
  ↓ audio_tower (Whisper encoder)
Encoder output: [1, 188, 1280], mean=0.000002, std=0.992
  ↓ whisper_projection
Projected: [1, 188, 5120], mean=-0.007, std=0.574
  ↓ multi_modal_projector (VQAdaptor)
Final features: [1, 188, 3584], mean=-0.000006, std=1.000 ✅
```

### 2. Custom Whisper Encoder
- Implements correct slicing logic: `token_len * 4` where `token_len = (L - 1) // (160 * 8) + 1`
- Produces 188 features for 3.69s audio (matches reference)
- Monkey-patched `_get_feat_extract_output_lengths` to use correct formula

### 3. Fusion Formula (FIXED)
**Before (WRONG)**:
```python
result_emb[multimodal_positions] = audio_features  # REPLACE
result_emb[multimodal_positions] *= sqrt(2)  # SCALE
```

**After (CORRECT)**:
```python
combined = discrete_emb + audio_features  # ADD discrete + continuous
combined = combined * sqrt(2)  # SCALE
result_emb[multimodal_positions] = combined
```

### 4. Debug Values
```
discrete_emb (BLANK tokens): mean=0.000031, std=0.003021
audio_features (continuous): mean=-0.000006, std=1.000000
combined_emb (after √2): mean=0.000044, std=1.414062
```

The fusion is mathematically correct and matches the reference implementation.

---

## What's Broken ❌

### Critical Issue: Immediate EOS Generation
**Symptom**: Model generates EOS token (151667) at step 1, producing empty text output

**Logs Show**:
```
[KimiAudio] slot 0 step 1: text_token=151667 (EOS), audio_token=151666 (BLANK)
```

**Impact**: 
- Text output: "" (empty)
- Audio output: Generated (but quality unknown)
- finish_reason: "stop"
- stop_reason: 151667 (EOS token)

---

## Root Cause Analysis

### Hypothesis 1: Architectural Mismatch (PARTIALLY ADDRESSED)
**Issue**: Reference implementation uses THREE inputs:
1. Discrete audio tokens (input_ids)
2. Text tokens (text_input_ids)
3. Continuous features (whisper_input_feature) with mask (is_continuous_mask)

**vllm-omni Implementation**:
- Only provides continuous features via multimodal_embeddings
- Discrete tokens are embedded separately and combined in embed_input_ids
- No explicit is_continuous_mask mechanism

**Status**: The fusion formula now matches the reference (ADD + √2), but the overall architecture still differs.

### Hypothesis 2: Missing Components (LIKELY)
The reference implementation may have additional components that vllm-omni lacks:
1. **Repetition penalty sampler**: Reference uses custom `KimiASampler` with repetition penalty
2. **Specific sampling parameters**: Temperature, top-k, top-p may need tuning
3. **Audio token feedback during decode**: May not be working correctly
4. **Training/inference mismatch**: Model may expect different input format

### Hypothesis 3: Numerical Issues (UNLIKELY)
Debug logs show proper numerical values:
- Audio features are properly normalized (std=1.0)
- Discrete embeddings are small but non-zero
- Fusion produces expected results (std=1.414 = √2)

---

## Comparison with Reference Implementation

### Reference Architecture
```python
# From modeling_moonshot_kimia.py
audio_emb = self.embed_tokens(input_ids)  # Discrete tokens
expanded_whisper[start:end] = whisper_input_feature  # Continuous features
whisper_emb = self.vq_adaptor(expanded_whisper)
combined = (audio_emb + whisper_emb) * sqrt(2)  # ADD + SCALE
audio_emb = audio_emb * (~mask) + combined * mask  # Apply at masked positions
if text_input_ids:
    inputs_embeds = audio_emb + embed(text_input_ids)  # Add text
```

### vllm-omni Architecture
```python
# From kimi_audio_llm.py
text_emb = self.embed_tokens(input_ids)  # Includes BLANK tokens
audio_features = self._process_audio_input(whisper_features)  # Through encoder + VQAdaptor
combined = text_emb[multimodal_pos] + audio_features  # ADD
combined = combined * sqrt(2)  # SCALE
text_emb[multimodal_pos] = combined  # Place back
```

**Key Differences**:
1. Reference uses explicit `is_continuous_mask`; vllm-omni uses `is_multimodal` from vLLM framework
2. Reference passes raw features to model; vllm-omni pre-processes through encoder
3. Reference has custom sampler with repetition penalty; vllm-omni uses standard sampler
4. Reference maintains separate audio/text streams during decode; vllm-omni uses per-slot state

---

## Files Modified

1. **`vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py`**
   - Added `KimiAudioCustomWhisperEncoder` class (lines ~63-120)
   - Added `_patch_output_length_calculation()` function (lines ~63-95)
   - Fixed fusion formula in `embed_input_ids` (lines ~600-640)
   - Added comprehensive debug logging throughout pipeline
   - Removed duplicate `audio_tower` assignment

---

## Test Results

### Test: Audio Input with Fixed Fusion
```bash
python test_kimi_audio_input.py
```

**Result**:
- ✅ Audio features extracted: [188, 3584]
- ✅ Features placed at correct positions: [2, 3, ..., 189]
- ✅ Fusion formula correct: ADD + √2 scaling
- ✅ All numerical values within expected ranges
- ❌ Text output: "" (empty)
- ❌ EOS generated at step 1: text_token=151667
- ✅ Audio output: Generated (63 seconds)

---

## Next Steps

### Priority 1: Implement Custom Sampler (HIGH)
Reference uses `KimiASampler` with repetition penalty. This may be critical for preventing immediate EOS.

**Action**: 
1. Find reference implementation of `KimiASampler`
2. Implement in vllm-omni
3. Test if this resolves the EOS issue

### Priority 2: Verify Audio Token Feedback (MEDIUM)
Ensure audio tokens from previous steps are being fed back correctly during decode.

**Action**:
1. Add logging to verify `_audio_state` is being populated
2. Verify audio tokens are being added to embeddings during decode
3. Check if this affects text generation

### Priority 3: Test with Reference Implementation (MEDIUM)
Verify the audio file is valid and produces expected output with reference implementation.

**Action**:
1. Run reference implementation with same audio file
2. Compare intermediate outputs
3. Identify any discrepancies

### Priority 4: Tune Sampling Parameters (LOW)
Experiment with different sampling parameters (temperature, top-k, top-p).

**Action**:
1. Try different temperature values (0.0, 0.5, 1.0)
2. Try different top-k values (1, 5, 10, 50)
3. Try different top-p values (0.9, 0.95, 1.0)

---

## Conclusion

The audio feature extraction and fusion pipeline is now fully functional and matches the reference implementation. The fusion formula has been corrected to ADD discrete token embeddings with continuous features, then scale by √2.

However, the model still generates EOS immediately, producing empty text output. This suggests a deeper issue beyond the fusion formula, likely related to:
1. Missing custom sampler with repetition penalty
2. Sampling parameter mismatch
3. Audio token feedback issues during decode
4. Training/inference architectural differences

**Recommendation**: Focus on implementing the custom sampler with repetition penalty, as this is the most likely cause of the immediate EOS generation.

---

## Appendix: Debug Logs

### Audio Processing Pipeline
```
=== DEBUG: _parse_and_validate_audio_input ===
whisper_features from HF processor:
  shape: torch.Size([1, 128, 370])
  dtype: torch.bfloat16
  mean: -0.322266, std: 0.417969

=== DEBUG: _process_audio_input ===
Input whisper_features: shape=[1, 128, 370], mean=-0.322266, std=0.417969
After audio_tower (Whisper encoder): shape=[1, 188, 1280], mean=0.000002, std=0.992188
After whisper_projection: shape=[1, 188, 5120], mean=-0.006714, std=0.574219
After multi_modal_projector (VQAdaptor): shape=[1, 188, 3584], mean=-0.000006, std=1.000000

=== DEBUG: embed_input_ids ===
discrete_emb (BLANK tokens): mean=0.000031, std=0.003021
audio_features (continuous): mean=-0.000006, std=1.000000
combined_emb (after √2): mean=0.000044, std=1.414062
✅ Audio features ADDED to discrete tokens and scaled by √2
```

### Generation Logs
```
[KimiAudio] slot 0 step 1: text_token=151667 (EOS), audio_token=151666 (BLANK), text_finished=False
```
