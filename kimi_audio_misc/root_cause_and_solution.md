# Kimi Audio Dual Stream - Root Cause Analysis & Solution

## Executive Summary

**ROOT CAUSE IDENTIFIED**: The model expects **discrete audio tokens** (from audio tokenizer) in addition to continuous Whisper features, but vllm-omni only provides BLANK tokens (151666) instead of actual audio tokens.

**Evidence**:
- Model generates EOS with logit 12.25 (vs second-best 10.06) - VERY confident
- This happens even with greedy sampling (temperature=0.0)
- Audio features are correctly extracted and fused
- But the model doesn't see the audio context it was trained with

---

## The Problem

### Reference Implementation Architecture
```python
# From kimia_infer/api/kimia.py
audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
audio_features = history.continuous_feature

# audio_input_ids contains DISCRETE AUDIO TOKENS from audio tokenizer
# audio_features contains CONTINUOUS WHISPER FEATURES

generated_wav_tokens, generated_text_tokens = self._generate_loop(
    audio_input_ids=audio_input_ids,          # ← Discrete tokens
    text_input_ids=text_input_ids,
    continous_feature=audio_features,         # ← Continuous features
    is_continuous_mask=is_continuous_mask,
    ...
)
```

### vllm-omni Implementation (BROKEN)
```python
# Input IDs contain BLANK tokens (151666) instead of audio tokens
# Only continuous features are provided via multimodal_embeddings

# Prompt format:
# [BOS, USER_START, MEDIA_BEGIN, BLANK, BLANK, ..., BLANK, MEDIA_END, text..., MSG_END, ASSISTANT_START]
#                      ↑──── 188 BLANK tokens ────↑

# Audio features are added to BLANK token embeddings
# But model was trained with ACTUAL audio tokens, not BLANK!
```

### Why This Causes Immediate EOS

1. **Training**: Model learned to associate specific audio token patterns with audio content
2. **Inference**: Model sees BLANK tokens (which mean "no audio") + continuous features
3. **Mismatch**: Model doesn't recognize this pattern, defaults to EOS
4. **Result**: EOS logit = 12.25 (very confident), generation stops immediately

---

## The Solution

### Required Architecture Change

vllm-omni needs to:
1. **Tokenize audio** into discrete tokens using the audio tokenizer
2. **Use audio tokens** in input_ids (not BLANK tokens)
3. **Extract continuous features** (Whisper features) as before
4. **Combine** discrete + continuous in the model

### Implementation Steps

#### Step 1: Add Audio Tokenizer to Stage 0

Currently:
- Stage 0: LLM (text processing only)
- Stage 1: Audio detokenizer (audio tokens → waveform)

Need to add:
- Audio tokenizer to Stage 0 (waveform → audio tokens)
- OR pass audio tokens from API server to Stage 0

#### Step 2: Tokenize Audio in Preprocessing

```python
# In KimiAudioMultiModalProcessor or similar
def process_audio(self, audio_path: str) -> dict:
    # 1. Tokenize audio into discrete tokens
    audio_tokens = self.audio_tokenizer.tokenize(audio_path)
    # audio_tokens: [152293, 152294, 152301, ...]
    
    # 2. Extract continuous features
    whisper_features = self.whisper_model.extract_features(audio_path)
    # whisper_features: [1, 128, 370]
    
    return {
        "audio_tokens": audio_tokens,      # NEW: discrete tokens
        "whisper_features": whisper_features,  # Existing: continuous features
    }
```

#### Step 3: Use Audio Tokens in Prompt

```python
# Instead of BLANK tokens, use actual audio tokens
prompt_ids = [
    BOS,
    USER_START,
    MEDIA_BEGIN,
    152293, 152294, 152301, ...,  # ← Actual audio tokens (188 tokens)
    MEDIA_END,
    text_tokens...,
    MSG_END,
    ASSISTANT_START,
]
```

#### Step 4: Combine in Model

```python
# In embed_input_ids
discrete_emb = self.embed_tokens(input_ids)  # Embeds audio tokens
continuous_emb = self._process_audio_input(whisper_features)  # As before

# Combine at audio positions
combined = (discrete_emb[audio_pos] + continuous_emb) * sqrt(2)
inputs_embeds[audio_pos] = combined
```

---

## Why vllm-omni Architecture Makes This Hard

### Current Pipeline
```
API Server
  ↓ (audio URL)
HF Processor (extracts Whisper features only)
  ↓ (whisper_features)
Stage 0: LLM
  ↓ (text + audio tokens)
Stage 1: Audio Detokenizer
  ↓ (waveform)
Output
```

### What's Missing
```
API Server
  ↓ (audio URL)
Audio Tokenizer ← MISSING!
  ↓ (audio_tokens)
HF Processor
  ↓ (whisper_features)
Stage 0: LLM (needs BOTH audio_tokens AND whisper_features)
  ↓
Stage 1: Audio Detokenizer
  ↓
Output
```

### Challenge
vllm-omni's multimodal processing pipeline is designed for:
- Images: pixel values → embeddings
- Video: frames → embeddings
- Audio: waveform → embeddings (continuous only)

But Kimi Audio needs:
- Audio: waveform → **discrete tokens** + **continuous features**

This is a fundamental architectural mismatch.

---

## Possible Solutions

### Option 1: Add Audio Tokenizer to HF Processor (RECOMMENDED)

**Approach**: Modify `KimiAudioMultiModalProcessor` to also tokenize audio

**Pros**:
- Clean integration with vLLM's multimodal pipeline
- Audio tokens available in input_ids
- Minimal changes to model code

**Cons**:
- Need to add audio tokenizer dependency to processor
- May require changes to vLLM's multimodal framework

**Implementation**:
```python
class KimiAudioMultiModalProcessor(BaseMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, ...):
        # Existing: extract Whisper features
        hf_inputs = super()._call_hf_processor(prompt, mm_data, ...)
        
        # NEW: tokenize audio
        if "audio" in mm_data:
            audio_tokens = self.audio_tokenizer.tokenize(mm_data["audio"])
            hf_inputs["audio_tokens"] = audio_tokens
        
        return hf_inputs
    
    def _get_prompt_updates(self, mm_items, ...):
        # Use audio_tokens to replace BLANK tokens in prompt
        audio_tokens = mm_items["audio_tokens"]
        return PromptReplacement(
            target_token=BLANK_TOKEN,
            replacement_tokens=audio_tokens,
        )
```

### Option 2: Two-Stage Audio Processing

**Approach**: Add a preprocessing stage before Stage 0

**Pros**:
- Clear separation of concerns
- Reusable for other models

**Cons**:
- Adds latency (extra stage)
- More complex pipeline

**Implementation**:
```yaml
stage_args:
  # NEW: Audio tokenization stage
  - stage_id: -1
    model_arch: KimiAudioTokenizer
    engine_output_type: audio_tokens
  
  # Stage 0: LLM (receives audio_tokens + whisper_features)
  - stage_id: 0
    engine_input_source: [-1]
    ...
```

### Option 3: Custom Input Preparation (QUICK HACK)

**Approach**: Tokenize audio in API server and pass as part of prompt

**Pros**:
- Quick to implement
- No framework changes

**Cons**:
- Breaks vLLM's abstractions
- Hard to maintain

**Implementation**:
```python
# In API server
audio_tokens = audio_tokenizer.tokenize(audio_path)
whisper_features = whisper_model.extract(audio_path)

# Create custom prompt with audio tokens
prompt_ids = build_prompt_with_audio(audio_tokens, text, whisper_features)

# Pass to vLLM
engine.generate(prompt_ids, ...)
```

---

## Recommendation

**Use Option 1**: Add audio tokenizer to HF processor

This is the cleanest solution that integrates well with vLLM's multimodal framework. It requires:
1. Adding audio tokenizer to `KimiAudioProcessingInfo`
2. Modifying `KimiAudioMultiModalProcessor` to tokenize audio
3. Updating prompt replacement logic to use audio tokens instead of BLANK

**Estimated Effort**: 2-3 days

**Risk**: Medium - requires understanding of vLLM's multimodal framework

---

## Verification

After implementing the fix, verify:
1. ✅ Input IDs contain actual audio tokens (not BLANK)
2. ✅ Audio tokens are in range [152064, 168447]
3. ✅ Top text logits are reasonable (not dominated by EOS)
4. ✅ Model generates meaningful text output
5. ✅ Audio output is also generated

---

## Conclusion

The root cause of the immediate EOS generation is that vllm-omni uses BLANK tokens instead of actual audio tokens. The model was trained with discrete audio tokens and doesn't recognize the BLANK token pattern, causing it to generate EOS with high confidence.

The solution is to add audio tokenization to the preprocessing pipeline and use the resulting audio tokens in the input_ids, matching the reference implementation's architecture.
