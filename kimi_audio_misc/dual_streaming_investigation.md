# Dual Streaming Implementation Investigation for Kimi Audio

## Executive Summary

**Problem**: Kimi Audio requires dual token streams (audio + text) to be fed back at each generation step, but vllm-omni currently only supports single-stream AR generation.

**Current Status**: vllm-omni uses standard AR generation where only `input_ids` (text tokens) are fed back. Audio logits are computed but audio tokens are never fed back, causing text output to be garbage.

**Root Cause**: The fundamental architectural mismatch - vllm-omni's scheduler and model runner manage ONE token sequence per request, but Kimi Audio needs TWO separate sequences (audio and text) that are both embedded and fed back.

---

## 1. Current Architecture Analysis

### 1.1 Generation Loop Flow

```
Scheduler → ModelRunner.execute_model()
    ↓
_preprocess(scheduler_output)
    ↓
    input_ids = self.input_ids.gpu[:num_tokens]  # ← Single token stream
    ↓
_model_forward(input_ids=input_ids, ...)
    ↓
    model.forward(input_ids, positions, ...)
    ↓
_sample(logits)
    ↓
    sampled_token_ids → append to input_ids
    ↓
Next iteration
```

**Key Files**:
- `/root/learning/vllm_integration/vllm-omni/vllm_omni/worker/gpu_ar_model_runner.py` (lines 841-1200)
- `/root/learning/vllm_integration/vllm-omni/vllm_omni/worker/gpu_model_runner.py` (lines 1521-1620)

### 1.2 Where input_ids is Managed

**Location**: `gpu_model_runner.py:1595`
```python
input_ids = self.input_ids.gpu[:num_input_tokens]
```

**Flow**:
1. `self.input_ids` is a persistent buffer (line 285 in `gpu_ar_model_runner.py`)
2. Populated by `_preprocess()` from scheduler output
3. Passed to `model.forward(input_ids=...)`
4. Sampled tokens are appended back to `self.input_ids`
5. Next iteration uses updated `self.input_ids`

**Critical**: There is only ONE `self.input_ids` buffer per model runner. No mechanism exists for managing a second token stream.

### 1.3 Extension Points Found

| Extension Point | Location | Purpose | Can Help? |
|---|---|---|---|
| `prepare_runner_inputs()` | `gpu_ar_model_runner.py:1095` | Model adjusts inputs before forward | ⚠️ Partial - can modify input_ids but can't add second stream |
| `prefer_model_sampler` | `gpu_ar_model_runner.py:1280` | Custom model sampler | ✅ Yes - can sample from both audio and text logits |
| `embed_input_ids()` | Model method | Customize embedding | ✅ Yes - can implement dual-stream embedding |
| `has_preprocess` | `gpu_model_runner.py:1585` | Custom preprocessing | ⚠️ Partial - can preprocess but still single stream |
| `make_omni_output()` | Model method | Extract multimodal outputs | ❌ No - this is for outputs, not inputs |

---

## 2. What Kimi Audio Requires

From `kimi_audio_misc/kimi_audio_model_arch.md` section 4.3:

```python
# Reference implementation (kimia_infer/api/kimia.py)
for step in range(max_new_tokens):
    
    # STEP 1: Forward with BOTH streams
    audio_logits, text_logits, past_key_values = model.forward(
        input_ids=decoder_input_audio_ids,          # Audio stream [1, 1]
        text_input_ids=decoder_input_text_ids,      # Text stream [1, 1]
        position_ids=decoder_position_ids,
        past_key_values=past_key_values,
    )
    
    # STEP 2: Sample BOTH tokens
    next_text_token = sample_text_logits(text_logits)
    next_audio_token = sample_audio_logits(audio_logits)
    
    # STEP 3: Handle audio delay (first 6 are blank)
    if step < 6:
        next_audio_token = BLANK_TOKEN
    
    # STEP 4: Feed BOTH back
    decoder_input_audio_ids = next_audio_token.unsqueeze(1)
    decoder_input_text_ids = next_text_token.unsqueeze(1)
```

**Key Requirements**:
1. Two separate token sequences maintained per request
2. Both sequences passed to `forward()` as separate arguments
3. Both sampled tokens fed back for next step
4. Audio delay handling (first 6 audio tokens are blank)
5. Text stream termination (blank padding after EOS)

---

## 3. Proposed Solutions

### Solution A: Encode Both Streams in Single input_ids (Recommended)

**Concept**: Leverage the unified vocabulary (text: 0-152063, audio: 152064-168447) to encode both streams in a single `input_ids` sequence.

**How It Works**:
```python
# Current (broken):
input_ids = [text_token_1, text_token_2, ...]

# Proposed (fixed):
input_ids = [text_token_1, audio_token_1, text_token_2, audio_token_2, ...]
```

**Implementation**:
1. **Model.forward()**: Split `input_ids` into text and audio streams based on token offset
   ```python
   def forward(self, input_ids, positions, ...):
       # Split by offset
       text_mask = input_ids < 152064
       audio_mask = input_ids >= 152064
       
       text_ids = input_ids[text_mask]
       audio_ids = input_ids[audio_mask] - 152064  # Shift to [0, 16383]
       
       # Embed both
       text_emb = self.embed_tokens(text_ids)
       audio_emb = self.embed_tokens(audio_ids)
       
       # Fuse (interleave or add based on position)
       inputs_embeds = self._fuse_streams(text_emb, audio_emb, positions)
       
       # Forward through backbone...
   ```

2. **Custom Sampler**: Implement `sample()` method with `prefer_model_sampler=True`
   ```python
   def sample(self, logits, sampling_metadata):
       # Split logits into text and audio
       text_logits = logits[:, :152064]
       audio_logits = logits[:, 152064:]
       
       # Sample both
       next_text = sample_text(text_logits)
       next_audio = sample_audio(audio_logits)
       
       # Apply audio delay
       if self.step < 6:
           next_audio = BLANK_TOKEN
       
       # Return interleaved tokens
       return torch.stack([next_text, next_audio], dim=1).flatten()
   ```

3. **Scheduler Integration**: Scheduler already handles single `input_ids`, so no changes needed

**Pros**:
- ✅ Minimal changes to vllm-omni infrastructure
- ✅ Reuses existing scheduler and model runner
- ✅ Works with CUDA graphs (no dynamic control flow)
- ✅ Compatible with batching

**Cons**:
- ⚠️ Requires careful position encoding (interleaved streams)
- ⚠️ May need custom position_ids handling
- ⚠️ Fusion logic must match reference implementation exactly

**Complexity**: Medium (model-side changes only)

---

### Solution B: Custom Generation Loop in Model

**Concept**: Override `execute_model()` in the model to run a custom generation loop that manages two token streams internally.

**How It Works**:
```python
class KimiAudioLLMForConditionalGeneration(nn.Module):
    def execute_model(self, scheduler_output, ...):
        # Custom generation loop
        for step in range(max_new_tokens):
            # Get both streams from internal state
            audio_ids = self.audio_token_buffer[step]
            text_ids = self.text_token_buffer[step]
            
            # Forward with both
            audio_logits, text_logits = self.forward(
                input_ids=audio_ids,
                text_input_ids=text_ids,
                ...
            )
            
            # Sample both
            next_audio = self.sample_audio(audio_logits)
            next_text = self.sample_text(text_logits)
            
            # Store for next step
            self.audio_token_buffer[step+1] = next_audio
            self.text_token_buffer[step+1] = next_text
            
            # Return outputs
            yield OmniOutput(...)
```

**Implementation**:
1. Add `execute_model()` method to `KimiAudioLLMForConditionalGeneration`
2. Maintain internal buffers for audio and text token streams
3. Bypass standard `_model_forward()` and `_sample()` flow
4. Handle KV caching manually for both streams

**Pros**:
- ✅ Full control over generation loop
- ✅ Can implement exact reference behavior
- ✅ Clean separation of concerns

**Cons**:
- ❌ Bypasses vLLM's optimized scheduling and batching
- ❌ Must reimplement KV caching, prefix caching, etc.
- ❌ Complex state management
- ❌ May not work with existing orchestrator

**Complexity**: High (requires deep understanding of vLLM internals)

---

### Solution C: Modify Scheduler to Support Multiple Token Sequences

**Concept**: Extend the scheduler to manage multiple token sequences per request.

**How It Works**:
```python
# Scheduler maintains multiple sequences
class Request:
    request_id: str
    text_token_ids: list[int]      # Text stream
    audio_token_ids: list[int]     # Audio stream
    
# Model runner passes both to forward
def _model_forward(self, ...):
    return self.model.forward(
        input_ids=text_token_ids,
        audio_input_ids=audio_token_ids,
        ...
    )
```

**Implementation**:
1. Modify `SchedulerOutput` to include multiple token sequences per request
2. Update `GPUARModelRunner` to handle multiple `input_ids` buffers
3. Modify `_preprocess()` to prepare both streams
4. Update `model.forward()` signature to accept both streams

**Pros**:
- ✅ Clean architectural solution
- ✅ Matches reference implementation exactly
- ✅ Future-proof for other dual-stream models

**Cons**:
- ❌ Requires changes to vLLM core (scheduler, model runner)
- ❌ May break compatibility with other models
- ❌ Large PR, hard to merge upstream
- ❌ Complex testing

**Complexity**: Very High (requires vLLM core changes)

---

### Solution D: Use additional_information for Audio Stream

**Concept**: Pass audio tokens through the `additional_information` metadata channel.

**How It Works**:
```python
# Model maintains audio stream state
class KimiAudioLLMForConditionalGeneration(nn.Module):
    def forward(self, input_ids, additional_information, ...):
        # Extract audio tokens from metadata
        audio_ids = additional_information.get("audio_token_ids")
        
        # Embed both
        text_emb = self.embed_tokens(input_ids)
        audio_emb = self.embed_tokens(audio_ids)
        
        # Fuse
        inputs_embeds = text_emb + audio_emb
        
        # Forward...
```

**Implementation**:
1. Store audio tokens in `additional_information` dict
2. Extract in `forward()` via `additional_information` parameter
3. Custom sampler returns both text and audio tokens
4. Audio tokens written back to `additional_information` for next step

**Pros**:
- ✅ Uses existing metadata channel
- ✅ No scheduler changes needed
- ✅ Works with current infrastructure

**Cons**:
- ⚠️ `additional_information` is designed for read-only metadata, not mutable state
- ⚠️ May not persist across generation steps correctly
- ⚠️ Unclear if this is the intended use of the channel

**Complexity**: Medium (but may have unintended side effects)

---

## 4. Recommended Approach

**Solution A (Encode Both Streams in Single input_ids)** is recommended because:

1. **Minimal Infrastructure Changes**: Works within existing vllm-omni architecture
2. **Leverages Unified Vocabulary**: The offset-based token separation is already designed for this
3. **Compatible with Optimization**: Works with CUDA graphs, batching, prefix caching
4. **Proven Pattern**: Similar to how multimodal models encode different modalities in a single sequence

### Implementation Plan for Solution A

**Phase 1: Model Changes**
1. Modify `KimiAudioLLMForConditionalGeneration.forward()` to:
   - Split `input_ids` by token offset
   - Implement dual-stream embedding with proper fusion
   - Handle position encoding for interleaved streams

2. Implement custom `sample()` method:
   - Set `prefer_model_sampler = True`
   - Sample from both text and audio logits
   - Apply audio delay logic
   - Return interleaved tokens

3. Implement `prepare_runner_inputs()`:
   - Initialize audio stream with blank tokens
   - Manage stream state across steps

**Phase 2: Testing**
1. Unit tests for dual-stream embedding
2. Unit tests for custom sampler
3. Integration tests with single request
4. Integration tests with batching

**Phase 3: Optimization**
1. Verify CUDA graph compatibility
2. Test with prefix caching
3. Benchmark performance

---

## 5. Key Technical Challenges

### 5.1 Position Encoding

**Problem**: How to assign position IDs when streams are interleaved?

**Options**:
- **Option 1**: Shared positions (both streams use same position IDs)
  ```python
  # input_ids: [text_0, audio_0, text_1, audio_1, ...]
  # positions: [0,      0,       1,      1,       ...]
  ```
- **Option 2**: Separate positions (each stream has own position counter)
  ```python
  # input_ids: [text_0, audio_0, text_1, audio_1, ...]
  # positions: [0,      0,       1,      1,       ...]
  # (same as Option 1, but internally tracked separately)
  ```

**Recommendation**: Option 1 - matches reference implementation where both streams are processed at the same position.

### 5.2 Fusion Logic

**Problem**: How to fuse text and audio embeddings?

**Reference** (from `kimi_audio_model_arch.md`):
```python
inputs_embeds = audio_emb + text_emb
```

**Challenge**: With interleaved streams, we need to:
1. Embed text tokens: `text_emb = embed_tokens(text_ids)`
2. Embed audio tokens: `audio_emb = embed_tokens(audio_ids - offset)`
3. Align by position and add: `inputs_embeds[pos] = text_emb[pos] + audio_emb[pos]`

**Implementation**:
```python
def _fuse_streams(self, text_emb, audio_emb, positions):
    # text_emb: [num_text_tokens, hidden]
    # audio_emb: [num_audio_tokens, hidden]
    # positions: [total_tokens]
    
    # Create output buffer
    batch_size = positions.shape[0] // 2  # Interleaved
    hidden_size = text_emb.shape[-1]
    fused = torch.zeros(batch_size, hidden_size, device=text_emb.device)
    
    # Add text embeddings at even positions
    text_positions = positions[::2]
    fused.scatter_add_(0, text_positions.unsqueeze(1).expand(-1, hidden_size), text_emb)
    
    # Add audio embeddings at odd positions
    audio_positions = positions[1::2]
    fused.scatter_add_(0, audio_positions.unsqueeze(1).expand(-1, hidden_size), audio_emb)
    
    return fused
```

### 5.3 KV Cache Management

**Problem**: How to handle KV cache for dual streams?

**Analysis**: The bifurcation at layer 21 means:
- Layers 0-21: Shared KV cache (both streams contribute)
- Layers 22-27 (text path): Separate KV cache for text
- MIMO layers (audio path): Separate KV cache for audio

**Solution**: 
- Use standard KV cache for layers 0-21 (fused hidden states)
- Use separate KV cache entries for text path (layers 22-27) and audio path (MIMO layers)
- This is already handled by the bifurcation logic in current implementation

### 5.4 Audio Delay

**Problem**: First 6 audio tokens must be blank.

**Solution**: Custom sampler tracks step count and forces blank tokens:
```python
def sample(self, logits, sampling_metadata):
    step = self._get_current_step()
    
    # Sample text
    text_logits = logits[:, :152064]
    next_text = self._sample_text(text_logits)
    
    # Sample audio (with delay)
    if step < 6:
        next_audio = BLANK_TOKEN
    else:
        audio_logits = logits[:, 152064:]
        next_audio = self._sample_audio(audio_logits)
    
    # Return interleaved
    return torch.stack([next_text, next_audio], dim=1).flatten()
```

---

## 6. Files to Modify

### Core Changes
1. **`vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py`**:
   - Modify `forward()` to split and fuse dual streams
   - Implement `sample()` method with `prefer_model_sampler=True`
   - Implement `prepare_runner_inputs()` for stream initialization
   - Add helper methods for stream splitting and fusion

2. **`vllm_omni/model_executor/models/kimi_audio/config_kimi_audio.py`**:
   - Add config flags for dual-stream mode
   - Add audio delay token count

### Testing
3. **`tests/models/kimi_audio/test_dual_stream.py`**:
   - Test stream splitting logic
   - Test fusion logic
   - Test custom sampler
   - Test end-to-end generation

### Documentation
4. **`kimi_audio_misc/kimi_audio_model_arch.md`**:
   - Update section 7 with implementation details
   - Add code examples

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Position encoding mismatch | Medium | High | Careful unit testing, compare with reference |
| Fusion formula incorrect | Low | High | Verify against reference implementation |
| KV cache corruption | Medium | High | Test with prefix caching enabled/disabled |
| CUDA graph incompatibility | Low | Medium | Test with graph capture, fallback to eager if needed |
| Batching issues | Medium | Medium | Test with batch_size > 1 |
| Performance regression | Low | Low | Benchmark against single-stream baseline |

---

## 8. Success Criteria

1. **Functional**: Text output is coherent (not garbage)
2. **Functional**: Audio output is valid (correct token range, proper duration)
3. **Performance**: Generation speed ≥ 10 tokens/sec
4. **Compatibility**: Works with batching (batch_size ≥ 4)
5. **Compatibility**: Works with prefix caching
6. **Compatibility**: Works with CUDA graphs

---

## 9. Next Steps

1. **Implement Solution A** (encode both streams in single input_ids)
2. **Test with single request** to verify correctness
3. **Test with batching** to verify compatibility
4. **Benchmark performance** to ensure no regression
5. **Document implementation** for future reference

---

## 10. Conclusion

Dual streaming for Kimi Audio is achievable within vllm-omni's current architecture by encoding both token streams in a single `input_ids` sequence using the unified vocabulary offset. This approach requires model-side changes only (no scheduler or model runner modifications) and is compatible with vLLM's optimization features.

The key implementation challenges are:
1. Correct stream splitting and fusion in `forward()`
2. Custom sampler that handles both logits and audio delay
3. Proper position encoding for interleaved streams

With careful implementation and testing, this solution should resolve the text garbage issue while maintaining performance and compatibility.
