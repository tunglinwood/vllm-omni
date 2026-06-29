# Kimi Audio Implementation Comparison

## Reference Implementation Analysis

### Key Findings from `/root/learning/Kimi-Audio/kimia_infer/api/kimia.py`

1. **Dual Token Streams**: The reference maintains SEPARATE token streams for audio and text throughout generation:
   - `decoder_input_audio_ids` - audio tokens
   - `decoder_input_text_ids` - text tokens
   - Both are embedded and fused: `inputs_embeds = audio_emb + self.embed_tokens(text_input_ids)` (line 757)

2. **Token Filtering** (lines 266-276):
   - Audio tokens: `t >= kimia_token_offset` (152064), then subtract offset
   - Text tokens: `t < kimia_token_offset`
   - This means the model outputs from a UNIFIED vocabulary (168448 tokens)
   - Audio tokens are in range [152064, 168447] (16384 audio tokens)
   - Text tokens are in range [0, 152063]

3. **Bifurcation Logic** (modeling_moonshot_kimia.py lines 788-789):
   ```python
   hidden_states = layer_outputs[0]
   if idx == self.kimia_mimo_transformer_from_layer_index:  # idx == 21
       mimo_hidden_states = hidden_states.clone()
   ```
   - Clones AFTER layer 21 processes (output of layer 21)
   - MIMO layers process the cloned hidden states

4. **LM Head Application** (lines 941-942):
   ```python
   text_logits = self.lm_head(hidden_states)  # text path
   audio_logits = self.mimo_output(mimo_hidden_states)  # audio path
   ```
   - Both use the FULL vocabulary size (168448)
   - Tied weights: `lm_head.weight` and `mimo_output.weight` share same weights

5. **Return Order** (line 945):
   ```python
   output = (audio_logits, text_logits) + outputs[2:]
   ```
   - Audio logits FIRST, then text logits

## Current vllm-omni Implementation Issues

### Issue 1: No Separate Token Streams
**Problem**: Our implementation only uses `input_ids`, not separate audio/text token streams.

**Reference**: 
- Passes both `input_ids` (audio) and `text_input_ids` (text)
- Fuses them during embedding

**Our Implementation**:
- Only passes `input_ids` (generated text tokens)
- No mechanism to feed back audio tokens during generation

**Impact**: The model expects to see both audio and text tokens in the input during generation, but we're only providing text tokens. This could cause the model to produce garbage outputs.

### Issue 2: Audio Token Filtering Missing
**Problem**: We're using raw argmax on audio_logits without filtering.

**Reference** (lines 266-276):
```python
generated_wav_tokens = [t for t in generated_wav_tokens if t >= self.kimia_token_offset]
generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset

generated_text_tokens = [t for t in generated_text_tokens if t < self.kimia_token_offset]
```

**Our Implementation**:
```python
audio_token_ids = torch.argmax(audio_logits, dim=-1)
```
- No filtering by `kimia_token_offset`
- No offset subtraction
- Audio detokenizer expects tokens in range [0, 16383] but we're giving it tokens in range [0, 168447]

**Impact**: Audio detokenizer receives wrong token IDs, produces garbage audio.

### Issue 3: Text Token Filtering Missing
**Problem**: Text tokens might include audio tokens (IDs >= 152064).

**Reference**: Filters text tokens to only include IDs < 152064.

**Our Implementation**: vLLM's sampler might select audio tokens as text tokens.

**Impact**: Text output could include audio token IDs, which decode to garbage characters.

### Issue 4: KV Cache for MIMO Layers
**Problem**: MIMO layers are created with `cache_config=None`.

**Question**: Do MIMO layers need KV caching during generation?

**Reference**: Uses HuggingFace's KV cache, which caches both main layers and MIMO layers.

**Our Implementation**: vLLM's KV cache might not include MIMO layers.

**Impact**: During generation, MIMO layers might not have access to cached KV states, causing incorrect outputs.

### Issue 5: Generation Loop Mismatch
**Problem**: vLLM's generation loop is designed for standard autoregressive models, not dual-output models.

**Reference**: Custom generation loop that:
1. Calls forward with both audio and text tokens
2. Samples both audio and text logits
3. Filters tokens by offset
4. Feeds back both token types for next step

**Our Implementation**: vLLM's standard generation loop that:
1. Calls forward with input_ids
2. Samples text logits only
3. Feeds back generated text tokens
4. Audio logits are extracted but not used in generation

**Impact**: The model doesn't see the audio tokens it generated, breaking the autoregressive pattern.

## Root Cause Analysis

The fundamental issue is that **Kimi Audio is NOT a standard autoregressive model**. It's a dual-stream model that:
1. Generates both text and audio tokens
2. Feeds BOTH back as input for the next step
3. Uses a unified vocabulary where audio and text tokens are distinguished by ID ranges

Our vllm-omni implementation treats it as a standard AR model with an auxiliary output, which doesn't match the reference architecture.

## Required Fixes

### Fix 1: Implement Dual Token Stream Feedback
During generation, we need to:
1. Generate both text and audio tokens
2. Filter them by `kimia_token_offset`
3. Feed BOTH back as input for the next step
4. Fuse them during embedding: `inputs_embeds = audio_emb + text_emb`

This requires modifying the generation loop or adding a custom sampler.

### Fix 2: Add Token Filtering
In the stage input processor:
```python
# Filter audio tokens
audio_token_ids = torch.argmax(audio_logits, dim=-1)
audio_token_ids = audio_token_ids[audio_token_ids >= kimia_token_offset]
audio_token_ids = audio_token_ids - kimia_token_offset

# Filter text tokens (in compute_logits or sampler)
text_logits[:, kimia_token_offset:] = -float('inf')  # Mask out audio tokens
```

### Fix 3: Enable KV Cache for MIMO Layers
Pass proper `cache_config` to MIMO layers:
```python
self.mimo_layers = nn.ModuleList([
    Qwen2DecoderLayer(
        config=self.config,
        cache_config=cache_config,  # Pass vLLM's cache config
        quant_config=quant_config,
        prefix=maybe_prefix(prefix, f"mimo_layers.{i}"),
    )
    for i in range(self.config.kimia_mimo_layers)
])
```

### Fix 4: Custom Generation Loop
We may need a custom generation loop that:
1. Maintains separate audio and text token histories
2. Calls forward with both token types
3. Samples and filters both outputs
4. Feeds both back for next step

This could be implemented as a custom sampler or custom worker.

## Verification Steps

1. Check server logs for token ID ranges:
   - Are text tokens in range [0, 152063]?
   - Are audio tokens in range [152064, 168447]?

2. Check audio_logits shape and values:
   - Should be [B, 168448]
   - After argmax, should get token IDs in full range

3. Check what tokens are being fed back during generation:
   - Are we feeding back only text tokens?
   - Should we be feeding back both text and audio tokens?

4. Test with reference implementation:
   - Run reference implementation on same input
   - Compare intermediate hidden states at layer 21
   - Compare audio and text logits

## Next Steps

1. **Immediate**: Add token filtering to stage input processor (Fix 2)
   - Filter audio tokens by offset
   - Subtract offset before sending to detokenizer

2. **Short-term**: Verify MIMO layer KV caching (Fix 3)
   - Pass cache_config to MIMO layers
   - Check if this fixes audio output

3. **Medium-term**: Implement dual token stream feedback (Fix 1, 4)
   - Modify generation loop or add custom sampler
   - This is the most complex fix but may be necessary for correct outputs

4. **Validation**: Compare with reference implementation
   - Run both implementations on same input
   - Compare intermediate states and final outputs
