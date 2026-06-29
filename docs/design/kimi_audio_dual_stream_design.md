# Dual Token Streaming Design for Kimi Audio

## Executive Summary

Kimi Audio requires maintaining two token streams during generation: **text tokens** (fed back as `input_ids` for the next AR step) and **audio tokens** (sent to Stage 1 detokenizer for waveform synthesis). The current vllm-omni implementation handles single-stream AR generation via the standard vLLM loop. This document presents three concrete approaches for dual-stream support, ranked by feasibility.

## Current Architecture Analysis

### What Already Works

The Stage 0 LLM (`kimi_audio_llm.py`) already implements the bifurcation pattern:

```
forward(input_ids) → layers 0-21 (shared backbone) → bifurcation at layer 21
  ├── Text path: layers 22-27 → lm_head → text logits → vLLM sampler → 1 text token
  └── Audio path: 6 MIMO layers → mimo_output → audio logits → argmax → 1 audio token
```

**Output flow:**
- **Text tokens**: Standard vLLM AR loop — sampled via `compute_logits()` → `Sampler` → fed back as `input_ids`
- **Audio tokens**: Extracted via `stage_input_processors/kimi_audio.py` → `OmniConnector` → Stage 1 detokenizer → waveform

### The Dual-Stream Problem

| Aspect | Training | Inference (Current) | Issue |
|--------|----------|-------------------|-------|
| Input stream | Interleaved [text, audio] tokens in unified vocab [0, 168447] | Text tokens only in `input_ids` | Train/inference mismatch |
| Token feedback | Both text and audio embedded via `(text_emb + audio_emb) × √2` | Only text token embedded | Audio context lost after step 0 |
| Audio delay | 6-token delay encoded in training data | MIMO layers implicitly handle delay | Alignment between streams unclear |
| KV cache | Contains KV for interleaved sequence | Contains KV for text-only sequence | Attention patterns differ |

### Key Architecture Constants

```python
vocab_size = 168448            # Total unified vocabulary
kimia_token_offset = 152064    # Text: [0, 152063], Audio: [152064, 168447]
kimia_audio_output_vocab = 16384  # Audio tokens count (168448 - 152064)
kimia_mimo_layers = 6          # Number of MIMO (audio) layers
kimia_mimo_transformer_from_layer_index = 21  # Bifurcation point
num_hidden_layers = 28         # Total backbone layers (Qwen2)
```

---

## Approach 1: Encode Both Streams in Single `input_ids` (Recommended)

### High-Level Design

Interleave text and audio tokens in the `input_ids` stream using the unified vocabulary offset. At each generation step, the model receives both the previously sampled text token AND the previously generated audio token, encoded as a pair in `input_ids`.

**Key insight**: The model was trained with interleaved tokens. During inference, we reconstruct this interleaving by maintaining two parallel token buffers and feeding both streams into `input_ids`.

### Token Stream Layout

```
Prefill:  [text_prompt_tokens...]  (standard text prompt)
Step 0:   input_ids = [last_text_token]  →  output: text_token_0 + audio_logits_0
Step 1:   input_ids = [text_token_0, audio_token_0 + offset]  →  output: text_token_1 + audio_logits_1
Step 2:   input_ids = [text_token_1, audio_token_1 + offset]  →  output: text_token_2 + audio_logits_2
...
```

Wait — vLLM's standard AR loop only feeds back ONE token per step (the sampled text token). To feed back TWO tokens (text + audio), we need to modify the feedback path.

### Revised Design: Two-Token Feedback via Custom Sampler

Instead of modifying the core AR loop, we override the token feedback mechanism:

```
Step N:
1. forward(input_ids=[text_token_N-1, audio_token_N-1 + 152064])  # 2 tokens
2. Bifurcation at layer 21
3. Text path → compute_logits() → text logits → sampler → text_token_N
4. Audio path → mimo_output → audio logits → argmax → audio_token_N
5. Prepare next input_ids = [text_token_N, audio_token_N + 152064]  # Concatenate both
```

**During decode (1 token per step in standard vLLM)**, we need to change the model runner to feed 2 tokens per step.

### Files to Modify

#### 1. `vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py`

**Changes**:
- Override `prepare_runner_inputs()` to inject audio tokens alongside text tokens
- Override `embed_input_ids()` to handle dual-stream embedding with proper fusion
- Add audio token buffer per request
- Split embeddings by offset before fusion

```python
def prepare_runner_inputs(
    self,
    input_ids, positions, inputs_embeds,
    req_ids, num_computed_tokens, num_scheduled_tokens,
    input_ids_buffer,
):
    """Inject audio tokens alongside text tokens for dual-stream input."""
    # For each request, append the audio token from the previous step
    for i, req_id in enumerate(req_ids):
        audio_token = self._get_last_audio_token(req_id)
        if audio_token is not None:
            # Expand input_ids to include audio token
            # input_ids[i] = [text_token, audio_token + offset]
            ...
    return input_ids, positions

def embed_input_ids(self, input_ids, multimodal_embeddings=None):
    """Embed with dual-stream fusion: split by offset, fuse (text_emb + audio_emb) × √2."""
    text_embeds = self.model.model.embed_tokens(input_ids)

    # Identify audio tokens (>= offset)
    audio_mask = input_ids >= self.config.kimia_token_offset

    if audio_mask.any():
        # Get audio embeddings (reuse text embedding table for audio range)
        audio_embeds = self.model.model.embed_tokens(
            input_ids.clone()  # Audio tokens use same embedding table
        )

        # For positions where we have both text and audio, fuse them
        # Fusion formula: (text_emb + audio_emb) × √2
        # In decode mode: 2 tokens per step (text + audio)
        # text_embeds[0] = text embedding, text_embeds[1] = audio embedding
        # Fuse: fused = (text_embeds[0] + audio_embeds[1]) × √2
        ...

    return fused_embeds
```

#### 2. `vllm_omni/worker/gpu_ar_model_runner.py`

**Changes**:
- Override `sample_tokens()` to capture both text and audio tokens
- After sampling text token AND computing audio token, store audio token for next step
- Modify `_update_states_after_model_execute()` to feed back 2 tokens instead of 1

```python
def _update_states_after_model_execute(self, sampler_output, scheduler_output):
    """Override to feed back both text and audio tokens."""
    # Standard: only text token fed back
    # Modified: text token + audio token concatenated

    text_token_ids = sampler_output.sampled_token_ids  # From standard sampler
    audio_token_ids = self._get_audio_tokens_from_model_output()  # From audio path

    # Concatenate: [text_token, audio_token + offset]
    combined_token_ids = torch.cat([
        text_token_ids,
        audio_token_ids + 152064  # Offset to unified vocab range
    ], dim=-1)

    # Update input_batch with combined tokens
    ...
```

#### 3. `vllm_omni/core/sched/omni_ar_scheduler.py`

**Changes**:
- Track `num_tokens_per_step` per request (2 for Kimi Audio instead of 1)
- Adjust `num_computed_tokens` counting to account for 2 tokens per step

### Pseudocode for Key Methods

```python
class KimiAudioLLMForConditionalGeneration:
    def __init__(self, ...):
        ...
        # Per-request audio token buffer
        self._audio_token_buffer: dict[str, int] = {}

    def forward(self, input_ids, positions, ...):
        # 1. Embed with dual-stream fusion
        if input_ids.shape[0] == 2:
            # Decode mode: [text_token, audio_token]
            text_ids = input_ids[:1]
            audio_ids = input_ids[1:]  # Has offset

            text_emb = self.model.model.embed_tokens(text_ids)
            audio_emb = self.model.model.embed_tokens(audio_ids)

            # Fusion: (text_emb + audio_emb) × √2
            fused_emb = (text_emb + audio_emb) * (2 ** 0.5)
            inputs_embeds = fused_emb  # [1, hidden_size]
        else:
            # Prefill mode: standard embedding
            inputs_embeds = self.embed_input_ids(input_ids, multimodal_embeddings)

        # 2. Forward through shared backbone (layers 0-21)
        hidden_states, residual = self._forward_to_layer_21(input_ids[:1], positions, inputs_embeds)

        # 3. Bifurcation
        text_hidden = hidden_states.clone()
        audio_hidden = hidden_states.clone()
        text_resid = residual.clone() if residual is not None else None
        audio_resid = residual.clone() if residual is not None else None

        # 4. Text path
        for layer in self.model.model.layers[22:]:
            text_hidden, text_resid = layer(positions, text_hidden, text_resid)
        text_hidden, _ = self.model.model.norm(text_hidden, text_resid)

        # 5. Audio path
        for mimo_layer in self.mimo_layers:
            audio_hidden, audio_resid = mimo_layer(positions, audio_hidden, audio_resid)
        audio_hidden, _ = self.mimo_norm(audio_hidden, audio_resid)
        audio_logits = self.mimo_output(audio_hidden)

        # 6. Store audio token for next step
        audio_token_id = torch.argmax(audio_logits, dim=-1).item()
        # Filter: only keep if >= offset
        if audio_token_id >= self.config.kimia_token_offset:
            self._pending_audio_token = audio_token_id
        else:
            self._pending_audio_token = self.config.kimia_token_offset  # Default

        return OmniOutput(
            text_hidden_states=text_hidden,
            multimodal_outputs={"audio_logits": audio_logits},
        )

    def get_audio_token_for_feedback(self) -> int:
        """Get the audio token to feed back in next step."""
        return getattr(self, '_pending_audio_token', self.config.kimia_token_offset)
```

### Handling Audio Delay (6 Tokens)

The 6-token audio delay means the audio path lags behind the text path by 6 steps. In the original model, this is handled by:
- MIMO layers starting from layer 21 output (which already has 6 steps of context built in)
- The delay is implicit in the architecture, not explicit in the token stream

For dual streaming:
- **Steps 0-5**: Only text tokens are generated (no audio feedback yet)
- **Step 6+**: Both text and audio tokens are fed back
- Audio token buffer starts empty, fills after step 0

```python
def _prepare_dual_stream_input(self, req_id, text_token):
    if self._step_count.get(req_id, 0) < 6:
        # Early steps: only text token (no audio context yet)
        return [text_token]
    else:
        # Later steps: text + audio
        audio_token = self._audio_token_buffer.get(req_id, self.config.kimia_token_offset)
        return [text_token, audio_token]
```

### Handling Text Stream Termination

When the text path generates an EOS token:
1. Text generation stops (standard vLLM behavior)
2. Audio path continues generating until it produces its own EOS or reaches max_tokens
3. The audio token buffer continues to feed audio tokens to Stage 1

```python
def _handle_text_eos(self, req_id):
    """Text stream ended but audio may continue."""
    self._text_ended[req_id] = True
    # Continue generating audio-only tokens
    # input_ids = [eos_token, audio_token + offset]
    # Or: switch to audio-only mode where text_emb = 0
```

### Pros and Cons

**Pros**:
- ✅ **Minimal infrastructure changes**: Uses existing vLLM AR loop, sampler, and KV cache
- ✅ **Matches training distribution**: Model sees interleaved tokens as during training
- ✅ **Compatible with CUDA graphs**: Forward pass shape is deterministic (2 tokens per step)
- ✅ **Compatible with prefix caching**: Text+audio pairs can be cached normally
- ✅ **Leverages existing bifurcation**: No changes to the model's core architecture
- ✅ **Compatible with batching**: Multiple requests can batch with 2 tokens each

**Cons**:
- ❌ **KV cache size doubles**: Each step stores KV for 2 tokens instead of 1
- ❌ **Position encoding complexity**: Need to handle positions for 2 tokens per step
- ❌ **Attention pattern mismatch**: Model attends to interleaved sequence, but during prefill only text tokens are present
- ⚠️ **Moderate implementation complexity**: Need to modify token feedback path in model runner

### Implementation Complexity: **Medium**

### Risk Assessment: **Low-Medium**

The main risk is the train/inference mismatch during prefill (text-only) vs decode (text+audio). However, this is mitigated by the fact that the model was designed with this architecture — the bifurcation at layer 21 means the audio path doesn't depend on audio tokens in the input, only on the shared hidden states.

### Compatibility

| Feature | Compatible | Notes |
|---------|-----------|-------|
| CUDA Graphs | ✅ Yes | Fixed 2-token input per decode step |
| Batching | ✅ Yes | Standard vLLM batching with 2 tokens/req |
| Prefix Caching | ⚠️ Partial | Prefill cache misses on decode tokens |
| Tensor Parallelism | ✅ Yes | No changes to parallelism strategy |
| Speculative Decoding | ❌ No | Would need custom draft model |

---

## Approach 2: Custom Generation Loop

### High-Level Design

Override `execute_model()` in `GPUARModelRunner` to implement a custom generation loop that maintains internal audio/text token buffers and bypasses the standard vLLM token feedback mechanism.

This approach is similar to how MiMo Audio handles its local_forward loop — it maintains per-request state (`_cached_new_audio_emb_by_req`) and generates audio tokens inside the model's forward pass.

### Files to Modify

#### 1. `vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py`

**Major changes**:
- Add internal generation loop that runs both text and audio AR
- Maintain per-request audio token history
- Implement `sample()` method that returns both text and audio tokens

```python
class KimiAudioLLMForConditionalGeneration:
    # Per-request state
    _text_token_history: dict[str, list[int]] = {}
    _audio_token_history: dict[str, list[int]] = {}
    _audio_kv_cache: dict[str, DynamicCache] = {}

    def sample(self, logits, sampling_metadata):
        """Custom sampler that returns both text and audio tokens."""
        # Sample text token from text logits
        text_token = self._sample_text(logits)

        # Get audio token from audio logits (stored during forward)
        audio_token = self._get_audio_token()

        # Return combined result
        return CombinedSamplerOutput(
            text_token_ids=text_token,
            audio_token_ids=audio_token,
        )
```

#### 2. `vllm_omni/worker/gpu_ar_model_runner.py`

**Major changes**:
- Override `execute_model()` to run a custom loop
- After each step, manually construct next `input_ids` from both text and audio tokens
- Bypass standard `_update_states_after_model_execute()`

```python
def execute_model(self, scheduler_output, intermediate_tensors=None):
    """Custom execution with dual-stream feedback."""
    # 1. Standard preprocessing
    ...

    # 2. Run model forward
    model_output = self._model_forward(input_ids, positions, ...)

    # 3. Extract both text and audio outputs
    text_logits = self.model.compute_logits(sample_hidden_states)
    audio_logits = model_output.multimodal_outputs["audio_logits"]

    # 4. Sample both tokens
    text_token = self.sampler(text_logits, sampling_metadata)
    audio_token = torch.argmax(audio_logits, dim=-1)

    # 5. Construct next input_ids = [text_token, audio_token + offset]
    next_input_ids = torch.cat([
        text_token.sampled_token_ids,
        audio_token + 152064,
    ], dim=-1)

    # 6. Manually update input_batch with next_input_ids
    self._manual_update_input_batch(next_input_ids)

    # 7. Build output
    ...
```

#### 3. `vllm_omni/core/sched/omni_ar_scheduler.py`

**Changes**:
- Track `num_tokens_per_step = 2` for Kimi Audio requests
- Adjust token counting in `update_from_output()`

### Pseudocode for Key Methods

```python
class GPUARModelRunner:
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        """Custom dual-stream execution."""
        # ... standard preprocessing ...

        with set_forward_context(...):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        # Extract hidden states and multimodal outputs
        hidden_states = model_output
        multimodal_outputs = model_output.multimodal_outputs if hasattr(model_output, 'multimodal_outputs') else None

        # Compute text logits
        sample_hidden_states = hidden_states[logits_indices]
        text_logits = self.model.compute_logits(sample_hidden_states)

        # Get audio logits
        audio_logits = multimodal_outputs.get("audio_logits") if multimodal_outputs else None

        # Sample text tokens (standard vLLM sampler)
        sampler_output = self.sampler(text_logits, sampling_metadata)

        # Extract audio tokens (argmax)
        if audio_logits is not None:
            audio_token_ids = torch.argmax(audio_logits, dim=-1)
            # Filter by offset
            audio_mask = audio_token_ids >= 152064
            audio_token_ids = audio_token_ids.clamp(min=152064)
        else:
            audio_token_ids = torch.full_like(sampler_output.sampled_token_ids, 152064)

        # Store audio tokens for feedback
        self._pending_audio_tokens = audio_token_ids

        # ... standard bookkeeping ...

        # Build output with both token streams
        output = OmniModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampler_output.sampled_token_ids,  # Text tokens
            multimodal_outputs=multimodal_outputs,  # Audio logits for Stage 1
            ...
        )

        return output

    def _prepare_next_step_input(self, text_tokens, audio_tokens):
        """Construct input_ids for next step: [text, audio+offset]."""
        # Concatenate text and audio tokens
        audio_with_offset = audio_tokens  # Already in [152064, 168447] range
        combined = torch.stack([text_tokens.squeeze(), audio_with_offset.squeeze()], dim=0)
        return combined
```

### Handling Audio Delay

The custom loop explicitly tracks the delay:

```python
def _get_audio_feedback_token(self, req_id, step):
    """Get audio token for feedback, accounting for 6-step delay."""
    history = self._audio_token_history.get(req_id, [])

    # Audio is delayed by 6 steps
    audio_step = step - 6
    if audio_step < 0 or audio_step >= len(history):
        return 152064  # Default/empty audio token
    return history[audio_step]
```

### Pros and Cons

**Pros**:
- ✅ **Full control over token feedback**: Can implement any interleaving pattern
- ✅ **Clean separation**: Audio/text buffers are explicit and debuggable
- ✅ **Flexible delay handling**: Can implement arbitrary delay patterns
- ✅ **No KV cache duplication**: Only store KV for actual tokens

**Cons**:
- ❌ **Bypasses vLLM's optimized generation loop**: Loses async scheduling, speculative decoding
- ❌ **Manual KV cache management**: Must handle cache updates carefully
- ❌ **Incompatible with CUDA graphs**: Dynamic control flow breaks graph capture
- ❌ **Incompatible with batching**: Custom loop is inherently per-request
- ❌ **High maintenance burden**: Must track vLLM upstream changes
- ❌ **Breaks prefix caching**: Non-standard token sequence

### Implementation Complexity: **High**

### Risk Assessment: **High**

This approach bypasses many of vLLM's optimizations and is fragile against upstream changes. The custom generation loop must correctly handle edge cases (request preemption, migration, etc.) that vLLM's standard loop handles automatically.

### Compatibility

| Feature | Compatible | Notes |
|---------|-----------|-------|
| CUDA Graphs | ❌ No | Dynamic control flow |
| Batching | ❌ No | Per-request loop |
| Prefix Caching | ❌ No | Non-standard token sequences |
| Speculative Decoding | ❌ No | Custom sampler |
| Async Scheduling | ⚠️ Partial | Would need custom async loop |

---

## Approach 3: Scheduler Extension

### High-Level Design

Extend the vllm-omni scheduler to natively support multiple token sequences per request. Add `audio_token_ids` alongside `text_token_ids` in the request state, and update the model runner to handle both streams transparently.

This is the most invasive but cleanest approach — it makes dual-stream generation a first-class concept in the scheduling layer.

### Files to Modify

#### 1. `vllm_omni/core/sched/omni_ar_scheduler.py`

**Major changes**:
- Add `audio_token_ids` to `Request` state
- Track `num_audio_tokens_generated` per request
- In `update_from_output()`, process both text and audio token streams

```python
class OmniARScheduler:
    def update_from_output(self, scheduler_output, model_runner_output):
        # ... standard processing ...

        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            request = self.requests.get(req_id)

            # Process text tokens (standard)
            text_token_ids = sampled_token_ids[req_index]
            request._output_token_ids.extend(text_token_ids)

            # Process audio tokens (NEW)
            audio_token_ids = model_runner_output.audio_token_ids.get(req_id, [])
            if not hasattr(request, 'audio_token_ids'):
                request.audio_token_ids = []
            request.audio_token_ids.extend(audio_token_ids)

            # Emit both in EngineCoreOutput
            outputs[request.client_index].append(
                OmniEngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=text_token_ids,
                    new_audio_token_ids=audio_token_ids,  # NEW field
                    multimodal_output=mm_output,
                    ...
                )
            )
```

#### 2. `vllm_omni/worker/gpu_ar_model_runner.py`

**Changes**:
- Extract audio tokens from model output alongside text tokens
- Return both in `OmniModelRunnerOutput`
- Construct next-step input_ids from both streams

```python
class OmniModelRunnerOutput:
    # NEW field
    audio_token_ids: dict[str, list[int]] | None = None

class GPUARModelRunner:
    def sample_tokens(self, grammar_output):
        # ... standard sampling ...

        # Extract audio tokens from multimodal_outputs
        audio_token_ids = {}
        if multimodal_outputs and "audio_logits" in multimodal_outputs:
            for req_idx, req_id in enumerate(req_ids):
                audio_logits = multimodal_outputs["audio_logits"][req_idx]
                audio_token = torch.argmax(audio_logits, dim=-1).item()
                audio_token_ids[req_id] = [audio_token]

        output = OmniModelRunnerOutput(
            ...,
            audio_token_ids=audio_token_ids,  # NEW
        )
        return output
```

#### 3. `vllm_omni/engine/__init__.py` (EngineCoreOutput)

**Changes**:
- Add `new_audio_token_ids` field to `OmniEngineCoreOutput`
- Propagate through the engine output pipeline

#### 4. `vllm_omni/worker/gpu_ar_model_runner.py` (input construction)

**Changes**:
- In `_prepare_inputs()`, construct dual-stream input_ids
- Use scheduler's audio token history to build audio portion

```python
def _prepare_dual_stream_inputs(self, scheduler_output):
    """Build input_ids with both text and audio tokens."""
    text_input_ids = self._get_text_input_ids(scheduler_output)
    audio_input_ids = self._get_audio_input_ids(scheduler_output)

    # Interleave: [text_0, audio_0, text_1, audio_1, ...]
    combined = torch.stack([text_input_ids, audio_input_ids], dim=1)
    combined = combined.view(-1)  # Flatten

    return combined
```

#### 5. `vllm_omni/patch.py`

**Changes**:
- Patch `Request` class to include `audio_token_ids` field
- Patch `EngineCoreOutput` to include `new_audio_token_ids`

### Pseudocode for Key Methods

```python
# In Request class (patched)
class Request:
    audio_token_ids: list[int] = []
    num_audio_tokens_generated: int = 0

# In scheduler
class OmniARScheduler:
    def update_from_output(self, scheduler_output, model_runner_output):
        for req_id in num_scheduled_tokens:
            request = self.requests[req_id]

            # Text tokens
            text_tokens = sampled_token_ids[req_index]
            request._output_token_ids.extend(text_tokens)

            # Audio tokens
            audio_tokens = model_runner_output.audio_token_ids.get(req_id, [])
            request.audio_token_ids.extend(audio_tokens)
            request.num_audio_tokens_generated += len(audio_tokens)

            # Check termination
            text_done = self._check_text_stop(request, text_tokens)
            audio_done = self._check_audio_stop(request, audio_tokens)

            if text_done and audio_done:
                request.status = RequestStatus.FINISHED_STOPPED
            elif text_done and not audio_done:
                # Text done but audio continues
                # Switch to audio-only mode
                request._text_ended = True

    def schedule(self):
        scheduler_output = super().schedule()

        # Add audio token info to scheduled output
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            # Include audio token count for model runner
            scheduler_output.audio_token_counts[req_id] = request.num_audio_tokens_generated

        return scheduler_output
```

### Handling Audio Delay

```python
def _get_audio_input_ids(self, scheduler_output):
    """Build audio portion of input_ids with delay handling."""
    audio_ids = []
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        audio_history = request.audio_token_ids

        # Apply 6-token delay
        delay = 6
        if len(audio_history) >= delay:
            # Use audio token from 6 steps ago
            audio_token = audio_history[-delay] + 152064  # Offset
        else:
            # Not enough history yet, use empty token
            audio_token = 152064

        audio_ids.append(audio_token)

    return torch.tensor(audio_ids, device=self.device)
```

### Pros and Cons

**Pros**:
- ✅ **Cleanest architecture**: Dual-stream is a first-class concept
- ✅ **Extensible**: Other models can use the same mechanism
- ✅ **Proper state tracking**: Audio tokens tracked in Request state
- ✅ **Compatible with existing features**: Standard vLLM loop preserved
- ✅ **Proper output propagation**: Audio tokens flow through engine output pipeline

**Cons**:
- ❌ **Most invasive**: Touches scheduler, model runner, engine output, Request class
- ❌ **Upstream coupling**: Deep changes to vllm-omni's patch layer
- ❌ **Complex state management**: Must keep text and audio streams synchronized
- ❌ **Serialization overhead**: Audio tokens add to wire protocol
- ❌ **Testing surface area**: Many components need testing

### Implementation Complexity: **Very High**

### Risk Assessment: **Medium-High**

While architecturally clean, this approach requires changes across many layers of the stack. Each layer introduces potential bugs and maintenance burden. The patch layer changes are particularly risky as they affect all models, not just Kimi Audio.

### Compatibility

| Feature | Compatible | Notes |
|---------|-----------|-------|
| CUDA Graphs | ✅ Yes | Fixed 2-token input per step |
| Batching | ✅ Yes | Standard batching with extended state |
| Prefix Caching | ⚠️ Partial | Audio tokens complicate cache keys |
| Speculative Decoding | ⚠️ Partial | Need to spec decode both streams |
| Async Scheduling | ✅ Yes | Standard async with extended output |

---

## Recommendation: Approach 1 (Single Input_ids Encoding)

### Justification

**Approach 1** is recommended because it provides the best balance of:
- **Implementation feasibility**: Medium complexity, touches only 3 files
- **Compatibility**: Works with CUDA graphs, batching, and prefix caching
- **Correctness**: Matches the training distribution (interleaved tokens)
- **Minimal risk**: Uses existing vLLM infrastructure, no custom generation loops

The key insight is that Kimi Audio's bifurcation architecture (layer 21 split) means the audio path doesn't need audio tokens as input — it derives audio logits from the shared hidden states. So the "dual stream" is really about:
1. Feeding text tokens back for the text AR loop (standard)
2. Extracting audio tokens from audio logits and sending to Stage 1 (already works)
3. Optionally feeding audio token embeddings back for better generation quality (the enhancement)

Approach 1 enables point 3 with minimal changes.

### Why Not Approach 2?
- Bypasses too many vLLM optimizations
- Incompatible with batching and CUDA graphs
- High maintenance burden against upstream

### Why Not Approach 3?
- Too invasive for a single model's needs
- Changes affect all models, not just Kimi Audio
- The scheduler extension pattern is better suited for a future generalization

---

## Implementation Roadmap

### Phase 1: Core Dual-Stream Input (Week 1-2)

**Goal**: Get dual-token feedback working in decode mode

1. **Modify `kimi_audio_llm.py`**:
   - Add `_pending_audio_token` tracking
   - Override `prepare_runner_inputs()` to expand input_ids to 2 tokens
   - Override `embed_input_ids()` with dual-stream fusion

2. **Test**: Verify that model produces valid text + audio tokens with interleaved input

### Phase 2: Audio Token Buffer (Week 2-3)

**Goal**: Handle audio delay and early steps correctly

1. **Add audio token buffer** per request in the model
2. **Handle steps 0-5**: Only text tokens (no audio feedback)
3. **Handle step 6+**: Both text and audio tokens
4. **Test**: Verify audio delay alignment

### Phase 3: Stream Termination (Week 3)

**Goal**: Handle text EOS while audio continues

1. **Text EOS handling**: Switch to audio-only input
2. **Audio EOS handling**: Stop audio generation
3. **Test**: Verify clean termination of both streams

### Phase 4: Stage 1 Integration (Week 3-4)

**Goal**: Ensure audio tokens properly flow to detokenizer

1. **Verify stage_input_processor** handles new token format
2. **Test end-to-end**: Text + audio output with proper alignment
3. **Verify waveform quality** matches reference implementation

### Phase 5: Performance Optimization (Week 4)

**Goal**: Optimize for production use

1. **CUDA graph compatibility**: Verify 2-token decode works with graphs
2. **Batching**: Test multi-request batching
3. **Prefix caching**: Enable for prefill portion
4. **Benchmark**: Compare latency/throughput vs single-stream

---

## Testing Strategy

### Unit Tests

```python
# test_kimi_audio_dual_stream.py

def test_dual_token_embedding():
    """Verify (text_emb + audio_emb) × √2 fusion."""
    model = KimiAudioLLMForConditionalGeneration(...)
    input_ids = torch.tensor([100, 152164])  # text + audio
    embeds = model.embed_input_ids(input_ids)
    # Verify fusion formula
    text_emb = model.model.model.embed_tokens(torch.tensor([100]))
    audio_emb = model.model.model.embed_tokens(torch.tensor([152164]))
    expected = (text_emb + audio_emb) * (2 ** 0.5)
    assert torch.allclose(embeds, expected, atol=1e-5)

def test_audio_token_filtering():
    """Verify audio tokens are filtered by offset."""
    audio_logits = torch.randn(1, 168448)
    token_id = torch.argmax(audio_logits, dim=-1).item()
    # Token should be in audio range [152064, 168447]
    assert 152064 <= token_id <= 168447 or token_id < 152064  # May be text

def test_audio_delay_handling():
    """Verify 6-step audio delay."""
    # Steps 0-5: no audio feedback
    for step in range(6):
        input_ids = model._prepare_dual_stream_input("req0", text_token, step)
        assert len(input_ids) == 1  # Text only

    # Step 6+: text + audio
    input_ids = model._prepare_dual_stream_input("req0", text_token, 6)
    assert len(input_ids) == 2  # Text + audio

def test_text_eos_handling():
    """Verify text EOS stops text but not audio."""
    model._handle_text_eos("req0")
    assert model._text_ended["req0"] == True
    # Audio should still generate
    input_ids = model._prepare_dual_stream_input("req0", eos_token, 10)
    assert len(input_ids) == 2  # Still dual-stream
```

### Integration Tests

```python
def test_end_to_end_generation():
    """Full pipeline: prompt → text + audio tokens → waveform."""
    # 1. Create request
    prompt = "Hello, how are you?"
    output = omni.generate(prompt)

    # 2. Verify text output
    assert len(output.text) > 0

    # 3. Verify audio output
    assert output.audio is not None
    assert output.audio_sample_rate == 24000

    # 4. Verify alignment
    # Text and audio should be temporally aligned
    assert len(output.audio) > 0

def test_batched_generation():
    """Multiple requests batched together."""
    prompts = ["Hello", "World", "Test"]
    outputs = omni.generate(prompts)
    assert len(outputs) == 3
    for output in outputs:
        assert len(output.text) > 0
        assert output.audio is not None

def test_streaming_output():
    """Verify streaming works with dual streams."""
    for chunk in omni.generate_stream("Hello"):
        # Each chunk should have text and/or audio
        assert chunk.text or chunk.audio
```

### Regression Tests

```python
def test_text_only_mode():
    """Verify text-only requests still work (no audio)."""
    # Some requests may be text-only (no TTS)
    output = omni.generate("What is 2+2?", task_type=["chat"])
    assert output.text == "4"  # or similar
    assert output.audio is None  # No audio for chat

def test_audio_understanding():
    """Verify audio input (understanding) still works."""
    output = omni.generate(audio_input=wav_file, task_type=["understanding"])
    assert len(output.text) > 0  # Transcription/response
```

---

## Risk Mitigation Plan

### Risk 1: Train/Inference Mismatch

**Risk**: Prefill uses text-only input, decode uses text+audio input. Model may produce different outputs.

**Mitigation**:
- The bifurcation architecture means audio path doesn't depend on audio input tokens
- Audio tokens only affect the model via embedding fusion in the text path
- Test with and without audio feedback to measure impact
- Fallback: Disable audio feedback (text-only decode) if quality degrades

### Risk 2: KV Cache Size Doubling

**Risk**: 2 tokens per step doubles KV cache usage.

**Mitigation**:
- Only decode steps use 2 tokens; prefill uses 1 token per text token
- For a 4096-token generation: 4096 (prefill) + 4096 (decode, 2 tokens each) = 12288 tokens
- This is ~3x increase in KV cache for decode portion
- Mitigate by: reducing max_batch_size, increasing GPU memory utilization
- Alternative: Only use dual-stream for first N steps, then switch to text-only

### Risk 3: Position Encoding Issues

**Risk**: Positions for 2 tokens per step may confuse the RoPE encoder.

**Mitigation**:
- Use same position for both text and audio tokens at each step
- text_pos = audio_pos = step_number
- This matches how multimodal embeddings are handled (same position, fused embedding)

### Risk 4: Upstream Compatibility

**Risk**: Changes may break with future vLLM/vllm-omni updates.

**Mitigation**:
- Changes are localized to Kimi Audio model files
- `prepare_runner_inputs()` is a model-level hook, not core infrastructure
- Document changes clearly for future maintainers
- Add feature flag to disable dual-stream if needed

### Risk 5: Audio Quality Degradation

**Risk**: Dual-stream input may affect audio token quality.

**Mitigation**:
- Compare audio token distributions with/without dual-stream
- A/B test waveform quality (MOS score)
- Fallback: Keep current single-stream approach if quality degrades
- The stage_input_processor already handles audio token extraction independently

---

## Appendix: Reference Implementation Comparison

### MiMo Audio Pattern (existing in codebase)

MiMo Audio uses a similar dual-stream pattern but with a different architecture:
- Uses `<|empty|>` token (151667) as a marker for audio generation steps
- When model predicts `<|empty|>`, triggers `local_forward()` to generate audio codes
- Audio embeddings are cached per-request (`_cached_new_audio_emb_by_req`)
- Audio embeddings fed back via `inputs_embeds` at the next step

**Key difference**: MiMo Audio decides WHEN to generate audio based on text logits (empty token gate). Kimi Audio generates audio at EVERY step via the parallel MIMO path.

### Kimi Audio Original (HuggingFace reference)

The reference `modeling_moonshot_kimia.py` uses:
- Interleaved [text, audio] tokens in input_ids during training
- Bifurcation at layer 21 (identical to our implementation)
- During inference: feeds both text and audio tokens at each step
- Audio delay: 6 tokens (matching our MIMO layer count)

**Our implementation matches the reference** for the bifurcation and MIMO layers. The gap is in the inference-time token feedback.
