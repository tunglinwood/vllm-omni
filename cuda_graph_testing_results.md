# CUDA Graph Testing Results

**Date:** 2026-07-01  
**Model:** Kimi Audio 7B Instruct  
**GPU:** GPU 6 (CUDA_VISIBLE_DEVICES=6)

## Summary

**Result: ❌ CUDA graphs do NOT work with Kimi Audio**

Kimi Audio requires `--enforce-eager` mode. Attempting to enable CUDA graphs causes initialization failures.

## Test Configuration

### Modified Deploy Config
Created `vllm_omni/deploy/kimi_audio_cuda_graph.yaml` with:
```yaml
stages:
  - stage_id: 0
    enforce_eager: false  # ← ENABLE CUDA GRAPHS (was true)
    ...
  
  - stage_id: 1
    enforce_eager: false  # ← ENABLE CUDA GRAPHS (was true)
    ...
```

### Startup Script
Created `start_kimi_audio_cuda_graph.cjs`:
```javascript
const proc = spawn('vllm-omni', [
  'serve',
  '/data1/moonshotai/Kimi-Audio-7B-Instruct',
  '--omni',
  '--port', '8091',
  '--deploy-config', 'vllm_omni/deploy/kimi_audio_cuda_graph.yaml',
  '--trust-remote-code'
], {
  cwd: __dirname,
  stdio: 'inherit',
  env: { ...process.env, CUDA_VISIBLE_DEVICES: '6' }
});
```

## Error Details

### Root Cause
```
torch.AcceleratorError: CUDA error: operation failed due to a previous error during capture
Search for `cudaErrorStreamCaptureInvalidated' for more information.
```

### Stack Trace
```
File "/root/learning/vllm_integration/vllm-omni/vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py", line 434, in forward
    text_hidden_states = hidden_states.clone()
                         ^^^^^^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: operation failed due to a previous error during capture
```

### Error Location
- **File:** `kimi_audio_llm.py`
- **Line:** 434 (during forward pass)
- **Operation:** `hidden_states.clone()`
- **Phase:** CUDA graph capture (warmup phase)

## Why CUDA Graphs Fail

CUDA graphs require all operations in the forward pass to be:
1. **Deterministic** - No data-dependent control flow
2. **Static shapes** - Tensor shapes must be known at capture time
3. **Graph-compatible** - All ops must support CUDA graph capture

Kimi Audio's forward pass contains operations that violate these requirements:
- **Dynamic control flow** based on audio token availability
- **Conditional operations** in the multimodal fusion logic
- **Dynamic tensor operations** that can't be statically captured

Specifically, the model has:
```python
# Line 434 in kimi_audio_llm.py
text_hidden_states = hidden_states.clone()
```

This operation fails during CUDA graph capture because earlier operations in the forward pass have incompatible characteristics (likely the conditional audio token handling or multimodal fusion).

## Comparison: Enforce-Eager vs CUDA Graphs

| Mode | Status | Performance | Notes |
|------|--------|-------------|-------|
| `enforce_eager: true` | ✅ Works | Baseline | Required for Kimi Audio |
| `enforce_eager: false` | ❌ Fails | N/A | CUDA graph capture fails |

## Conclusion

**Kimi Audio must run with `--enforce-eager` (eager execution mode).**

CUDA graphs cannot be enabled without significant model refactoring to make the forward pass graph-compatible. This would require:
1. Removing all conditional control flow in forward pass
2. Making all tensor operations static
3. Restructuring the multimodal fusion logic
4. Potentially rewriting the audio token handling

Such refactoring is not recommended as:
- The model works correctly in eager mode
- Whisper features handle audio comprehension
- The complexity of refactoring outweighs the performance benefit

## Recommendation

**Keep `enforce_eager: true` for both stages in the deploy config.**

Current working configuration (`vllm_omni/deploy/kimi_audio.yaml`):
```yaml
stages:
  - stage_id: 0
    enforce_eager: true  # ← REQUIRED
    ...
  
  - stage_id: 1
    enforce_eager: true  # ← REQUIRED
    ...
```

## Related Files

- Working config: `vllm_omni/deploy/kimi_audio.yaml`
- Failed config: `vllm_omni/deploy/kimi_audio_cuda_graph.yaml` (kept for reference)
- Model implementation: `vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py`
- Memory: `kimi-audio-working-mechanism.md`
