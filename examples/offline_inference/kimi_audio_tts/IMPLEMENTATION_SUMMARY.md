# Kimi-Audio TTS Integration Summary

## Implementation Complete ✅

Successfully implemented Kimi-Audio TTS integration for vLLM-Omni following the Qwen3-TTS pattern.

---

## Files Created

### 1. Model Implementation (`vllm_omni/model_executor/models/kimi_audio_tts/`)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 17 | Package initialization and exports |
| `configuration_kimi_audio_tts.py` | 160 | Config classes (KimiAudioTalkerConfig, KimiAudioTTSConfig) |
| `kimi_audio_talker.py` | 420 | Stage 0: TTS generation with bifurcation logic |
| `kimi_audio_code2wav.py` | 320 | Stage 1: Audio detokenizer wrapper |
| `audio_detokenizer_loader.py` | 190 | Decoder architectures (HiFi-GAN, Vocos, Generic) |

**Total:** ~1,107 lines of model code

### 2. Stage Configuration (`vllm_omni/model_executor/stage_configs/`)

| File | Lines | Purpose |
|------|-------|---------|
| `kimi_audio_tts.yaml` | 85 | 2-stage pipeline config with async_chunk support |

### 3. Stage Processor (`vllm_omni/model_executor/stage_input_processors/`)

| File | Lines | Purpose |
|------|-------|---------|
| `kimi_audio_tts.py` | 280 | Inter-stage data flow (talker2detokenizer_async_chunk) |

### 4. Registry Update (`vllm_omni/model_executor/models/`)

| File | Change | Purpose |
|------|--------|---------|
| `registry.py` | +8 lines | Registered KimiAudioTalkerForConditionalGeneration, KimiAudioCode2Wav |

### 5. Offline Inference Example (`examples/offline_inference/kimi_audio_tts/`)

| File | Lines | Purpose |
|------|-------|---------|
| `README.md` | 150 | Documentation and usage guide |
| `end2end.py` | 200 | Offline inference script |
| `run_single_prompt.sh` | 35 | Single prompt demo |
| `run_batch_prompts.sh` | 45 | Batch processing demo |

**Total:** ~430 lines of example code

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 0: KimiAudioTalker                                       │
│  Model: KimiAudioTalkerForConditionalGeneration                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  - Embeddings (vocab: 168448)                             │ │
│  │  - Backbone Layers 0-21 (shared)                          │ │
│  │  - Bifurcation @ L21: clone hidden_states                 │ │
│  │  - MIMO Layers 0-5 (audio path)                           │ │
│  │  - mimo_norm + mimo_output                                │ │
│  └───────────────────────────────────────────────────────────┘ │
│  Output: audio_codes [batch, seq_len]                         │
└─────────────────────────────────────────────────────────────────┘
                          │
                          │ SharedMemoryConnector
                          │ codec_chunk_frames: 25
                          │ codec_left_context_frames: 25
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: KimiAudioCode2Wav                                     │
│  Model: KimiAudioCode2Wav                                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  - Load audio_detokenizer/model.pt (19GB)                 │ │
│  │  - Decode audio tokens → 24kHz waveform                   │ │
│  │  - Support chunked decoding with left context             │ │
│  └───────────────────────────────────────────────────────────┘ │
│  Output: waveform [batch, 1, samples] @ 24kHz                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### ✅ Implemented

1. **Bifurcation Logic**
   - Clone hidden states at layer 21
   - Separate audio path through MIMO layers
   - Audio vocabulary: 152064-168447 (16,384 tokens)

2. **2-Stage Pipeline**
   - Stage 0: AR talker (auto-regressive audio token generation)
   - Stage 1: Non-AR detokenizer (direct waveform synthesis)
   - Async chunk streaming support

3. **Stage Connectors**
   - SharedMemoryConnector for inter-stage communication
   - Chunked streaming (25 frames per chunk)
   - Left context overlap for smooth decoding

4. **Audio Detokenizer**
   - Multiple decoder architectures (HiFi-GAN, Vocos, Generic)
   - Lazy loading from checkpoint
   - 24kHz output sample rate

5. **vLLM Integration**
   - Follows Qwen3-TTS patterns
   - Uses vLLM core infrastructure (VllmConfig, layers, utils)
   - OmniOutput for multimodal outputs

---

## Usage

### Quick Start

```bash
cd /root/learning/vllm-omni/examples/offline_inference/kimi_audio_tts

# Single prompt
bash run_single_prompt.sh

# Batch processing
bash run_batch_prompts.sh
```

### Python API

```python
from vllm_omni.engine import OmniEngineCore
from vllm_omni.inputs.data import OmniTokensPrompt

engine = OmniEngineCore(
    model="/data1/moonshotai/Kimi-Audio-7B-Instruct",
    stage_configs_path="vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml",
)

request = OmniTokensPrompt(
    prompt_token_ids=[],
    additional_information={"text": ["Hello world"]},
)

outputs = engine.generate([request])
```

---

## Testing Checklist

- [ ] **Model Loading**: Verify Kimi-Audio weights load correctly
- [ ] **Bifurcation**: Confirm hidden_states clone at layer 21
- [ ] **MIMO Layers**: Test audio token generation
- [ ] **Detokenizer**: Verify audio_detokenizer/model.pt loads
- [ ] **Stage 0 → Stage 1**: Test connector data flow
- [ ] **Streaming**: Verify async_chunk mode works
- [ ] **Audio Quality**: Generate test audio and verify playback
- [ ] **Batch Processing**: Test multiple concurrent requests

---

## Next Steps

1. **Test Model Loading**
   ```bash
   python -c "from vllm_omni.model_executor.models.kimi_audio_tts import KimiAudioTalkerForConditionalGeneration; print('Import OK')"
   ```

2. **Verify Registry**
   ```bash
   python -c "from vllm_omni.model_executor.models.registry import OmniModelRegistry; print(OmniModelRegistry.get_supported_archs())"
   ```

3. **Run End-to-End Test**
   ```bash
   cd examples/offline_inference/kimi_audio_tts
   bash run_single_prompt.sh
   ```

4. **Profile Performance**
   - Measure TTFA (Time to First Audio)
   - Calculate RTF (Real-Time Factor)
   - Test batch throughput

---

## Model Configuration

### Kimi-Audio Constants

```python
# Architecture
NUM_HIDDEN_LAYERS = 28
MIMO_TRANSFORMER_FROM_LAYER_INDEX = 21  # Bifurcation point
MIMO_LAYERS = 6
HIDDEN_SIZE = 3584
VOCAB_SIZE = 168448

# Audio tokens
TEXT_OUTPUT_VOCAB = 152064  # Text: 0-152063
AUDIO_OUTPUT_VOCAB = 16384  # Audio: 152064-168447
AUDIO_TOKEN_OFFSET = 152064

# Special tokens
MEDIA_BEGIN_TOKEN_ID = 151661
MEDIA_END_TOKEN_ID = 151663
TEXT_BLANK_TOKEN_ID = 151666
```

---

## Dependencies

### vLLM Core (imported)
- `vllm.config.VllmConfig`
- `vllm.model_executor.layers.*`
- `vllm.model_executor.models.utils`
- `vllm.distributed`
- `vllm.sequence`

### vLLM-Omni (imported)
- `vllm_omni.model_executor.models.output_templates.OmniOutput`
- `vllm_omni.core.sched.*`
- `vllm_omni.engine.*`

### External
- `torch`
- `transformers`
- `numpy`
- `wave` (stdlib)

---

## Performance Targets

Based on Qwen3-TTS benchmarks:

| Metric | Target (H100) | Target (H20) |
|--------|---------------|--------------|
| TTFA | < 500ms | < 800ms |
| RTF | 0.1 (10x real-time) | 0.2 (5x real-time) |
| Batch Throughput | 10 req/sec | 5 req/sec |
| GPU Memory | 8GB (Stage 0) + 4GB (Stage 1) | 12GB + 6GB |

---

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   FileNotFoundError: audio_detokenizer/model.pt not found
   ```
   **Solution**: Verify model path points to `/data1/moonshotai/Kimi-Audio-7B-Instruct/`

2. **Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce `gpu_memory_utilization` in stage config

3. **No Audio Output**
   ```
   Warning: No audio data in output
   ```
   **Solution**: Check that MIMO layers are loaded (not skipped like ASR)

---

## References

- Qwen3-TTS Implementation: `/root/learning/vllm-omni/vllm_omni/model_executor/models/qwen3_tts/`
- Kimi-Audio ASR: `/root/learning/vllm/vllm/model_executor/models/kimi_audio.py`
- Kimi-Audio Model: `/data1/moonshotai/Kimi-Audio-7B-Instruct/`
- vLLM-Omni Docs: https://vllm-omni.readthedocs.io/

---

**Implementation Date:** 2026-03-12  
**Status:** Ready for Testing 🚀
