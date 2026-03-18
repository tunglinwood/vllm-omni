# Kimi-Audio TTS Offline Inference

This directory contains an offline demo for running Kimi-Audio TTS with vLLM-Omni.

## Model Overview

Kimi-Audio-7B-Instruct is a unified speech model that supports:
- **ASR (Automatic Speech Recognition)**: Speech-to-text
- **TTS (Text-to-Speech)**: Text-to-speech synthesis

This demo focuses on the **TTS** capability, generating speech from text input.

## Architecture

Kimi-Audio TTS uses a 2-stage pipeline:
1. **Stage 0 (Talker)**: Text → Audio tokens
   - Shared backbone (layers 0-21)
   - Bifurcation at layer 21
   - MIMO layers (0-5) for audio generation
2. **Stage 1 (Code2Wav)**: Audio tokens → Waveform
   - Audio detokenizer (vocoder)
   - Outputs 24kHz audio

## Setup

### Prerequisites

1. Install vLLM-Omni dependencies:
```bash
cd /root/learning/vllm-omni
pip install -e .
```

2. Ensure Kimi-Audio model is available:
```bash
# Model should be at:
/data1/moonshotai/Kimi-Audio-7B-Instruct/
```

3. Configure stage config for your hardware:
```bash
# Edit stage_configs/kimi_audio_tts.yaml
# Adjust gpu_memory_utilization and devices based on your GPU
```

## Quick Start

### Single Prompt TTS

```bash
cd /root/learning/vllm-omni/examples/offline_inference/kimi_audio_tts
bash run_single_prompt.sh
```

Generated audio files are saved to `output_audio/` by default.

### Batch TTS

```bash
bash run_batch_prompts.sh
```

## Usage

### Basic TTS

```python
python end2end.py \
    --model-path /data1/moonshotai/Kimi-Audio-7B-Instruct \
    --text "Hello, this is a test of Kimi-Audio text-to-speech." \
    --output-dir output_audio
```

### With Stage Config

```python
python end2end.py \
    --model-path /data1/moonshotai/Kimi-Audio-7B-Instruct \
    --text "Hello world" \
    --stage-configs-path ../../vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml \
    --output-dir output_audio
```

### Streaming Mode

```python
python end2end.py \
    --model-path /data1/moonshotai/Kimi-Audio-7B-Instruct \
    --text "Streaming audio generation test" \
    --streaming \
    --output-dir /tmp/kimi_audio_stream
```

## Configuration

### Stage Config Parameters

Edit `vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml`:

```yaml
async_chunk: true  # Enable streaming

# Stage 0: Talker
- stage_id: 0
  runtime:
    devices: "0"           # GPU ID
    max_batch_size: 10     # Max concurrent requests
  engine_args:
    gpu_memory_utilization: 0.5
    max_model_len: 4096

# Stage 1: Code2Wav
- stage_id: 1
  runtime:
    devices: "0"
    max_batch_size: 1
  engine_args:
    gpu_memory_utilization: 0.3
    max_model_len: 32768

# Connector settings
connectors:
  connector_of_shared_memory:
    extra:
      codec_chunk_frames: 25        # Frames per chunk
      codec_left_context_frames: 25  # Overlap for smooth decoding
```

## Output Format

Generated audio files are saved as WAV format:
- Sample rate: 24kHz
- Format: 16-bit PCM
- Channels: Mono

## Troubleshooting

### Out of Memory

Reduce `gpu_memory_utilization` in stage config:
```yaml
engine_args:
  gpu_memory_utilization: 0.3  # Reduce from 0.5
```

### Audio Quality Issues

1. Ensure you're using the correct model checkpoint
2. Check that audio detokenizer is loaded properly
3. Verify `codec_chunk_frames` and `codec_left_context_frames` settings

### Slow Generation

1. Enable async_chunk streaming
2. Reduce max_batch_size if GPU is overloaded
3. Use CUDA graphs (enforce_eager: false)

## Examples

### Generate from Text File

```bash
python end2end.py \
    --model-path /data1/moonshotai/Kimi-Audio-7B-Instruct \
    --text-file prompts.txt \
    --output-dir output_audio \
    --batch-size 4
```

### Custom Voice Settings

```python
python end2end.py \
    --model-path /data1/moonshotai/Kimi-Audio-7B-Instruct \
    --text "Custom voice test" \
    --temperature 0.8 \
    --top-k 40 \
    --output-dir output_audio
```

## Performance

Typical performance on H100:
- Time to First Audio (TTFA): ~500ms
- Real-time Factor (RTF): ~0.1 (10x faster than real-time)
- Batch throughput: ~10 requests/sec

## Notes

- The script uses the model path embedded in `end2end.py`. Update if your cache path differs.
- Use `--output-dir` to change the output folder.
- Streaming mode requires `async_chunk: true` in stage config.
