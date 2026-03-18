#!/bin/bash
# Run single prompt Kimi-Audio TTS inference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="/data1/moonshotai/Kimi-Audio-7B-Instruct"
OUTPUT_DIR="output_audio"
STAGE_CONFIG="../../vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml"

echo "========================================"
echo "Kimi-Audio TTS - Single Prompt"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Stage Config: $STAGE_CONFIG"
echo ""

python end2end.py \
    --model-path "$MODEL_PATH" \
    --text "Hello, this is a test of Kimi-Audio text-to-speech synthesis using vLLM-Omni." \
    --output-dir "$OUTPUT_DIR" \
    --stage-configs-path "$STAGE_CONFIG" \
    --temperature 0.9 \
    --top-k 50 \
    --seed 42

echo ""
echo "========================================"
echo "Generation complete!"
echo "Output files: $OUTPUT_DIR"
echo "========================================"
