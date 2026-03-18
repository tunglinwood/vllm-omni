#!/bin/bash
# Run batch Kimi-Audio TTS inference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="/data1/moonshotai/Kimi-Audio-7B-Instruct"
OUTPUT_DIR="output_audio_batch"
STAGE_CONFIG="../../vllm_omni/model_executor/stage_configs/kimi_audio_tts.yaml"
PROMPTS_FILE="prompts.txt"

echo "========================================"
echo "Kimi-Audio TTS - Batch Inference"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Stage Config: $STAGE_CONFIG"
echo ""

# Create sample prompts file if it doesn't exist
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Creating sample prompts file..."
    cat > "$PROMPTS_FILE" << 'EOF'
Hello, this is the first test prompt for Kimi-Audio TTS.
Welcome to vLLM-Omni, the fast omni-modal inference engine.
Kimi-Audio supports both speech recognition and text-to-speech.
This is a batch processing demonstration with multiple prompts.
Thank you for using Kimi-Audio with vLLM-Omni!
EOF
fi

python end2end.py \
    --model-path "$MODEL_PATH" \
    --text-file "$PROMPTS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --stage-configs-path "$STAGE_CONFIG" \
    --batch-size 4 \
    --temperature 0.9 \
    --top-k 50 \
    --seed 42

echo ""
echo "========================================"
echo "Batch generation complete!"
echo "Output files: $OUTPUT_DIR"
echo "========================================"
