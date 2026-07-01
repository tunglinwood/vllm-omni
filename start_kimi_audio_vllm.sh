#!/bin/bash
# Start Kimi Audio with vllm serve on GPU 7
# Note: This uses standard vLLM, not vllm-omni multi-stage pipeline

# Use GPU 7
export CUDA_VISIBLE_DEVICES=7

# Model path
MODEL_PATH="/data1/moonshotai/Kimi-Audio-7B-Instruct"

# Server configuration
PORT=8092  # Using different port to avoid conflict with vllm-omni instance
HOST="0.0.0.0"
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.9

# Start vllm serve
echo "Starting Kimi Audio with vllm serve on GPU 7..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""
echo "Note: Standard vLLM may not fully support Kimi Audio's multi-stage architecture."
echo "If this fails, use vllm-omni serve instead (see ecosystem.config.cjs)."
echo ""

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --enforce-eager
