#!/bin/bash
set -e
source "$(dirname "$0")/.env" 2>/dev/null || true
MODEL=${STUDENT_MODEL:-"nvidia/OpenMath-Nemotron-7B"}
PORT=${STUDENT_PORT:-8009}
GPU_MEM_UTIL=${STUDENT_GPU_MEMORY_UTIL:-0.85}
MAX_MODEL_LEN=${STUDENT_MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${STUDENT_MAX_NUM_SEQS:-8}
SWAP_SPACE=${STUDENT_SWAP_SPACE:-8}
SPECULATIVE=${SPECULATIVE_ENABLED:-false}
DRAFT_MODEL=${DRAFT_MODEL:-""}
echo "Starting Student vLLM server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEM_UTIL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Max Num Seqs: $MAX_NUM_SEQS"
echo "Swap Space: ${SWAP_SPACE}GB"
CMD="python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --port $PORT \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --enforce-eager \
  --swap-space $SWAP_SPACE"
if [ "$SPECULATIVE" = "true" ] && [ -n "$DRAFT_MODEL" ]; then
  echo "Speculative decoding enabled with draft model: $DRAFT_MODEL"
  CMD="$CMD --speculative-model $DRAFT_MODEL"
fi
echo "Command: $CMD"
exec $CMD
