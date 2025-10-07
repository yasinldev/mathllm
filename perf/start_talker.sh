#!/bin/bash
set -e
source "$(dirname "$0")/.env" 2>/dev/null || true
MODE=${TALKER_MODE:-"vllm"}
MODEL=${TALKER_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
PORT=${TALKER_PORT:-8010}
MAX_MODEL_LEN=${TALKER_MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${TALKER_MAX_NUM_SEQS:-6}
echo "Starting Talker server..."
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Port: $PORT"
if [ "$MODE" = "vllm" ]; then
  echo "Using vLLM backend"
  echo "Max Model Length: $MAX_MODEL_LEN"
  echo "Max Num Seqs: $MAX_NUM_SEQS"
  exec python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS
elif [ "$MODE" = "llamacpp" ]; then
  GGUF_PATH=${TALKER_GGUF_PATH:-"./models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"}
  NGL=${TALKER_NGL:-20}
  THREADS=${TALKER_THREADS:-8}
  echo "Using llama.cpp backend"
  echo "GGUF Path: $GGUF_PATH"
  echo "GPU Layers (-ngl): $NGL"
  echo "CPU Threads: $THREADS"
  exec ./llama.cpp/main \
    -m "$GGUF_PATH" \
    -c $MAX_MODEL_LEN \
    -ngl $NGL \
    -t $THREADS \
    --port $PORT
else
  echo "Error: Unknown MODE=$MODE (must be 'vllm' or 'llamacpp')"
  exit 1
fi
