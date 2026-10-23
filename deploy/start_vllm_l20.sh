#!/usr/bin/env bash
set -euo pipefail

# Launch Qwen3-VL-8B-Instruct + LoRA on NVIDIA L20 (48GB SM89, default GPU 4)
# Enforces host memory/CPU check and container resource caps per OOM safety rules.

GPU_DEVICE="${GPU_DEVICE:-4}"
VLLM_PORT="${VLLM_PORT:-8009}"
MODEL_DIR="${QWEN3VL_MODEL_DIR:-/data/models/Qwen3-VL-8B-Instruct}"
LORA_DIR="${QWEN3VL_LORA_DIR:-/data/models/qwen3vl-ktd-lora}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen3vl-l20}"
MEM_LIMIT="${MEM_LIMIT:-32g}"
CPU_LIMIT="${CPU_LIMIT:-8}"

echo "[preflight] Host resource check before starting ${CONTAINER_NAME} on GPU ${GPU_DEVICE}:"
free -g || true
nproc || true
df -h || true
nvidia-smi --query-gpu=index,name,memory.free,memory.used,memory.total --format=csv,noheader,nounits

exec docker run --rm \
  --name "${CONTAINER_NAME}" \
  --label owner=agent \
  --memory="${MEM_LIMIT}" \
  --memory-swap="${MEM_LIMIT}" \
  --cpus="${CPU_LIMIT}" \
  --gpus "\"device=${GPU_DEVICE}\"" \
  --ipc=host \
  -p "127.0.0.1:${VLLM_PORT}:8000" \
  -v "${MODEL_DIR}:/models/Qwen3-VL-8B-Instruct:ro" \
  -v "${LORA_DIR}:/models/qwen3vl-ktd-lora:ro" \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules "ktd-verifier=/models/qwen3vl-ktd-lora"
