#!/usr/bin/env bash
set -euo pipefail

# Tier 2.5 —— Jev-Omni Fast Gate 推理微服务启动脚本
#
# 在空闲 GPU 上以容器方式启动 akhilaaa3/Jev-Omni（12B）推理服务，主后端通过
# JEV_OMNI_BASE_URL=http://<host>:<port> 调用 /v1/decide 获取校准 P(true)。
#
# 卡型策略：
#   * L20 (48GB, SM89) -> BF16，24g 容器内存足够
#   * T4  (16GB, SM75) -> 4-bit 量化（load_in_4bit），显存约 8-10GB
#   默认 PRECISION=auto，按所选 GPU 型号自动选择。
#
# 用法：
#   GPU_DEVICE=4 bash deploy/start_jev_omni.sh              # 自动选精度（L20 -> bf16）
#   GPU_DEVICE=1 PRECISION=4bit bash deploy/start_jev_omni.sh
#   GPU_DEVICE=4 PRECISION=bf16 JEV_OMNI_PORT=18010 bash deploy/start_jev_omni.sh
#
# 端口：宿主机 8010 已被无关容器 myolotrain-web-1 占用，因此默认改为 18010。
# 容器内部仍监听 8010，映射关系为 `127.0.0.1:18010 -> 8010`。
#
# 只读预检（不启动容器）：DRY_RUN=1 bash deploy/start_jev_omni.sh

GPU_DEVICE="${GPU_DEVICE:-4}"
JEV_OMNI_PORT="${JEV_OMNI_PORT:-18010}"
CONTAINER_NAME="${CONTAINER_NAME:-jev-omni-t25}"
PRECISION="${PRECISION:-auto}"
MODEL_ID="${JEV_OMNI_MODEL_ID:-akhilaaa3/Jev-Omni}"
MODEL_DIR="${JEV_OMNI_MODEL_DIR:-/data/models/Jev-Omni}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
IMAGE="${JEV_OMNI_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
MEM_LIMIT="${MEM_LIMIT:-28g}"
CPU_LIMIT="${CPU_LIMIT:-8}"
DRY_RUN="${DRY_RUN:-0}"
# 镜像内若未预装 jev_omni/fastapi，可用 JEV_OMNI_BOOTSTRAP 在启动前安装依赖，例如：
#   JEV_OMNI_BOOTSTRAP="pip install -q -U jev-omni fastapi uvicorn pillow httpx"
BOOTSTRAP="${JEV_OMNI_BOOTSTRAP:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[preflight] 资源水位与选卡（GPU ${GPU_DEVICE}）："
free -g || true
nproc || true
df -h "${REPO_ROOT}" || true
nvidia-smi --query-gpu=index,name,memory.free,memory.used,memory.total --format=csv,noheader,nounits || true

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU_DEVICE}" 2>/dev/null || echo unknown)"
echo "[preflight] GPU ${GPU_DEVICE} = ${GPU_NAME}"

if [ "${PRECISION}" = "auto" ]; then
  case "${GPU_NAME}" in
    *L20*|*L40*|*A100*|*H100*|*H800*|*A800*) PRECISION="bf16" ;;
    *) PRECISION="4bit" ;;
  esac
fi
echo "[preflight] 精度策略：${PRECISION}"

MODEL_ARGS=()
if [ -d "${MODEL_DIR}" ]; then
  echo "[preflight] 使用本地权重目录：${MODEL_DIR}"
  MODEL_ARGS=(-v "${MODEL_DIR}:/models/Jev-Omni:ro" -e JEV_OMNI_MODEL_PATH=/models/Jev-Omni)
else
  echo "[preflight] 本地权重目录不存在，回退 HuggingFace 缓存：${HF_CACHE_DIR}"
  mkdir -p "${HF_CACHE_DIR}"
  MODEL_ARGS=(-v "${HF_CACHE_DIR}:/hf-cache" -e HF_HOME=/hf-cache)
  if [ -n "${HF_TOKEN:-}" ]; then
    MODEL_ARGS+=(-e "HF_TOKEN=${HF_TOKEN}")
  fi
fi

PRECISION_ARGS=(-e JEV_OMNI_DEVICE=cuda)
if [ "${PRECISION}" = "bf16" ]; then
  PRECISION_ARGS+=(-e JEV_OMNI_DTYPE=bfloat16 -e JEV_OMNI_QUANTIZATION=)
else
  PRECISION_ARGS+=(-e JEV_OMNI_DTYPE=float16 -e JEV_OMNI_QUANTIZATION=4bit)
fi

if [ -n "${BOOTSTRAP}" ]; then
  ENTRY=(bash -lc "${BOOTSTRAP} && python deploy/start_jev_omni_service.py")
else
  ENTRY=(python deploy/start_jev_omni_service.py)
fi

DOCKER_CMD=(docker run --rm
  --name "${CONTAINER_NAME}"
  --label owner=agent
  --memory="${MEM_LIMIT}"
  --memory-swap="${MEM_LIMIT}"
  --cpus="${CPU_LIMIT}"
  --gpus "\"device=${GPU_DEVICE}\""
  --ipc=host
  -p "127.0.0.1:${JEV_OMNI_PORT}:8010"
  -v "${REPO_ROOT}:/workspace:ro"
  -w /workspace
  -e "JEV_OMNI_MODEL_ID=${MODEL_ID}"
  -e JEV_OMNI_PORT=8010
  -e JEV_OMNI_PRELOAD=true
  "${MODEL_ARGS[@]}"
  "${PRECISION_ARGS[@]}"
  "${IMAGE}"
  "${ENTRY[@]}")

if [ "${DRY_RUN}" = "1" ]; then
  echo "[dry-run] ${DOCKER_CMD[*]}"
  exit 0
fi

echo "[start] 健康检查：curl -s http://127.0.0.1:${JEV_OMNI_PORT}/healthz"
echo "[start] 就绪探针：curl -s -X POST http://127.0.0.1:${JEV_OMNI_PORT}/v1/warmup"
exec "${DOCKER_CMD[@]}"
