#!/usr/bin/env bash
# 在 ai-lingxi 上启动本任务**独立**的 Luna worker 实例（不触碰 18396 共享 worker）。
#
# 资源上限：MemoryMax=4G、MemorySwapMax=0、CPUQuota=200%，绑定 127.0.0.1:18398。
# 复用线上当前实际代码与提示词版本 p000-pure-class-cards；token 现场随机生成、仅本用户可读。
# 不打印任何密钥。
set -euo pipefail

WORKDIR="${WORKDIR:-/data/workspace/jzheng/luna-live-baseline-eval}"
RELEASE="${RELEASE:-/data/workspace/jzheng/codex-luna-review/releases/25a612f28403bed971db96430ab638508589b5f6}"
SHARED_ENV="${SHARED_ENV:-/data/workspace/jzheng/codex-luna-review/private.env}"
UNIT="${UNIT:-luna-live-baseline-eval-worker}"
PORT="${PORT:-18398}"
MEM="${MEM:-4G}"
CPU="${CPU:-200%}"
MODEL="${CODEX_REVIEW_MODEL:-gpt-5.6-luna}"
REASONING_EFFORT="${CODEX_REVIEW_REASONING_EFFORT:-none}"

mkdir -p "$WORKDIR/private" "$WORKDIR/tmp" "$WORKDIR/logs"
chmod 700 "$WORKDIR/private"

# 从共享 private.env 只取二进制/账号目录等非密钥路径；取不到则报错，不猜测。
getpath() { grep -E "^$1=" "$SHARED_ENV" | head -1 | cut -d= -f2-; }
BINARY="$(getpath CODEX_REVIEW_BINARY)"
HOME_DIR="$(getpath CODEX_REVIEW_HOME)"
if [ -z "$BINARY" ] || [ -z "$HOME_DIR" ]; then
  echo "无法从共享 private.env 读取 CODEX_REVIEW_BINARY / CODEX_REVIEW_HOME 路径" >&2
  exit 1
fi

TOKEN_FILE="$WORKDIR/private/token"
ENV_FILE="$WORKDIR/private/worker.env"
umask 077
if [ ! -s "$TOKEN_FILE" ]; then
  openssl rand -hex 32 > "$TOKEN_FILE"
fi
TOKEN="$(cat "$TOKEN_FILE")"
{
  echo "CODEX_REVIEW_WORKER_TOKEN=$TOKEN"
  echo "CODEX_REVIEW_BIND=127.0.0.1"
  echo "CODEX_REVIEW_PORT=$PORT"
  echo "CODEX_REVIEW_PROMPT_VERSION=p000-pure-class-cards"
  echo "CODEX_REVIEW_MODEL=$MODEL"
  echo "CODEX_REVIEW_REASONING_EFFORT=$REASONING_EFFORT"
  echo "CODEX_REVIEW_BINARY=$BINARY"
  echo "CODEX_REVIEW_HOME=$HOME_DIR"
  echo "CODEX_REVIEW_TMP_DIR=$WORKDIR/tmp"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"

# 停掉同名的旧临时实例（只动本任务自己的 unit）。
systemctl --user stop "$UNIT" 2>/dev/null || true
systemctl --user reset-failed "$UNIT" 2>/dev/null || true

systemd-run --user --unit="$UNIT" \
  -p EnvironmentFile="$ENV_FILE" \
  -p MemoryMax="$MEM" -p MemorySwapMax=0 -p CPUQuota="$CPU" \
  -p Restart=no -p KillMode=control-group -p TimeoutStopSec=20 \
  /usr/bin/python3 "$RELEASE/scripts/run_codex_luna_worker.py"

# 等健康检查；只探测本任务端口。
for i in $(seq 1 30); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "独立 worker 就绪：$UNIT @ 127.0.0.1:$PORT"
    echo "资源上限：MemoryMax=$MEM CPUQuota=$CPU MemorySwapMax=0"
    exit 0
  fi
  sleep 1
done
echo "独立 worker 启动后未通过健康检查" >&2
systemctl --user status "$UNIT" --no-pager 2>&1 | tail -20 >&2 || true
exit 1
