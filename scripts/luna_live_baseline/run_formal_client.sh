#!/usr/bin/env bash
# 在 ai-lingxi 上以**限资源**方式启动本任务 baseline 客户端（只停/启自己的 unit）。
#
# 资源上限：MemoryMax=1G、MemorySwapMax=0、CPUQuota=100%。
# 正式范围：历史 holdout 137 图，两轮；失败每请求最多重试 1 次；总尝试上限 548（=4N）。
set -euo pipefail

WORKDIR="${WORKDIR:-/data/workspace/jzheng/luna-live-baseline-eval}"
ASSETS="${ASSETS:-/data/workspace/jzheng/luna-prompt-autotune/assets}"
UNIT="${UNIT:-luna-live-baseline-eval-client}"
PORT="${PORT:-18398}"
MEM="${MEM:-1G}"
CPU="${CPU:-100%}"
OUT="${OUT:-$WORKDIR/formal}"
LEDGER="${LEDGER:-$WORKDIR/logs/formal_ledger.jsonl}"

if [ -f "$WORKDIR/STOP" ]; then
  echo "STOP 文件存在，拒绝启动：$(cat "$WORKDIR/STOP")" >&2
  exit 1
fi

systemctl --user stop "$UNIT" 2>/dev/null || true
systemctl --user reset-failed "$UNIT" 2>/dev/null || true

systemd-run --user --unit="$UNIT" \
  -p MemoryMax="$MEM" -p MemorySwapMax=0 -p CPUQuota="$CPU" \
  -p Restart=no -p KillMode=control-group -p TimeoutStopSec=30 \
  /usr/bin/python3 "$WORKDIR/scripts/evaluate.py" \
    --worker-url "http://127.0.0.1:$PORT/review" \
    --token-file "$WORKDIR/private/token" \
    --assets "$ASSETS" --labels "$ASSETS/labels.json" \
    --out "$OUT" --splits holdout --rounds 2 --concurrency 2 --max-retries 1 \
    --resume --ledger "$LEDGER" --stop-file "$WORKDIR/STOP" \
    --budget-cap 548 --shared-container yoloe_service_test --resource-interval 60

echo "已启动 $UNIT（MemoryMax=$MEM CPUQuota=$CPU MemorySwapMax=0）"
echo "日志：journalctl --user -u $UNIT -f"
