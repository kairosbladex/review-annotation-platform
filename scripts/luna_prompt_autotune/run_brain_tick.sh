#!/bin/sh
# 定时检查是否需要唤醒 Hermes（由 systemd 用户定时器每 10 分钟调用）。
# 只有“有待写报告的版本”或“可以提交下一版”时才加载大脑模型；共享服务 L20 空闲显存不足、STOP、上一轮未结束时跳过。
# Hermes 连续 3 次唤醒都没有让状态前进（没写报告也没提交）时，写 STOP 并停用定时器，避免空转占卡。
set -u
ROOT=/data/workspace/jzheng/luna-prompt-autotune
MODEL=qwen3.6:latest
NEED_MIB=30000
SERVICE_GPU=GPU-9f2ac4d0-be45-fe5b-d16a-86d1f3fd10f7
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$ROOT/logs/brain"
if [ -e "$ROOT/state/STOP" ]; then echo "$TS 跳过：STOP 文件存在"; exit 0; fi
if ! python3 "$ROOT/src/runner.py" --brain-needed; then echo "$TS 无需唤醒"; exit 0; fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$SERVICE_GPU" | tr -d ' ')
if [ "$FREE" -lt "$NEED_MIB" ]; then echo "$TS 跳过：共享卡空闲显存 ${FREE}MiB < ${NEED_MIB}MiB"; exit 0; fi
if [ -n "$(docker ps -q --filter name=luna-autotune-brain)" ]; then echo "$TS 跳过：上一轮仍在运行"; exit 0; fi
fingerprint() { cat "$ROOT"/state/rounds/*.json "$ROOT"/state/ledger.jsonl 2>/dev/null | sha256sum | cut -c1-16; }
BEFORE=$(fingerprint)
timeout 1500 docker run --rm --name luna-autotune-brain --label owner=agent --label autotune.role=brain \
  --memory=4g --memory-swap=4g --cpus=2 --network diagnostic-pilot-internal \
  -v "$ROOT/state":/work/state -v "$ROOT/src/mcp":/work/src:ro -v "$ROOT/hermes":/work/hermes \
  -e HERMES_HOME=/work/hermes -e HOME=/work/hermes -e OPENAI_API_KEY=local-only -e NO_PROXY='*' \
  --entrypoint hermes diagnostic-agent-runtime:pilot-v7 \
  chat --query-file /work/src/prompt.md --oneshot --toolsets luna_autotune,vision > "$ROOT/logs/brain/$TS.log" 2>&1
CODE=$?
if [ "$CODE" = 124 ]; then docker stop -t 10 luna-autotune-brain >/dev/null 2>&1; fi
AFTER=$(fingerprint)
if [ "$BEFORE" = "$AFTER" ]; then N=$(( $(cat "$ROOT/state/brain_no_progress" 2>/dev/null || echo 0) + 1 )); else N=0; fi
echo "$N" > "$ROOT/state/brain_no_progress"
if [ "$N" -ge 3 ]; then
  echo "Hermes 连续 3 次唤醒无进展，自动停止" > "$ROOT/state/STOP"
  systemctl --user disable --now luna-autotune-brain.timer luna-autotune-runner.timer
fi
echo "$TS 完成，退出码 $CODE，共享卡空闲显存 ${FREE}MiB，无进展计数 $N"
