#!/bin/sh
# 定时唤醒 Hermes 执行一轮提示词（由 systemd 用户定时器调用）。
# 停止文件存在或共享服务 L20 空闲显存不足时跳过本轮；大脑模型 qwen3.6 由 diagnostic-ollama 按需加载后卸载。
set -u
LIVE=/REDACTED_RUNTIME_ROOT/recheck-autotrain/live
MODEL=qwen3.6:latest
NEED_MIB=30000
SERVICE_GPU=GPU-REDACTED
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LIVE/logs/brain"
if [ -e "$LIVE/state/STOP" ]; then echo "$TS 跳过：STOP 文件存在"; exit 0; fi
if [ "$(date +%Y%m%d)" -gt 20261008 ]; then echo "$TS 跳过：已过截止日期 2026-10-08"; exit 0; fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$SERVICE_GPU" | tr -d ' ')
if [ "$FREE" -lt "$NEED_MIB" ]; then echo "$TS 跳过：共享卡空闲显存 ${FREE}MiB < ${NEED_MIB}MiB"; exit 0; fi
if [ -n "$(docker ps -q --filter name=autotrain-brain)" ]; then echo "$TS 跳过：上一轮仍在运行"; exit 0; fi
timeout 1500 docker run --rm --name autotrain-brain --label owner=agent --label autotrain.role=brain \
  --memory=4g --memory-swap=4g --cpus=2 --network diagnostic-pilot-internal \
  -v "$LIVE/state":/work/state -v "$LIVE/src/mcp":/work/src:ro -v "$LIVE/hermes":/work/hermes \
  -e HERMES_HOME=/work/hermes -e HOME=/work/hermes -e OPENAI_API_KEY=local-only -e NO_PROXY='*' \
  --entrypoint hermes diagnostic-agent-runtime:pilot-v7 \
  chat --query-file /work/src/prompt.md --oneshot --toolsets autotrain,vision > "$LIVE/logs/brain/$TS.log" 2>&1
CODE=$?
# timeout 只结束 docker 客户端，超时后需显式停止容器
if [ "$CODE" = 124 ]; then docker stop -t 10 autotrain-brain >/dev/null 2>&1; fi
echo "$TS 完成，退出码 $CODE，共享卡空闲显存 ${FREE}MiB"
