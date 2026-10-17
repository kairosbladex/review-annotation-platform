#!/bin/sh
# 用法：run_brain_trial.sh <ollama 模型名> <演练目录> <需要的共享卡空闲显存 MiB>
# 在资源受限的临时容器中让 Hermes 以指定大脑模型执行一轮提示词；模型由 diagnostic-ollama（共享服务 L20）按需加载。
set -eu
MODEL="$1"; DIR="$2"; NEED_MIB="$3"
SERVICE_GPU=GPU-REDACTED
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$SERVICE_GPU" | tr -d ' ')
if [ "$FREE" -lt "$NEED_MIB" ]; then echo "共享卡空闲显存 ${FREE}MiB 不足 ${NEED_MIB}MiB，放弃本轮"; exit 3; fi
mkdir -p "$DIR/hermes"
cat > "$DIR/hermes/config.yaml" <<EOF
agent:
  max_turns: 24
  reasoning_effort: none
compression:
  enabled: false
memory:
  memory_enabled: false
  user_profile_enabled: false
model:
  base_url: http://diagnostic-ollama:11434/v1
  context_length: 65536
  default: $MODEL
  provider: diagnostic-local
providers:
  diagnostic-local:
    api: http://diagnostic-ollama:11434/v1
    context_length: 65536
    default_model: $MODEL
    extra_body:
      temperature: 0
      think: false
mcp_servers:
  autotrain:
    command: env
    args: [AUTOTRAIN_ROOT=/work/state, python, /work/src/mcp_server.py]
    timeout: 120
platform_toolsets:
  cli: [autotrain, vision]
tools:
  tool_search:
    enabled: "off"   # 直接暴露 MCP 工具，避免小模型写错 tool_call 嵌套格式
EOF
cp "$DIR/src/prompt.md" "$DIR/prompt.md"
START=$(date +%s)
set +e
docker run --rm --name "autotrain-brain-$(basename "$DIR")" --label owner=agent \
  --memory=4g --memory-swap=4g --cpus=2 --network diagnostic-pilot-internal \
  -v "$DIR":/work -e HERMES_HOME=/work/hermes -e HOME=/work/hermes -e OPENAI_API_KEY=local-only -e NO_PROXY='*' \
  --entrypoint hermes diagnostic-agent-runtime:pilot-v7 \
  chat --query-file /work/prompt.md --oneshot --toolsets autotrain,vision > "$DIR/hermes.log" 2>&1
CODE=$?
set -e
echo "{\"model\": \"$MODEL\", \"exit_code\": $CODE, \"seconds\": $(( $(date +%s) - START )), \"service_gpu_free_before_mib\": $FREE}" > "$DIR/trial.json"
cat "$DIR/trial.json"
