#!/usr/bin/env bash
# 停止并清理本任务在 ai-lingxi 上的独立临时 worker（不触碰 18396 共享 worker）。
set -euo pipefail

UNIT="${UNIT:-luna-live-baseline-eval-worker}"
WORKDIR="${WORKDIR:-/data/workspace/jzheng/luna-live-baseline-eval}"
systemctl --user stop "$UNIT" 2>/dev/null || true
systemctl --user reset-failed "$UNIT" 2>/dev/null || true

echo "已停止 $UNIT。保留工作目录 $WORKDIR（失败日志/原始输出备查）；"
echo "需要彻底清理时手动删除该目录。"
