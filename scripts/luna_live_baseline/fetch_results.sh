#!/usr/bin/env bash
# 从 ai-lingxi 取回**脱敏**结果（decisions/units/台账），不含模型自由文本与任何密钥。
# 用法：bash scripts/luna_live_baseline/fetch_results.sh
set -euo pipefail

HOST="${HOST:-ai-lingxi}"
D="${D:-/data/workspace/jzheng/luna-live-baseline-eval}"
LOCAL="$(cd "$(dirname "$0")/../.." && pwd)/data/luna-live-baseline/private/results/formal"
mkdir -p "$LOCAL/decisions" "$LOCAL/units" "$LOCAL/logs"

fetch() { ssh -o BatchMode=yes "$HOST" "cat $1" > "$2"; echo "  <- $1 -> $2"; }

for r in 1 2; do
  fetch "$D/formal/decisions/round$r.json" "$LOCAL/decisions/round$r.json"
  fetch "$D/formal/units/round$r.jsonl" "$LOCAL/units/round$r.jsonl"
done
fetch "$D/logs/formal_ledger.jsonl" "$LOCAL/logs/formal_ledger.jsonl"
if ssh -o BatchMode=yes "$HOST" "test -f $D/logs/prior_calls.jsonl"; then
  fetch "$D/logs/prior_calls.jsonl" "$(cd "$(dirname "$0")/../.." && pwd)/data/luna-live-baseline/private/prior_calls.jsonl"
fi
echo "脱敏结果已取回到 ${LOCAL}（原始模型自由文本留在服务器，不入库）"
