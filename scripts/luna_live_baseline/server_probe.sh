#!/usr/bin/env bash
# 只读核实 ai-lingxi 上当前 codex-luna-review-worker 的脚本/CLI/模型/提示词/预处理事实。
# 不读取、不打印 private.env（只列键名）；不调用模型。
set -uo pipefail

D=/data/workspace/jzheng/codex-luna-review
REL="$D/releases/25a612f28403bed971db96430ab638508589b5f6"
PENV="$D/releases/47ca276bf77f4901627b12e81b49b5363555757b/prompt.env"

echo "### date"; date
echo "### service"
systemctl --user is-active codex-luna-review-worker || true
systemctl --user status codex-luna-review-worker --no-pager 2>&1 | sed -n '1,12p'
echo "### unit + drop-in"
systemctl --user cat codex-luna-review-worker 2>&1 | sed -n '1,60p'
echo "### worker source sha256"
sha256sum "$REL/scripts/run_codex_luna_worker.py"
echo "### SHA256SUMS (release)"
cat "$REL/SHA256SUMS"
echo "### prompt.env (非密钥)"
cat "$PENV"
echo "### private.env 键名（不打印值）"
cut -d= -f1 "$D/private.env"
echo "### CLI"
"$D/bin/codex-0.162.0-alpha.17.2/codex" --version 2>&1 | head -1
sha256sum "$D/bin/codex-0.162.0-alpha.17.2/codex"
echo "### assets"
A=/data/workspace/jzheng/luna-prompt-autotune/assets
sha256sum "$A/labels.json" "$A/items.json" "$A/build.json"
echo "images: $(ls "$A/images" | wc -l)"
echo "### resources"
free -g
nproc
df -h /data /tmp
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1 | head -8
echo "### listeners"
ss -ltn 2>/dev/null | grep -E '1839[0-9]' || echo "(none)"
