#!/bin/sh
# 用法：run_eval_shard.sh <分片编号> <分片总数> <GPU UUID> <adapter SHA-256>
# 在评测容器内依次评测楼顶难例与随机流量两批的该分片（只跑微调模型一路，基础模型沿用已有输出）
set -e
K="$1"; N="$2"; GPU="$3"; SHA="$4"
export HOME=/out/home; mkdir -p "$HOME"
for S in hard random; do
  if [ "$S" = hard ]; then R=/hard/input/rows.jsonl; I=/hard/images; else R=/random/input/sample.jsonl; I=/random/images; fi
  python3 /src/run_resolution_ab_eval.py --rows "$R" --images "$I" --verifier /src/clean_source_qwen_verifier.py \
    --base-model /opt/clean-source/base --adapter /opt/clean-source/adapter --adapter-sha256 "$SHA" --models adapter \
    --expect-gpu "$GPU" --out "/out/out-$S-q$K" --shard-index "$K" --shard-count "$N"
done
