#!/usr/bin/env python3
"""把新微调模型的输出与已有基础模型输出合并，得到与 analyze_resolution_ab.py 相同格式的预测文件。

基础模型不随微调变化，因此比较新旧微调模型时只需重跑 adapter 一路；
合并时按 (candidate_id, arm) 对齐，基础模型字段取自旧预测，adapter 字段取自新预测。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(paths: list[Path]) -> dict[tuple[str, str], dict]:
    rows = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = (row["candidate_id"], row["arm"])
                if key in rows:
                    raise SystemExit(f"重复预测：{key}")
                rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--adapter-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = load(args.base_predictions)
    adapter = load(args.adapter_predictions)
    if set(base) != set(adapter):
        raise SystemExit(f"两组预测的样本不一致：仅基础 {len(set(base) - set(adapter))}，仅新 {len(set(adapter) - set(base))}")
    merged = []
    for key in sorted(base):
        row = {k: v for k, v in base[key].items() if not k.startswith("adapter_")}
        row.update({k: v for k, v in adapter[key].items() if k.startswith("adapter_")})
        merged.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in merged), encoding="utf-8")
    print(f"合并 {len(merged)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
