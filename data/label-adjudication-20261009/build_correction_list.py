#!/usr/bin/env python3
"""从 adjudication.csv 生成更正清单。只列出实际改了标签的框，不修改任何金标文件。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from adjudication import LABEL_ZH, read_csv, write_corrections


def main() -> None:
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="生成标签更正清单（不改金标）")
    parser.add_argument("--csv", type=Path, default=package / "adjudication.csv")
    parser.add_argument("--out", type=Path, default=package / "corrections.csv")
    args = parser.parse_args()
    changed = write_corrections(read_csv(args.csv), args.out)
    counts = Counter(f"{LABEL_ZH.get(row['original_label'], row['original_label'])}→{LABEL_ZH.get(row['corrected_label'], row['corrected_label'])}" for row in changed)
    print(f"更正 {len(changed)} 个框，已写入 {args.out}")
    for name, count in sorted(counts.items()):
        print(f"  {name}：{count}")
    if not changed:
        print("没有需要更正的框。维持原标签、看不清和未复核的框不会列入。")


if __name__ == "__main__":
    main()
