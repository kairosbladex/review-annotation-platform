#!/usr/bin/env python3
"""从只读的 Luna 自动迭代数据生成复核队列。不改服务器，不调用模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adjudication import build_package


def main() -> None:
    parser = argparse.ArgumentParser(description="按固定规则生成疑似标签错误复核队列")
    parser.add_argument("--dataset", type=Path, required=True, help="含 assets/ 与 runs/p000、p000b、p001 的数据根目录")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    summary = build_package(args.dataset, args.out)
    print(json.dumps({
        "selected_boxes": summary["selected_boxes"],
        "selected_images": summary["selected_images"],
        "by_rule": summary["by_rule"],
        "by_class": summary["by_class"],
        "by_class_and_rule": summary["by_class_and_rule"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
