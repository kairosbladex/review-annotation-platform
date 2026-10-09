#!/usr/bin/env python3
"""根据 adjudication.csv 生成本地局部放大图和整图缩略图。保留已填写的复核结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adjudication import prepare_views


def main() -> None:
    parser = argparse.ArgumentParser(description="生成复核页用的局部放大图和整图缩略图")
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(prepare_views(args.package), ensure_ascii=False))


if __name__ == "__main__":
    main()
