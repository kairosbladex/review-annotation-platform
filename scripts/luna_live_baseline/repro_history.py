#!/usr/bin/env python3
"""用冻结官方计分复现历史 Luna 运行结果（不调用模型）。

复现 luna-relabel-rescore-20261010 记录的自检数字，证明计分口径与历史一致：
  p000 holdout 0.7562 / p000 dev 0.7299 / p000b dev 0.7124 / p001 dev 0.7608 /
  p000b-holdout 0.7675。

输入目录需含（见 rescore.py 说明）：
  assets/labels.json
  runs/<版本>/out/*.json          （raw-runs.tar.gz --strip-components=1 解出）
  human/final-20261009/*.json

用法：
  python3 scripts/luna_live_baseline/repro_history.py --autotune-final /tmp/luna_final
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

DEFAULT_FROZEN = Path("data/luna-live-baseline/private/frozen")
EXPECTED = {
    "p000 holdout": ("p000", "holdout", 0.7562),
    "p000 dev": ("p000", "dev", 0.7299),
    "p000b dev": ("p000b", "dev", 0.7124),
    "p001 dev": ("p001", "dev", 0.7608),
    "p000b-holdout": ("p000b-holdout", "holdout", 0.7675),
}


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def res_from_outdir(d: Path) -> dict:
    res = {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        for x in r.get("results", []):
            res[x["key"]] = x["raw"]
    return res


def res_from_decisions(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["raw"] for k, v in d["decisions"].items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autotune-final", type=Path, default=Path("/tmp/luna_final"))
    ap.add_argument("--official", type=Path, default=Path("scripts/luna_live_baseline/metrics_official.py"))
    ap.add_argument("--labels", type=Path, default=None,
                    help="默认用 --autotune-final/assets/labels.json")
    args = ap.parse_args()

    root = args.autotune_final
    official = load_official(args.official)
    labels = json.loads((args.labels or root / "assets/labels.json").read_text(encoding="utf-8"))

    n_dev = sum(1 for v in labels.values() if v["split"] == "dev")
    n_hold = sum(1 for v in labels.values() if v["split"] == "holdout")
    print(f"冻结标签：dev {n_dev} / holdout {n_hold} / 合计 {len(labels)}")

    res = {v: res_from_outdir(root / "runs" / v / "out") for v in ("p000", "p000b", "p001")}
    res["p000b-holdout"] = res_from_decisions(root / "human/final-20261009/p000b-holdout.json")

    failures = []
    for name, (version, split, exp) in EXPECTED.items():
        got = official.score(res[version], labels, split)["overall"]["accuracy"]
        ok = abs(got - exp) < 5e-5
        print(f"  {name}: {got}  期望 {exp}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(name)
    if n_dev != 685 or n_hold != 443:
        failures.append("labels_scale")
        print(f"  规模不符：期望 dev 685 / holdout 443，实际 {n_dev} / {n_hold}")
    if failures:
        raise SystemExit(f"复现失败：{failures}")
    print("历史结果复现：全部一致")


if __name__ == "__main__":
    main()
