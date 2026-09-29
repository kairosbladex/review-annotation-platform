#!/usr/bin/env python3
"""用高清复核标签检查车辆支撑面 v3 验收的两个数量门槛。

门槛口径（f09df00）：
  1. 人工标为屋顶设施 >= 30；
  2. 首批抽样层 facility_or_unclear_candidate 与补充包中人工标为真车 >= 30。
另外按 v3 实际作用范围（四视图一致确认 + 支撑面楼顶/未知）单独统计有效样本数。
门槛未满足时不计算 v3 放过数、上界和找回率。
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "data/vehicle-surface-gate-eval-20260929/prerelease-evidence"
THRESHOLD = 30


def load_rows(evidence: Path) -> list[dict]:
    with (evidence / "hires-review-package/annotations.csv").open(encoding="utf-8-sig", newline="") as handle:
        labels = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for package in ("annotation-package", "supplement-annotation-package"):
        for line in (evidence / package / "selected.jsonl").read_text(encoding="utf-8").splitlines():
            sel = json.loads(line)
            label = labels[sel["candidate_id"]]
            rows.append({
                "candidate_id": sel["candidate_id"],
                "package": package,
                "stratum": sel.get("selection_stratum"),
                "v3_in_scope": sel["binary_outcome"] == "confirmed" and sel["surface"] in ("building_roof", "unknown"),
                "label": label["label"],
                "lowres_label": label["lowres_label"],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.evidence)
    missing = [r["candidate_id"] for r in rows if r["label"] not in {"真车", "屋顶设施", "看不清"}]
    if missing:
        raise SystemExit(f"还有 {len(missing)} 个候选未标或标签无效：{missing[:5]}")

    def count(pred) -> int:
        return sum(1 for r in rows if pred(r))

    real_pool = lambda r: r["package"] == "supplement-annotation-package" or r["stratum"] == "facility_or_unclear_candidate"
    facility = count(lambda r: r["label"] == "屋顶设施")
    real = count(lambda r: r["label"] == "真车" and real_pool(r))
    facility_scope = count(lambda r: r["label"] == "屋顶设施" and r["v3_in_scope"])
    real_scope = count(lambda r: r["label"] == "真车" and real_pool(r) and r["v3_in_scope"])
    passed = facility >= THRESHOLD and real >= THRESHOLD
    result = {
        "schema": "vehicle_surface_gate_check_v1",
        "label_source": "hires-review-package/annotations.csv",
        "candidates": len(rows),
        "label_counts": dict(collections.Counter(r["label"] for r in rows)),
        "labels_changed_from_lowres": count(lambda r: r["label"] != r["lowres_label"]),
        "label_changes": {f"{a}->{b}": n for (a, b), n in collections.Counter(
            (r["lowres_label"], r["label"]) for r in rows if r["label"] != r["lowres_label"]).items()},
        "gates": {
            "roof_facility": {"count": facility, "threshold": THRESHOLD, "gap": max(0, THRESHOLD - facility)},
            "real_vehicle_in_roof_or_unclear_pool": {"count": real, "threshold": THRESHOLD, "gap": max(0, THRESHOLD - real)},
        },
        "v3_in_scope_counts": {
            "note": "只计四视图一致确认且支撑面为楼顶/未知的候选；其余候选 v3 部件规则不会运行",
            "roof_facility": facility_scope,
            "real_vehicle_in_roof_or_unclear_pool": real_scope,
            "supplement_in_scope": count(lambda r: r["package"] == "supplement-annotation-package" and r["v3_in_scope"]),
            "supplement_total": count(lambda r: r["package"] == "supplement-annotation-package"),
        },
        "gates_passed": passed,
        "v3_metrics": "gates met; compute v3 metrics next" if passed else "withheld: quantity gates not met",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
