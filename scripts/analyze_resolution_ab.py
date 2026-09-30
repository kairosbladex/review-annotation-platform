#!/usr/bin/env python3
"""用 108 个高清人工标签统计复检四视图判定在低清/高清输入下的准确率差异。

真车应判 CONFIRMED，屋顶设施/非车应判 REJECTED；看不清不计准确率，单独列分布。
--labels 可指定其他标注包（如随机流量包）；默认读取楼顶难例的两个高清包。
两路一致为自动结论，不一致为转人工（与线上四视图一致性规则相同）。
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from math import comb
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "data/vehicle-surface-gate-eval-20260929/prerelease-evidence"
EXPECTED = {"真车": "CONFIRMED", "屋顶设施": "REJECTED", "非车": "REJECTED"}


def mcnemar_exact(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def consensus(pred: dict) -> str:
    if pred["base_decision"] == pred["adapter_decision"] == "CONFIRMED":
        return "CONFIRMED"
    if pred["base_decision"] == pred["adapter_decision"] == "REJECTED":
        return "REJECTED"
    return "MANUAL"


def load_labels(evidence: Path, extra: list[Path] | None = None) -> dict[str, dict]:
    labels = {}
    if extra:
        for path in extra:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    labels[row["candidate_id"]] = row
        return labels
    for package in ("hires-review-package", "new-events-hires-package"):
        with (evidence / package / "annotations.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                labels[row["candidate_id"]] = {**row, "package": row.get("origin_package") or package}
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="prepare 阶段的 inputs.jsonl")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="*", help="标注 CSV；不指定时用楼顶难例的两个高清包")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    labels = load_labels(EVIDENCE, args.labels)
    inputs = {json.loads(l)["event_id"]: json.loads(l) for l in args.inputs.read_text().splitlines() if l.strip()}
    rows = {json.loads(l)["candidate_id"]: json.loads(l) for l in args.rows.read_text().splitlines() if l.strip()}
    preds = collections.defaultdict(dict)
    for path in args.predictions:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                preds[p["candidate_id"]][p["arm"]] = p
    missing = [cid for cid in rows if set(preds[cid]) != {"low", "high"}]
    if missing:
        raise SystemExit(f"{len(missing)} 个候选缺少两组预测：{missing[:3]}")

    items = []
    for cid, row in rows.items():
        lab = labels[cid]
        low, high = preds[cid]["low"], preds[cid]["high"]
        box_w = (row["bbox"][2] - row["bbox"][0]) * inputs[row["event_id"]]["low_size"][0]
        box_h = (row["bbox"][3] - row["bbox"][1]) * inputs[row["event_id"]]["low_size"][1]
        items.append({
            "candidate_id": cid, "label": lab["label"], "package": row.get("package") or row.get("population"),
            "scale": inputs[row["event_id"]]["scale"], "low_origin": inputs[row["event_id"]]["low_origin"],
            "low_box_max_px": round(max(box_w, box_h), 1),
            "recorded": row.get("recorded_binary_outcome"),
            **{f"{arm}_{m}": (p[f"{m}_decision"] or "PARSE_ERROR") for arm, p in (("low", low), ("high", high))
               for m in ("base", "adapter")},
            "low_consensus": consensus(low), "high_consensus": consensus(high),
        })

    def evaluate(subset: list[dict]) -> dict:
        scored = [i for i in subset if i["label"] in EXPECTED]
        result = {"n": len(scored)}
        for model in ("base", "adapter"):
            entry = {}
            for arm in ("low", "high"):
                entry[f"{arm}_correct"] = sum(i[f"{arm}_{model}"] == EXPECTED[i["label"]] for i in scored)
            b = sum(i[f"low_{model}"] == EXPECTED[i["label"]] and i[f"high_{model}"] != EXPECTED[i["label"]] for i in scored)
            c = sum(i[f"low_{model}"] != EXPECTED[i["label"]] and i[f"high_{model}"] == EXPECTED[i["label"]] for i in scored)
            entry.update(low_only_correct=b, high_only_correct=c, mcnemar_p=mcnemar_exact(b, c))
            result[model] = entry
        cons = {}
        for arm in ("low", "high"):
            cons[arm] = {
                "auto_correct": sum(i[f"{arm}_consensus"] == EXPECTED[i["label"]] for i in scored),
                "auto_wrong": sum(i[f"{arm}_consensus"] not in ("MANUAL", EXPECTED[i["label"]]) for i in scored),
                "manual": sum(i[f"{arm}_consensus"] == "MANUAL" for i in scored),
            }
        b = sum(i["low_consensus"] == EXPECTED[i["label"]] and i["high_consensus"] != EXPECTED[i["label"]] for i in scored)
        c = sum(i["low_consensus"] != EXPECTED[i["label"]] and i["high_consensus"] == EXPECTED[i["label"]] for i in scored)
        cons.update(low_only_correct=b, high_only_correct=c, mcnemar_p=mcnemar_exact(b, c))
        result["consensus"] = cons
        return result

    changed = [i for i in items if i["scale"] != 1]
    report = {
        "schema": "vehicle_recheck_resolution_ab_v1",
        "candidates": len(items),
        "label_counts": dict(collections.Counter(i["label"] for i in items)),
        "overall": evaluate(changed),
        "by_label": {lab: evaluate([i for i in changed if i["label"] == lab])
                     for lab in EXPECTED if any(i["label"] == lab for i in changed)},
        "by_package": {pkg: evaluate([i for i in changed if i["package"] == pkg])
                       for pkg in dict.fromkeys(i["package"] for i in changed)},
        "by_low_box_size": {
            "under_20px": evaluate([i for i in changed if i["low_box_max_px"] < 20]),
            "20_to_40px": evaluate([i for i in changed if 20 <= i["low_box_max_px"] < 40]),
            "40px_plus": evaluate([i for i in changed if i["low_box_max_px"] >= 40]),
        },
        "unclear_consensus": {arm: dict(collections.Counter(i[f"{arm}_consensus"] for i in changed if i["label"] == "看不清"))
                              for arm in ("low", "high")},
        "control_identical_input": [
            {k: i[k] for k in ("candidate_id", "low_base", "high_base", "low_adapter", "high_adapter")}
            for i in items if i["scale"] == 1
        ],
        "reproducibility_low_vs_recorded": {
            "compared": sum(1 for i in items if i["recorded"] in ("confirmed", "rejected", "uncertain")),
            "matched": sum(1 for i in items if i["recorded"] in ("confirmed", "rejected", "uncertain")
                           and {"CONFIRMED": "confirmed", "REJECTED": "rejected", "MANUAL": "uncertain"}[i["low_consensus"]] == i["recorded"]),
        },
        "parse_errors": sum(1 for i in items for k in ("low_base", "low_adapter", "high_base", "high_adapter")
                            if i[k] == "PARSE_ERROR"),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(text + "\n", encoding="utf-8")
        with (args.out / "per_candidate.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    print(text)


if __name__ == "__main__":
    main()
