#!/usr/bin/env python3
"""用人员标注包的人工标签给两种复检回放打分（旧版单图 / 新版原图 + Qwen）。"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

GOLD = {"真目标": True, "误检": False}


def load_decisions(out_dir: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(out_dir.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        for cid, res in zip(item["candidate_ids"], item["response"]["results"]):
            cs = res.get("cleanSource") or {}
            rows[cid] = {"decision": res["decision"], "reason": res.get("reasonCode"),
                         "base": cs.get("baseDecision"), "adapter": cs.get("adapterDecision")}
    return rows


def score(labels: dict[str, bool], decisions: dict[str, dict], key=lambda d: d["decision"]) -> dict:
    c = collections.Counter()
    for cid, truth in labels.items():
        d = key(decisions[cid])
        c[("真人" if truth else "非人", d)] += 1
    true_n = sum(v for (g, _), v in c.items() if g == "真人")
    false_n = sum(v for (g, _), v in c.items() if g == "非人")
    right = c[("真人", "confirmed")] + c[("非人", "rejected")]
    wrong = c[("真人", "rejected")] + c[("非人", "confirmed")]
    manual = sum(v for (g, d), v in c.items() if d not in ("confirmed", "rejected"))
    return {"n": true_n + false_n, "right": right, "wrong": wrong, "manual": manual,
            "real_person": {"total": true_n, "confirmed": c[("真人", "confirmed")], "wrongly_rejected": c[("真人", "rejected")],
                            "manual": true_n - c[("真人", "confirmed")] - c[("真人", "rejected")]},
            "not_person": {"total": false_n, "rejected": c[("非人", "rejected")], "wrongly_confirmed": c[("非人", "confirmed")],
                           "manual": false_n - c[("非人", "rejected")] - c[("非人", "confirmed")]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.annotations.open(encoding="utf-8-sig")))
    labels = {r["candidate_id"]: GOLD[r["label"]] for r in rows if r["label"] in GOLD}
    single, clean = load_decisions(args.single), load_decisions(args.clean)

    def consensus(d: dict) -> str:
        # 假设开启人员自动确认：base 与 adapter 一致时直接采用
        if d["base"] and d["base"] == d["adapter"]:
            return {"CONFIRMED": "confirmed", "REJECTED": "rejected"}[d["base"]]
        return "uncertain"

    report = {
        "labels": collections.Counter(r["label"] for r in rows),
        "single_image_old": score(labels, single),
        "clean_source_current_config": score(labels, clean),
        "clean_source_if_person_auto_confirm": score(labels, clean, consensus),
        "clean_reasons": collections.Counter(clean[c]["reason"] for c in labels),
        "single_reasons": collections.Counter(single[c]["reason"] for c in labels),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=dict), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1, default=dict))


if __name__ == "__main__":
    main()
