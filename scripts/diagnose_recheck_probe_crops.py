#!/usr/bin/env python3
"""方案 5 探针的来源诊断：只看目标本身 / 只看周边环境 / 标准裁剪，各自能分多开。

沿用 analyze_recheck_score_probe.py 的标签、分组和探针拟合；只报折外 AUC，不做阈值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import analyze_recheck_score_probe as base

CROPS = {
    "standard(f2+f4)": ("high.npz", ["dino_cls_f2", "dino_patch_f2", "dino_cls_f4", "dino_patch_f4", "siglip_f2", "siglip_f4"]),
    "object_only(tight)": ("high_variants.npz", ["dino_cls_tight", "dino_patch_tight", "siglip_tight"]),
    "context_only(box masked)": ("high_variants.npz", ["dino_cls_ctxmask_f4", "dino_patch_ctxmask_f4", "siglip_ctxmask_f4"]),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--grouping", default="scene:0.85")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = base.load_rows(args.score_dir)
    y = np.array([r["y"] for r in rows])
    sets = np.array([r["set"] for r in rows])
    groups, extra_groups = base.scene_groups(rows, args.feature_dir, args.grouping)
    metas = base.extra_meta()
    ey_all = np.array([1 if m["human_label"] == "true" else 0 for m in metas])
    result = {"grouping": args.grouping, "crops": {}}
    for name, (fname, keys) in CROPS.items():
        data = np.load(args.feature_dir / fname)
        index = {u: i for i, u in enumerate(data["uid"])}
        matrix = np.concatenate([data[k] for k in keys], axis=1)
        x = matrix[[index[f"{r['set']}|high|{r['candidate_id']}"] for r in rows]]
        ex_all = matrix[[index[f"extra|high|{m['candidate_id']}"] for m in metas]]
        oof = np.zeros((base.REPEATS, len(rows)))
        for rep in range(base.REPEATS):
            for tr, te in GroupKFold(n_splits=base.FOLDS, shuffle=True, random_state=rep).split(x, y, groups):
                keep = np.array([g not in set(groups[te]) for g in extra_groups])
                probe, _ = base.fit_probe(np.concatenate([x[tr], ex_all[keep]]), np.concatenate([y[tr], ey_all[keep]]),
                                          np.concatenate([groups[tr], extra_groups[keep]]))
                oof[rep, te] = probe.decision_function(x[te])
        score = oof.mean(0)
        entry = {}
        for subset in ("hard", "random", "all"):
            mask = np.ones(len(rows), bool) if subset == "all" else sets == subset
            entry[subset] = {"auc": round(base.auc(y[mask], score[mask]), 3),
                             "ci95": base.event_bootstrap_auc(y[mask], score[mask], groups[mask])}
        result["crops"][name] = entry
        print(name, json.dumps(entry, ensure_ascii=False))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "crop_diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
