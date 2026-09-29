#!/usr/bin/env python3
"""复检方案 1（Qwen 连续分数 + 三段阈值）与方案 5（冻结特征探针）离线对比。

标签：真车=1；屋顶设施/非车=0；看不清与未标注不参与。
难例集（108 高清人工标签）与线上随机集（随机包当前已标注部分）合并，按事件分组交叉验证：
阈值和探针只在训练折上确定，测试折结果拼成全集的折外结果，再按集合分别统计。
不读取、不修改任何线上服务。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("data/vehicle-surface-gate-eval-20260929")
LABELS = {"真车": 1, "屋顶设施": 0, "非车": 0}
REPEATS = 3
FOLDS = 5
CLIP = 20.0  # 对数几率裁剪，避免极端值主导


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def logodds(ll: dict) -> float:
    return float(np.clip(ll["CONFIRMED"] - ll["REJECTED"], -CLIP, CLIP))


def load_rows(score_dir: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    for r in csv.DictReader(open(ROOT / "resolution-ab/per_candidate.csv", encoding="utf-8-sig")):
        if r["label"] in LABELS:
            rows[r["candidate_id"]] = {"candidate_id": r["candidate_id"], "event_id": r["candidate_id"].split(":")[0],
                                       "set": "hard", "y": LABELS[r["label"]], "label": r["label"],
                                       "consensus": {"low": r["low_consensus"], "high": r["high_consensus"]}}
    random_pred = defaultdict(dict)
    for p in glob.glob(str(ROOT / "resolution-ab-random/raw/run*/predictions.jsonl")):
        for r in read_jsonl(Path(p)):
            b, a = r["base_decision"], r["adapter_decision"]
            random_pred[r["candidate_id"]][r["arm"]] = b if (b == a and b) else "MANUAL"
    for r in csv.DictReader(open(ROOT / "random-traffic-hires-package/annotations.csv", encoding="utf-8-sig")):
        if r["label"] not in LABELS:
            continue
        if r["candidate_id"] in rows:  # 沿用高清标签的 6 个框：保留难例集记录，另记所属随机集
            rows[r["candidate_id"]]["also_random"] = True
            continue
        rows[r["candidate_id"]] = {"candidate_id": r["candidate_id"], "event_id": r["event_id"], "set": "random",
                                   "y": LABELS[r["label"]], "label": r["label"],
                                   "consensus": random_pred[r["candidate_id"]]}
    scores = {}
    for name in ("hard", "random"):
        for r in read_jsonl(score_dir / name / "scores.jsonl"):
            if not r.get("error"):
                scores[(r["candidate_id"], r["arm"])] = r
    for row in rows.values():
        for arm in ("low", "high"):
            s = scores.get((row["candidate_id"], arm))
            if s:
                row[f"{arm}_base"] = logodds(s["base_ll"])
                row[f"{arm}_adapter"] = logodds(s["adapter_ll"])
    return [r for r in rows.values() if all(f"{a}_{m}" in r for a in ("low", "high") for m in ("base", "adapter"))]


def greedy_agreement(score_dir: Path) -> dict:
    """教师强制分数的较大者 vs 既往生成输出，逐模型逐分支核对。"""
    recorded = {}
    for pattern in ("resolution-ab/raw/run*/predictions.jsonl", "resolution-ab-random/raw/run*/predictions.jsonl"):
        for p in glob.glob(str(ROOT / pattern)):
            for r in read_jsonl(Path(p)):
                recorded[(r["candidate_id"], r["arm"])] = r
    stats = Counter()
    mismatches = []
    for name in ("hard", "random"):
        for r in read_jsonl(score_dir / name / "scores.jsonl"):
            old = recorded.get((r["candidate_id"], r["arm"]))
            if not old or r.get("error"):
                stats["missing"] += 1
                continue
            for m in ("base", "adapter"):
                ll = r[f"{m}_ll"]
                new = "CONFIRMED" if ll["CONFIRMED"] >= ll["REJECTED"] else "REJECTED"
                if old[f"{m}_decision"] is None:
                    stats["old_invalid"] += 1
                elif new == old[f"{m}_decision"]:
                    stats["agree"] += 1
                else:
                    stats["disagree"] += 1
                    mismatches.append({"candidate_id": r["candidate_id"], "arm": r["arm"], "model": m,
                                       "recorded": old[f"{m}_decision"], "logodds": round(logodds(ll), 3)})
    return {"counts": dict(stats), "mismatches": mismatches}


def load_features(feature_dir: Path, arm: str, keys: list[str]) -> dict[str, np.ndarray]:
    data = np.load(feature_dir / f"{arm}.npz")
    matrix = np.concatenate([data[k] for k in keys], axis=1)
    return {uid: matrix[i] for i, uid in enumerate(data["uid"])}


def extra_meta() -> list[dict]:
    src = Path("data/ktd-clean-source-qwen-20260925-v1")
    return [r["meta"] for r in read_jsonl(src / "train.jsonl") + read_jsonl(src / "validation.jsonl")
            if r["meta"]["class_name"] == "车"]


def extra_training(feature_dir: Path, arm: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    feats = load_features(feature_dir, arm, keys)
    metas = extra_meta()
    return (np.array([feats[f"extra|{arm}|{m['candidate_id']}"] for m in metas]),
            np.array([1 if m["human_label"] == "true" else 0 for m in metas]))


def scene_groups(rows: list[dict], feature_dir: Path | None, grouping: str) -> tuple[np.ndarray, np.ndarray]:
    """分组口径：event=按事件；task=同一飞行任务合并；scene:<阈值>=再把跨事件外观近似重复
    （DINOv2 框 4 倍裁剪特征余弦相似度 ≥ 阈值，视为同一实物或同一场景）的事件合并。
    评测行与额外训练行放在同一个并查集里，返回两者的组号。"""
    task = {}
    for line in Path("data/ktd-test-real-monitoring-full-20260924/source_manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        task[r["event_id"]] = r.get("task_id")
    metas = extra_meta()
    events = [r["event_id"] for r in rows] + [m["source_event_id"] for m in metas]
    tasks = [task.get(r["event_id"]) for r in rows] + [m.get("source_task_id") for m in metas]
    parent = list(range(len(events)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    first = {}
    for i, (e, t) in enumerate(zip(events, tasks)):
        union(i, first.setdefault(("e", e), i))
        if grouping != "event" and t:
            union(i, first.setdefault(("t", t), i))
    if grouping.startswith("scene:"):
        threshold = float(grouping.split(":", 1)[1])
        data = np.load(feature_dir / "high.npz")
        index = {u: i for i, u in enumerate(data["uid"])}
        x = np.concatenate([data["dino_cls_f4"], data["dino_patch_f4"]], axis=1)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        keys = [f"{r['set']}|high|{r['candidate_id']}" for r in rows] + [f"extra|high|{m['candidate_id']}" for m in metas]
        sim = x[[index[k] for k in keys]] @ x[[index[k] for k in keys]].T
        for i, j in zip(*np.where(np.triu(sim, 1) >= threshold)):
            union(int(i), int(j))
    comp = np.array([find(i) for i in range(len(events))])
    return comp[:len(rows)], comp[len(rows):]


def fit_probe(x, y, groups, c_fixed=None):
    """在训练折内用分组内层交叉验证选正则强度，再全量拟合；返回 (模型, C)。"""
    best, best_c = None, c_fixed or 1.0
    n_groups = len(set(groups))
    for c in (() if c_fixed else (0.001, 0.01, 0.1, 1.0)):
        losses = []
        for tr, va in GroupKFold(n_splits=min(4, n_groups)).split(x, y, groups):
            if len(set(y[tr])) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
            model.fit(x[tr], y[tr])
            p = np.clip(model.predict_proba(x[va])[:, 1], 1e-6, 1 - 1e-6)
            losses.append(-np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)))
        loss = float(np.mean(losses)) if losses else math.inf
        if best is None or loss < best:
            best, best_c = loss, c
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, class_weight="balanced", max_iter=5000))
    model.fit(x, y)
    return model, best_c


def choose_thresholds(score, y, alpha, beta):
    """t_hi：训练折里负样本被确认比例 ≤ alpha 的最低阈值；t_lo：真车被驳回比例 ≤ beta 的最高阈值。"""
    neg = np.sort(score[y == 0])[::-1]
    pos = np.sort(score[y == 1])
    k = int(math.floor(alpha * len(neg)))
    t_hi = (neg[k] if k < len(neg) else -np.inf) + 1e-9
    j = int(math.floor(beta * len(pos)))
    t_lo = (pos[j] if j < len(pos) else np.inf) - 1e-9
    if t_lo >= t_hi:
        t_lo = t_hi - 1e-9
    return t_hi, t_lo


def decide(score, t_hi, t_lo):
    return np.where(score >= t_hi, "CONFIRMED", np.where(score <= t_lo, "REJECTED", "MANUAL"))


def tally(decisions, y) -> dict:
    out = Counter()
    for d, label in zip(decisions, y):
        if d == "MANUAL":
            out["manual"] += 1
        elif (d == "CONFIRMED") == (label == 1):
            out["auto_correct"] += 1
        elif d == "CONFIRMED":
            out["wrong_confirm"] += 1  # 假目标被自动确认
        else:
            out["wrong_reject"] += 1  # 真车被自动驳回
    return {k: out.get(k, 0) for k in ("auto_correct", "wrong_confirm", "wrong_reject", "manual")}


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def event_bootstrap_auc(y, s, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])
        if len(set(y[pick])) == 2:
            vals.append(roc_auc_score(y[pick], s[pick]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grouping", default="event", help="event | task | scene:<相似度阈值>")
    args = parser.parse_args()

    rows = load_rows(args.score_dir)
    y = np.array([r["y"] for r in rows])
    groups, extra_groups = scene_groups(rows, args.feature_dir, args.grouping)
    sets = np.array([r["set"] for r in rows])
    report = {"schema": "recheck_score_probe_v1", "n": len(rows), "grouping": args.grouping,
              "group_count": len(set(groups)), "largest_group": int(np.bincount(np.unique(groups, return_inverse=True)[1]).max()),
              "labels": {s: dict(Counter(r["label"] for r in rows if r["set"] == s)) for s in ("hard", "random")},
              "greedy_agreement": greedy_agreement(args.score_dir)}

    feature_sets = {"siglip2": ["siglip_f2", "siglip_f4"],
                    "dinov2": ["dino_cls_f2", "dino_patch_f2", "dino_cls_f4", "dino_patch_f4"]}
    feature_sets["dinov2+siglip2"] = feature_sets["dinov2"] + feature_sets["siglip2"]
    policies = {"strict": (0.0, 0.0), "loose": (0.10, 0.05)}

    per_arm = {}
    per_candidate = {r["candidate_id"]: {"candidate_id": r["candidate_id"], "set": r["set"], "label": r["label"]}
                     for r in rows}
    for arm in ("low", "high"):
        base = np.array([r[f"{arm}_base"] for r in rows])
        adapter = np.array([r[f"{arm}_adapter"] for r in rows])
        fixed_scores = {"qwen_base": base, "qwen_adapter": adapter, "qwen_mean": (base + adapter) / 2,
                        "qwen_min": np.minimum(base, adapter)}
        feats = {}
        if args.feature_dir:
            for name, keys in feature_sets.items():
                table = load_features(args.feature_dir, arm, keys)
                feats[name] = (np.stack([table[f"{r['set']}|{arm}|{r['candidate_id']}"]
                                         for r in rows]), *extra_training(args.feature_dir, arm, keys))
        method_names = list(fixed_scores) + [f"probe_{n}" for n in feats] + [f"stack_{n}" for n in feats]
        oof = {m: np.zeros((REPEATS, len(rows))) for m in method_names}
        decisions = {(m, p): np.empty((REPEATS, len(rows)), dtype=object) for m in method_names for p in policies}
        gate = {p: np.empty((REPEATS, len(rows)), dtype=object) for p in feats}
        for rep in range(REPEATS):
            splitter = GroupKFold(n_splits=FOLDS, shuffle=True, random_state=rep)
            for tr, te in splitter.split(np.zeros(len(rows)), y, groups):
                fold_scores = dict(fixed_scores)
                test_groups = set(groups[te])
                extra_keep = np.array([g not in test_groups for g in extra_groups])  # 与测试折同组的额外样本不进训练
                for name, (x, ex_all, ey_all) in feats.items():
                    ex, ey = ex_all[extra_keep], ey_all[extra_keep]
                    xtr = np.concatenate([x[tr], ex])
                    ytr = np.concatenate([y[tr], ey])
                    gtr = np.concatenate([groups[tr], extra_groups[extra_keep]])
                    probe, c = fit_probe(xtr, ytr, gtr)
                    s = np.zeros(len(rows))
                    s[te] = probe.decision_function(x[te])
                    # 训练折内的探针分数用内层折外预测，避免阈值在拟合过的数据上被选得过松
                    inner = np.zeros(len(tr))
                    for itr, ite in GroupKFold(n_splits=4).split(x[tr], y[tr], groups[tr]):
                        inner_probe, _ = fit_probe(np.concatenate([x[tr][itr], ex]), np.concatenate([y[tr][itr], ey]),
                                                   np.concatenate([groups[tr][itr], gtr[len(tr):]]), c_fixed=c)
                        inner[ite] = inner_probe.decision_function(x[tr][ite])
                    s[tr] = inner
                    fold_scores[f"probe_{name}"] = s
                    # 叠加：Qwen 两路对数几率 + 探针分数 → 逻辑回归
                    stack_x = np.stack([base, adapter, s], axis=1)
                    stacker = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(stack_x[tr], y[tr])
                    fold_scores[f"stack_{name}"] = stacker.decision_function(stack_x)
                for method, score in fold_scores.items():
                    oof[method][rep, te] = score[te]
                    for pname, (alpha, beta) in policies.items():
                        t_hi, t_lo = choose_thresholds(score[tr], y[tr], alpha, beta)
                        decisions[(method, pname)][rep, te] = decide(score[te], t_hi, t_lo)
                for name in feats:
                    # 门控：沿用两路一致规则，但一致确认还需探针分数高于阈值（只拦截，不新增确认）
                    s = fold_scores[f"probe_{name}"]
                    t_hi, _ = choose_thresholds(s[tr], y[tr], 0.0, 0.0)
                    cons = np.array([r["consensus"].get(arm, "MANUAL") for r in rows])
                    g = np.where((cons == "CONFIRMED") & (s < t_hi), "MANUAL", cons)
                    gate[name][rep, te] = g[te]

        arm_report = {}
        cons = np.array([r["consensus"].get(arm, "MANUAL") for r in rows])
        for subset in ("hard", "random", "all"):
            mask = np.ones(len(rows), bool) if subset == "all" else sets == subset
            sub = {"n": int(mask.sum()), "positives": int(y[mask].sum()), "negatives": int((1 - y[mask]).sum()),
                   "baseline_consensus": tally(cons[mask], y[mask]), "methods": {}}
            for method in method_names:
                score = oof[method].mean(0)
                entry = {"auc": auc(y[mask], score[mask])}
                if entry["auc"] is not None:
                    entry["auc_ci95_event_bootstrap"] = event_bootstrap_auc(y[mask], score[mask], groups[mask])
                    neg_max = score[mask][y[mask] == 0].max()
                    entry["positives_above_all_negatives"] = int((score[mask][y[mask] == 1] > neg_max).sum())
                for pname in policies:
                    reps = [tally(decisions[(method, pname)][rep][mask], y[mask]) for rep in range(REPEATS)]
                    entry[pname] = {k: round(float(np.mean([t[k] for t in reps])), 1) for k in reps[0]}
                    entry[pname]["range_wrong_confirm"] = [min(t["wrong_confirm"] for t in reps),
                                                           max(t["wrong_confirm"] for t in reps)]
                sub["methods"][method] = entry
            for name in feats:
                reps = [tally(gate[name][rep][mask], y[mask]) for rep in range(REPEATS)]
                sub["methods"][f"gate_consensus_and_probe_{name}"] = {
                    "strict": {k: round(float(np.mean([t[k] for t in reps])), 1) for k in reps[0]}}
            arm_report[subset] = sub
        per_arm[arm] = arm_report
        for i, r in enumerate(rows):
            for method in method_names:
                per_candidate[r["candidate_id"]][f"{arm}_{method}"] = round(float(oof[method][:, i].mean()), 4)
    report["arms"] = per_arm
    report["protocol"] = {"repeats": REPEATS, "folds": FOLDS, "grouping": args.grouping,
                          "policies": {k: {"max_wrong_confirm_rate_train": a, "max_wrong_reject_rate_train": b}
                                       for k, (a, b) in policies.items()},
                          "extra_training": "clean-source Qwen 训练/验证集中的车辆候选（86 个，与评测事件无交集），只进训练折",
                          "score": "Qwen 为教师强制对数几率 ll(CONFIRMED)-ll(REJECTED)，裁剪到 ±20"}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with open(args.out / "per_candidate.csv", "w", newline="", encoding="utf-8-sig") as handle:
        fields = sorted({k for v in per_candidate.values() for k in v}, key=lambda k: (k not in ("candidate_id", "set", "label"), k))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_candidate.values())
    print(json.dumps({"n": report["n"], "labels": report["labels"], "greedy": report["greedy_agreement"]["counts"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
