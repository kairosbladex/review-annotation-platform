#!/usr/bin/env python3
"""用冻结官方计分对 baseline 两轮结果打分（不调用模型）。

输入：
  --labels   冻结 labels.json（dev 685 / holdout 443）
  --items    冻结 items.json（用于 key→图 聚类）
  --results  evaluate.py 的输出目录（含 decisions/round*.json 与 units/round*.jsonl）
  --official 冻结官方 metrics.py（默认 frozen/metrics_official.py）

产出：
  metrics.json        机器可读指标
  report_tables.md    中文表格（覆盖、各类、交叉表、翻转、聚类区间）

指标口径：
  * accuracy 用官方 metrics.score（只在成功返回的框上算，失败不计对错）；
  * 覆盖率单独报告，失败单独列出，绝不静默剔除；
  * 真目标召回率 = (gold_true - true_rejected) / gold_true；
  * 非目标误放率 = false_released / gold_false；
  * uncertain 交叉表 = 金标真假 × 模型 true/false/uncertain；
  * 翻转率 = 两轮同一候选结论变化比例（decision：true/非true；raw：三值任意变化）；
  * 聚类区间 = 按图（group=unit）有放回重采样 1000 次的 95% 分位区间。
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
from pathlib import Path

DEFAULT_FROZEN = Path("data/luna-live-baseline/private/frozen")
SPLITS = ("dev", "holdout")


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pct(x):
    return None if x is None else round(x, 4)


def cross_tab(labels, decisions, keys):
    """返回金标×结论交叉计数与派生指标。"""
    tab = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)
    for k in keys:
        gold = labels[k]["label"] == "true"
        d = decisions.get(k)
        slot = d if d in ("true", "false", "uncertain") else "missing"
        tab[("gold_true" if gold else "gold_false", slot)] += 1
        per_class[labels[k]["class_name"]][("gold_true" if gold else "gold_false", slot)] += 1
    return tab, per_class


def derive(counter):
    gt_t, gt_f, gt_u = counter[("gold_true", "true")], counter[("gold_true", "false")], counter[("gold_true", "uncertain")]
    gf_t, gf_f, gf_u = counter[("gold_false", "true")], counter[("gold_false", "false")], counter[("gold_false", "uncertain")]
    gold_true = gt_t + gt_f + gt_u
    gold_false = gf_t + gf_f + gf_u
    n = gold_true + gold_false
    right = gt_t + gf_f + gf_u  # 只有 true 放过：true 且金标是，或非 true 且金标不是
    true_rejected = gt_f + gt_u
    false_released = gf_t
    uncertain = gt_u + gf_u
    return {
        "n": n,
        "gold_true": gold_true,
        "gold_false": gold_false,
        "right": right,
        "accuracy": pct(right / n) if n else None,
        "true_recall": pct(gt_t / gold_true) if gold_true else None,
        "true_rejected": true_rejected,
        "false_release_rate": pct(false_released / gold_false) if gold_false else None,
        "false_released": false_released,
        "uncertain": uncertain,
        "uncertain_rate": pct(uncertain / n) if n else None,
        "cross": {
            "gold_true:true": gt_t, "gold_true:false": gt_f, "gold_true:uncertain": gt_u,
            "gold_false:true": gf_t, "gold_false:false": gf_f, "gold_false:uncertain": gf_u,
        },
    }


def _bootstrap(labels, decisions, keys, clusters_for_key, iters, seed):
    by_cluster = collections.defaultdict(list)
    for k in keys:
        c = clusters_for_key.get(k)
        if c is not None and k in decisions:
            by_cluster[c].append(k)
    cluster_ids = sorted(by_cluster)
    if not cluster_ids:
        return {"iters": iters, "n_clusters": 0, "n_boxes": 0, "accuracy_ci95": None, "accuracy_median": None}
    rng = random.Random(seed)
    accs = []
    for _ in range(iters):
        right = tot = 0
        for _ in cluster_ids:
            c = cluster_ids[rng.randrange(len(cluster_ids))]
            for k in by_cluster[c]:
                tot += 1
                right += (decisions[k] == "true") == (labels[k]["label"] == "true")
        accs.append(right / tot if tot else 0.0)
    accs.sort()
    lo = accs[int(0.025 * (iters - 1))]
    hi = accs[int(0.975 * (iters - 1))]
    return {"iters": iters, "n_clusters": len(cluster_ids),
            "n_boxes": sum(len(v) for v in by_cluster.values()),
            "accuracy_ci95": [round(lo, 4), round(hi, 4)],
            "accuracy_median": round(accs[len(accs) // 2], 4)}


def cluster_bootstrap(labels, decisions, keys, key_to_unit, iters=1000, seed=20261011):
    """按图（unit）有放回重采样。**未解决同航次相关，不能据此证明泛化提升。**"""
    r = _bootstrap(labels, decisions, keys, {k: key_to_unit[k] for k in keys}, iters, seed)
    r["level"] = "image(unit)"
    return r


def cluster_bootstrap_groups(labels, decisions, keys, key_to_unit, unit_to_group, iters=1000, seed=20261011):
    """按**已知航次/任务分组**对可映射子集做有放回重采样，并报告覆盖率。

    未知分组的图被排除，不静默当成独立；覆盖率单独给出。
    """
    clusters = {k: unit_to_group.get(key_to_unit[k]) for k in keys}
    r = _bootstrap(labels, decisions, keys, clusters, iters, seed)
    r["level"] = "group_known_subset"
    covered_units = {key_to_unit[k] for k in keys if unit_to_group.get(key_to_unit[k]) is not None}
    covered_boxes = [k for k in keys if unit_to_group.get(key_to_unit[k]) is not None]
    r["coverage"] = {
        "units_covered": len(covered_units),
        "units_total": len({key_to_unit[k] for k in keys}),
        "boxes_covered": len(covered_boxes),
        "boxes_total": len(keys),
        "groups": len({unit_to_group[key_to_unit[k]] for k in covered_boxes}),
    }
    return r


def flips(a, b):
    keys = sorted(set(a) & set(b))
    raw = sum(a[k] != b[k] for k in keys)
    dec = sum((a[k] == "true") != (b[k] == "true") for k in keys)
    return {"compared": len(keys), "raw_flips": raw, "raw_flip_rate": pct(raw / len(keys)) if keys else None,
            "decision_flips": dec, "decision_flip_rate": pct(dec / len(keys)) if keys else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=DEFAULT_FROZEN / "labels.json")
    ap.add_argument("--items", type=Path, default=DEFAULT_FROZEN / "items.json")
    ap.add_argument("--official", type=Path, default=Path("scripts/luna_live_baseline/metrics_official.py"))
    ap.add_argument("--results", type=Path, default=Path("data/luna-live-baseline/private/results/formal"))
    ap.add_argument("--out", type=Path, default=Path("data/luna-live-baseline/private/results/formal/metrics.json"))
    ap.add_argument("--report", type=Path, default=Path("data/luna-live-baseline/private/results/formal/report_tables.md"))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--splits", default="dev,holdout",
                    help="要计分的划分；协调方更正后正式范围只跑 holdout")
    ap.add_argument("--grouping", type=Path, default=None,
                    help="私有映射 unit->已核实 task/航次哈希 或 null；提供时增加已知航次子集 CI")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    items = json.loads(args.items.read_text(encoding="utf-8"))
    official = load_official(args.official)
    key_to_unit = {c["key"]: u["unit"] for u in items for c in u["candidates"]}
    unit_to_group = json.loads(args.grouping.read_text(encoding="utf-8")) if args.grouping else None

    decisions = {}
    units_log = {}
    for r in range(1, args.rounds + 1):
        decisions[r] = json.loads((args.results / "decisions" / f"round{r}.json").read_text(encoding="utf-8"))
        units_log[r] = [json.loads(l) for l in (args.results / "units" / f"round{r}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    groups = {s: [k for k, v in labels.items() if v["split"] == s] for s in splits}
    groups["all"] = [k for s in splits for k in groups[s]]

    out = {"labels_total": len(labels), "rounds": {}, "splits": {}}
    for s in splits:
        out["splits"][s] = {"n_labeled": len(groups[s])}

    # 官方计分交叉核对
    official_crosscheck = []
    for r in range(1, args.rounds + 1):
        rnd = {"decisions_total": len(decisions[r]), "splits": {}}
        for s in splits:
            off = official.score(decisions[r], labels, s)
            keys = [k for k in groups[s] if k in decisions[r]]
            tab, per_class = cross_tab(labels, decisions[r], groups[s])
            d = derive(tab)
            d["coverage"] = pct(len(keys) / len(groups[s]))
            d["failed"] = len(groups[s]) - len(keys)
            off_acc = off["overall"]["accuracy"]
            official_crosscheck.append({"round": r, "split": s, "official_accuracy": off_acc,
                                        "recomputed_accuracy": d["accuracy"],
                                        "match": off_acc == d["accuracy"]})
            d["by_class"] = {cls: derive(c) for cls, c in sorted(per_class.items(), key=lambda x: -sum(x[1].values()))}
            d["bootstrap"] = cluster_bootstrap(labels, decisions[r], groups[s], key_to_unit)
            if unit_to_group is not None:
                d["bootstrap_group_known"] = cluster_bootstrap_groups(
                    labels, decisions[r], groups[s], key_to_unit, unit_to_group)
            d["official"] = off["overall"]
            rnd["splits"][s] = d
        # all
        keys = [k for s in splits for k in groups[s]]
        tab, per_class = cross_tab(labels, decisions[r], keys)
        d = derive(tab)
        d["coverage"] = pct(sum(1 for k in keys if k in decisions[r]) / len(keys))
        d["failed"] = sum(1 for k in keys if k not in decisions[r])
        d["by_class"] = {cls: derive(c) for cls, c in sorted(per_class.items(), key=lambda x: -sum(x[1].values()))}
        d["bootstrap"] = cluster_bootstrap(labels, decisions[r], keys, key_to_unit)
        if unit_to_group is not None:
            d["bootstrap_group_known"] = cluster_bootstrap_groups(
                labels, decisions[r], keys, key_to_unit, unit_to_group)
        rnd["splits"]["all"] = d
        failed_units = [u for u in units_log[r] if not u["success"]]
        rnd["failed_units"] = failed_units
        rnd["units"] = len(units_log[r])
        rnd["units_failed"] = len(failed_units)
        rnd["attempts"] = sum(u["attempts"] for u in units_log[r])
        out["rounds"][str(r)] = rnd

    out["official_crosscheck"] = official_crosscheck
    out["cluster_meta"] = {
        "image_level": "按图（unit）聚类；未解决同航次相关，不能用它证明泛化提升。",
        "group_known_subset": (
            "按已核实 task/航次哈希对可映射子集聚类；未知分组的图被排除，覆盖率见各轮 "
            "bootstrap_group_known.coverage。"
        ),
        "grouping_provided": unit_to_group is not None,
    }
    # 两轮比较
    out["round_comparison"] = {}
    for s in ["all"] + splits:
        a, b = decisions[1], decisions[2]
        keys = groups[s]
        common = {k: a[k] for k in keys if k in a and k in b}
        common_b = {k: b[k] for k in common}

        def acc(dec, ks):
            if not ks:
                return None
            right = sum((dec[k] == "true") == (labels[k]["label"] == "true") for k in ks)
            return pct(right / len(ks))

        out["round_comparison"][s] = flips(common, common_b)
        out["round_comparison"][s]["r1_accuracy"] = out["rounds"]["1"]["splits"][s]["accuracy"]
        out["round_comparison"][s]["r2_accuracy"] = out["rounds"]["2"]["splits"][s]["accuracy"]
        # 两轮共同覆盖子集口径：只有两轮都成功返回的框才纳入比较，避免覆盖差造成假差异。
        out["round_comparison"][s]["common_coverage"] = {
            "n_common": len(common),
            "n_r1": sum(1 for k in keys if k in a),
            "n_r2": sum(1 for k in keys if k in b),
            "n_denominator": len(keys),
            "common_is_full": len(common) == len(keys),
            "r1_accuracy_common": acc(a, common),
            "r2_accuracy_common": acc(b, common),
        }

    out["coverage"] = {s: {str(r): out["rounds"][str(r)]["splits"][s]["coverage"] for r in range(1, args.rounds + 1)}
                       for s in ["all"] + splits}
    out["all_failures"] = {str(r): out["rounds"][str(r)]["failed_units"] for r in range(1, args.rounds + 1)}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {args.out}")

    # ---- 中文表格 ----
    def pc(v):
        return "—" if v is None else f"{v:.2%}"

    lines = ["# Luna 当前版本 baseline 两轮计分表（脱敏）", ""]
    lines.append("口径：只有 true 放过；false/uncertain 都算驳回；accuracy 只在成功返回的框上计算，失败单列。")
    lines.append("")
    for r in range(1, args.rounds + 1):
        lines.append(f"## 第 {r} 轮")
        lines.append("")
        lines.append("| 划分 | 冻结框 | 有效框 | 覆盖 | 金标是 | 金标不是 | 准确率 | 真目标召回 | 非目标误放率 | 真目标驳回 | 非目标放过 | uncertain |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in ["all"] + splits:
            d = out["rounds"][str(r)]["splits"][s]
            lines.append(f"| {s} | {d['n'] + d['failed']} | {d['n']} | {pc(d['coverage'])} | {d['gold_true']} | "
                         f"{d['gold_false']} | {pc(d['accuracy'])} | {pc(d['true_recall'])} | "
                         f"{pc(d['false_release_rate'])} | {d['true_rejected']} | {d['false_released']} | {d['uncertain']} |")
        lines.append("")
        # 逐类
        lines.append(f"逐类（{'+'.join(splits)}）：")
        lines.append("")
        lines.append("| 类别 | 框数 | 金标是 | 金标不是 | 准确率 | 真目标召回 | 非目标误放率 | uncertain |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        d = out["rounds"][str(r)]["splits"]["all"]
        for cls, c in d["by_class"].items():
            lines.append(f"| {cls} | {c['n']} | {c['gold_true']} | {c['gold_false']} | {pc(c['accuracy'])} | "
                         f"{pc(c['true_recall'])} | {pc(c['false_release_rate'])} | {c['uncertain']} |")
        lines.append("")
        # 金标 × 结论交叉表（含 uncertain）
        lines.append(f"金标 × 结论交叉表（{'+'.join(splits)}）：")
        lines.append("")
        lines.append("| 金标 | 判 true | 判 false | 判 uncertain |")
        lines.append("|---|---:|---:|---:|")
        cross = d["cross"]
        for gold in ("gold_true", "gold_false"):
            lines.append(f"| {gold} | {cross[gold + ':true']} | {cross[gold + ':false']} | {cross[gold + ':uncertain']} |")
        lines.append("")
        # 逐类交叉表（仅大类别，避免刷屏）
        lines.append("逐类交叉表（框数 ≥ 10 的类别）：")
        lines.append("")
        lines.append("| 类别 | 金标是:true | 金标是:false/unc | 金标不是:true | 金标不是:false/unc |")
        lines.append("|---|---:|---:|---:|---:|")
        for cls, c in d["by_class"].items():
            if c["n"] < 10:
                continue
            cc = c["cross"]
            lines.append(f"| {cls} | {cc['gold_true:true']} | {cc['gold_true:false'] + cc['gold_true:uncertain']} | "
                         f"{cc['gold_false:true']} | {cc['gold_false:false'] + cc['gold_false:uncertain']} |")
        lines.append("")
    lines.append("## 两轮一致性 / 翻转")
    lines.append("")
    lines.append("| 划分 | 同候选 | 三值翻转 | 三值翻转率 | 真假翻转 | 真假翻转率 | 第1轮准确率 | 第2轮准确率 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in ["all"] + splits:
        c = out["round_comparison"][s]
        lines.append(f"| {s} | {c['compared']} | {c['raw_flips']} | {pc(c['raw_flip_rate'])} | {c['decision_flips']} | "
                     f"{pc(c['decision_flip_rate'])} | {pc(c['r1_accuracy'])} | {pc(c['r2_accuracy'])} |")
    lines.append("")
    lines.append("### 两轮共同覆盖子集口径（避免覆盖差造成假差异）")
    lines.append("")
    lines.append("| 划分 | 共同框 | 分母 | 全覆盖 | 共同子集 第1轮 | 共同子集 第2轮 |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for s in ["all"] + splits:
        cc = out["round_comparison"][s]["common_coverage"]
        lines.append(f"| {s} | {cc['n_common']} | {cc['n_denominator']} | "
                     f"{'是' if cc['common_is_full'] else '否'} | {pc(cc['r1_accuracy_common'])} | "
                     f"{pc(cc['r2_accuracy_common'])} |")
    lines.append("")
    lines.append("## 聚类 95% 分位区间（有放回重采样 1000 次）")
    lines.append("")
    lines.append("> 按图（unit）聚类**未解决同航次相关**，不能据此证明泛化提升；"
                 "已知航次子集另列并标注覆盖率。")
    lines.append("")
    lines.append("| 轮次 | 划分 | 层级 | 聚类数 | 框数 | 准确率中位 | 95%区间 |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for r in range(1, args.rounds + 1):
        for s in ["all"] + splits:
            b = out["rounds"][str(r)]["splits"][s]["bootstrap"]
            ci = b["accuracy_ci95"]
            lines.append(f"| {r} | {s} | 按图 | {b['n_clusters']} | {b['n_boxes']} | {pc(b['accuracy_median'])} | "
                         f"{'—' if ci is None else f'[{ci[0]:.2%}, {ci[1]:.2%}]'} |")
            gb = out["rounds"][str(r)]["splits"][s].get("bootstrap_group_known")
            if gb is not None:
                gci = gb["accuracy_ci95"]
                cov = gb["coverage"]
                lines.append(f"| {r} | {s} | 已核实航次子集 | {gb['n_clusters']} | {gb['n_boxes']} | "
                             f"{pc(gb['accuracy_median'])} | "
                             f"{'—' if gci is None else f'[{gci[0]:.2%}, {gci[1]:.2%}]'} |")
                lines.append(f"| {r} | {s} | ↳ 覆盖率 | 图 {cov['units_covered']}/{cov['units_total']} | "
                             f"框 {cov['boxes_covered']}/{cov['boxes_total']} | 分组 {cov['groups']} | — |")
    lines.append("")
    lines.append("## 失败与覆盖")
    lines.append("")
    lines.append("| 轮次 | 图数 | 失败图 | 请求尝试 |")
    lines.append("|---|---:|---:|---:|")
    for r in range(1, args.rounds + 1):
        rr = out["rounds"][str(r)]
        lines.append(f"| {r} | {rr['units']} | {rr['units_failed']} | {rr['attempts']} |")
    lines.append("")
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {args.report}")
    if not all(x["match"] for x in out["official_crosscheck"]):
        print("警告：重算准确率与官方计分不一致")
        raise SystemExit(1)
    print("官方计分交叉核对：全部一致")


if __name__ == "__main__":
    main()
