"""复检微调模型候选与 v1 的对比：按固定合格标准逐条判定，供自动迭代 Agent 直接引用。

输入为 analyze_resolution_ab.py 输出的 per_candidate.csv（每行一个候选框，含标签与各组两路一致结论）。
只使用高清输入组（high），口径与线上“两路一致才自动下结论”相同。
"""

from __future__ import annotations

import csv
from math import comb
from pathlib import Path

NEGATIVE_LABELS = {"非车", "屋顶设施"}

# 合格标准：改动需人工审核，Agent 不可修改
GATES = {
    "random_real_confirmed_min_drop": 2,   # 随机流量真车自动确认 ≥ v1 − 2
    "hard_real_confirmed_min_drop": 3,     # 楼顶难例真车自动确认 ≥ v1 − 3
}


def _read(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle) if row.get("scale") != "1"}


def _mcnemar(worse: int, better: int) -> float | None:
    n = worse + better
    if n == 0:
        return None
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(worse, better) + 1)) / 2 ** n)


def _set_stats(v1: dict[str, dict], cand: dict[str, dict]) -> dict:
    if set(v1) != set(cand):
        raise ValueError("候选与 v1 的评测样本不一致")
    real = [cid for cid, row in v1.items() if row["label"] == "真车"]
    neg = [cid for cid, row in v1.items() if row["label"] in NEGATIVE_LABELS]

    def confirmed(rows, cid):
        return rows[cid]["high_consensus"] == "CONFIRMED"

    def manual(rows, cid):
        return rows[cid]["high_consensus"] == "MANUAL"

    real_worse = [cid for cid in real if confirmed(v1, cid) and not confirmed(cand, cid)]
    real_better = [cid for cid in real if not confirmed(v1, cid) and confirmed(cand, cid)]
    fp_better = [cid for cid in neg if confirmed(v1, cid) and not confirmed(cand, cid)]
    fp_worse = [cid for cid in neg if not confirmed(v1, cid) and confirmed(cand, cid)]
    return {
        "real_total": len(real),
        "real_auto_confirmed": {"v1": sum(confirmed(v1, c) for c in real), "candidate": sum(confirmed(cand, c) for c in real)},
        "real_to_manual": {"v1": sum(manual(v1, c) for c in real), "candidate": sum(manual(cand, c) for c in real)},
        "real_paired": {"worse": len(real_worse), "better": len(real_better),
                        "p_value": _mcnemar(len(real_worse), len(real_better)), "worse_ids": real_worse},
        "negative_total": len(neg),
        "negative_released": {"v1": sum(confirmed(v1, c) for c in neg), "candidate": sum(confirmed(cand, c) for c in neg)},
        "negative_paired": {"better": len(fp_better), "worse": len(fp_worse),
                            "p_value": _mcnemar(len(fp_worse), len(fp_better)), "worse_ids": fp_worse},
        "negative_still_released_ids": [c for c in neg if confirmed(cand, c)],
    }


def compare_to_v1(v1_random: Path, cand_random: Path, v1_hard: Path, cand_hard: Path, parse_errors: int) -> dict:
    random = _set_stats(_read(v1_random), _read(cand_random))
    hard = _set_stats(_read(v1_hard), _read(cand_hard))
    gates = {
        "1_random_real_confirmed": random["real_auto_confirmed"]["candidate"]
        >= random["real_auto_confirmed"]["v1"] - GATES["random_real_confirmed_min_drop"],
        "2_random_negative_released_lower": random["negative_released"]["candidate"] < random["negative_released"]["v1"]
        and random["negative_paired"]["better"] > random["negative_paired"]["worse"],
        "3_hard_real_confirmed": hard["real_auto_confirmed"]["candidate"]
        >= hard["real_auto_confirmed"]["v1"] - GATES["hard_real_confirmed_min_drop"],
        "4_no_parse_errors": parse_errors == 0,
    }
    return {
        "input_arm": "high（高清输入，两路一致口径）",
        "random_traffic_200": random,
        "rooftop_hard_108": hard,
        "parse_errors": parse_errors,
        "gates": gates,
        "passed_all_gates": all(gates.values()),
        "failed_gates": [name for name, ok in gates.items() if not ok],
        "note": "p_value 为配对精确检验；样本量小，p 值大于 0.05 时不能认为差异确定。",
    }
