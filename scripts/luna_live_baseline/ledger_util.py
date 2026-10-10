#!/usr/bin/env python3
"""调用台账的顺序配对与统计。

关键修正：不能把 (phase, unit, attempt) 组成的 set 用来配对——进程重启后
同一单元会再次出现 attempt=0，set 会把“在途未配对”的旧尝试误当成已配对。
本模块按**时间顺序**把 `attempt` 与随后的 `attempt_result` 做 FIFO 配对，
保留每一条尝试；额外尝试按 (phase, unit) 计数（超过 1 次的次数）。
"""
from __future__ import annotations

import collections
import hashlib


def unit_hash(u) -> str:
    return hashlib.sha256(str(u).encode("utf-8")).hexdigest()[:16]


def _key(r: dict):
    return (r.get("phase"), r.get("round"), r.get("unit"), r.get("attempt"))


def summarize(records: list[dict], hash_units: bool = True) -> dict:
    pending = collections.defaultdict(collections.deque)
    calls: list[dict] = []
    by_uid: dict[str, dict] = {}
    orphan_results = 0
    for r in records:
        kind = r.get("kind")
        if kind == "attempt":
            entry = {"phase": r.get("phase"), "round": r.get("round"), "unit": r.get("unit"),
                     "attempt": r.get("attempt"), "attempt_uid": r.get("attempt_uid"),
                     "status": None, "at": r.get("at")}
            pending[_key(r)].append(entry)
            if r.get("attempt_uid"):
                by_uid[r["attempt_uid"]] = entry
            calls.append(entry)
        elif kind == "attempt_result":
            entry = None
            uid = r.get("attempt_uid")
            if uid and uid in by_uid and by_uid[uid]["status"] is None:
                entry = by_uid[uid]
                try:
                    pending[_key(r)].remove(entry)
                except ValueError:
                    pass
            if entry is None and not uid:
                dq = pending[_key(r)]
                if dq:
                    entry = dq.popleft()
            if entry is not None:
                entry["status"] = r.get("status")
                entry["result_at"] = r.get("at")
            else:
                orphan_results += 1

    per_unit = collections.Counter((c["phase"], c["unit"]) for c in calls)
    extra_attempts = sum(max(0, n - 1) for n in per_unit.values())
    over_cap = sorted(
        ({"phase": p, "unit": u, "attempts": n} for (p, u), n in per_unit.items() if n > 2),
        key=lambda x: -x["attempts"],
    )

    def view(c: dict) -> dict:
        d = {"phase": c["phase"], "round": c["round"], "attempt": c["attempt"],
             "attempt_uid": c["attempt_uid"], "status": c["status"] if c["status"] is not None else "in_flight"}
        if hash_units:
            d["unit_hash"] = unit_hash(c["unit"])
        else:
            d["unit"] = c["unit"]
        return d

    return {
        "attempts": len(calls),
        "attempt_results": sum(1 for c in calls if c["status"] is not None),
        "orphan_results": orphan_results,
        "extra_attempts": extra_attempts,
        "by_phase": dict(collections.Counter(c["phase"] for c in calls)),
        "by_status": dict(collections.Counter(
            (c["status"] if c["status"] is not None else "in_flight") for c in calls)),
        "in_flight": [view(c) for c in calls if c["status"] is None],
        "failed": [view(c) for c in calls if c["status"] not in (None, "ok")],
        "units_over_cap": [
            {"phase": x["phase"], "unit_hash": unit_hash(x["unit"]), "attempts": x["attempts"]}
            if hash_units else x for x in over_cap
        ],
        "calls": [view(c) for c in calls],
    }
