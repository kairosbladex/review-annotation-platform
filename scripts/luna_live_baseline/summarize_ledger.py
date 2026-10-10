#!/usr/bin/env python3
"""汇总 append-only 调用台账（脱敏）。

用 ledger_util 做**顺序配对**：保留每条尝试，重复的 (round,unit,attempt) 不会被
set 合并；在途未配对的旧尝试仍会显示；额外尝试按 (phase,unit) 计数。

用法：
  python3 scripts/luna_live_baseline/summarize_ledger.py --ledger PATH [--prior PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger_util  # noqa: E402


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def summarize(records: list[dict], title: str) -> None:
    s = ledger_util.summarize(records, hash_units=True)
    print(f"[{title}] attempt={s['attempts']} attempt_result={s['attempt_results']} "
          f"额外尝试={s['extra_attempts']} 孤儿结果={s['orphan_results']}")
    print(f"  phase: {s['by_phase']}")
    print(f"  status: {s['by_status']}")
    print(f"  在途未配对={len(s['in_flight'])}  超过每单元2次={len(s['units_over_cap'])}")
    for u in s["in_flight"][:20]:
        print(f"    IN_FLIGHT {u['phase']} unit_hash={u['unit_hash']} attempt={u['attempt']} uid={u.get('attempt_uid')}")
    for u in s["units_over_cap"][:20]:
        print(f"    OVER_CAP {u['phase']} unit_hash={u['unit_hash']} attempts={u['attempts']}")
    for r in s["failed"][:40]:
        print(f"    FAIL {r['phase']} unit_hash={r['unit_hash']} attempt={r['attempt']} status={r['status']}")
    for r in records:
        if r.get("kind") == "run_end":
            print(f"  run_end attempts_total={r.get('attempts_total')} stopped={r.get('stopped_reason')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--prior", type=Path, default=None)
    args = ap.parse_args()
    if args.prior:
        summarize(read(args.prior), f"prior {args.prior.name}")
    if args.ledger:
        summarize(read(args.ledger), f"ledger {args.ledger.name}")


if __name__ == "__main__":
    main()
