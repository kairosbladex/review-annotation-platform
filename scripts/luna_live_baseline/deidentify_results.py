#!/usr/bin/env python3
"""把私有原始结果转成可入库的**去标识**产物。

输入（私有目录）：
  <formal>/decisions/round*.json      框 key（含业务标识）-> 结论
  <formal>/units/round*.jsonl         图 unit -> 状态
  <formal>/logs/formal_ledger.jsonl   调用台账
  <formal>/metrics.json               完整指标（含 unit 字段）
  <formal>/report_tables.md           聚合表
  <prior>                             正式范围外调用台账

输出（可入库）：
  deidentified/round*.json            {sha256(key)[:16]: decision}
  ledger_summary.json                 计数/状态；失败与在途只保留 unit 哈希
  metrics_aggregate.json              指标（unit 字段全部哈希化）
  report_tables.md                    聚合表原样
  prior_summary.json                  范围外调用计数（无 unit）

用法：
  python3 scripts/luna_live_baseline/deidentify_results.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger_util  # noqa: E402


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def redact(obj):
    """递归把任何键名为 unit 的值替换为哈希。"""
    if isinstance(obj, dict):
        return {k: (h(v) if k == "unit" and isinstance(v, str) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return obj


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def ledger_summary(records: list[dict]) -> dict:
    return ledger_util.summarize(records, hash_units=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", type=Path, default=Path("data/luna-live-baseline/private/results/formal"))
    ap.add_argument("--prior", type=Path, default=Path("data/luna-live-baseline/private/prior_calls.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/luna-live-baseline/results"))
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "deidentified").mkdir(parents=True, exist_ok=True)

    for r in range(1, args.rounds + 1):
        dec_path = args.formal / "decisions" / f"round{r}.json"
        if not dec_path.exists():
            print(f"跳过缺失的 {dec_path}")
            continue
        dec = json.loads(dec_path.read_text(encoding="utf-8"))
        hashed = {h(k): v for k, v in dec.items()}
        (args.out / "deidentified" / f"round{r}.json").write_text(
            json.dumps(hashed, ensure_ascii=False, indent=1), encoding="utf-8")

    led = read_jsonl(args.formal / "logs" / "formal_ledger.jsonl")
    (args.out / "ledger_summary.json").write_text(
        json.dumps(ledger_summary(led), ensure_ascii=False, indent=1), encoding="utf-8")

    metrics_path = args.formal / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        (args.out / "metrics_aggregate.json").write_text(
            json.dumps(redact(m), ensure_ascii=False, indent=1), encoding="utf-8")
    report_path = args.formal / "report_tables.md"
    if report_path.exists():
        (args.out / "report_tables.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    prior = read_jsonl(args.prior)
    (args.out / "prior_summary.json").write_text(
        json.dumps(ledger_summary(prior), ensure_ascii=False, indent=1), encoding="utf-8")

    # 去标识后自检：不得出现任何原始 32 位业务 unit（用私有的 unit 集合比对）。
    items_path = Path("data/luna-live-baseline/private/frozen/items.json")
    if items_path.exists():
        units = {u["unit"] for u in json.loads(items_path.read_text(encoding="utf-8"))}
        leaked = []
        for p in args.out.rglob("*"):
            if p.is_file() and p.suffix in (".json", ".md", ".jsonl"):
                text = p.read_text(encoding="utf-8")
                leaked += [u for u in units if u in text]
        if leaked:
            raise SystemExit(f"去标识自检失败：发现原始 unit {sorted(set(leaked))[:5]}")
        print("去标识自检通过：可入库产物中未出现原始 unit。")
    print(f"去标识产物写入 {args.out}")


if __name__ == "__main__":
    main()
