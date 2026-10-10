#!/usr/bin/env python3
"""隔离的 Luna 当前版本 baseline 执行器（客户端，v2）。

范围（协调方 2026-10-11 更正后）：只跑历史 holdout 137 图 / 443 有标签框（全部候选 452），
两轮共 274 个请求；每个失败请求最多重试 1 次。不跑 dev。

关键约束：
  * 并发 <= 2；
  * 每次尝试写**持久台账**（append-only + fsync），崩溃不丢已用预算；
  * 断点恢复（--resume）跳过已成功图，且**不重置**重试上限与总预算（从台账累计）；
  * STOP 文件存在即停；每 `--resource-interval` 秒检查整机可用内存与共享服务健康，
    可用 < 15% 或共享不健康则写 STOP 并停止自己的任务；
  * 每图全部候选一次请求，target_index=0..n-1，返回后校验映射；
  * 结果分两份：`raw/`（含模型自由文本，留服务器）与 `decisions/`、`units/`（脱敏）。

本脚本只通过 --token-file 读取本任务 worker 的 bearer token，不读取任何账号/模型密钥。
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

VALID = {"true", "false", "uncertain"}


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_token(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("CODEX_REVIEW_WORKER_TOKEN="):
            return line.split("=", 1)[1].strip()
    return text.strip()


def build_candidates(unit: dict) -> list[dict]:
    return [{"target_index": i, "class_name": c["class_name"], "bbox": list(c["bbox"])}
            for i, c in enumerate(unit["candidates"])]


class Ledger:
    """append-only JSONL 台账；每次尝试都落盘并 fsync。"""

    def __init__(self, path: Path | None):
        self.path = path
        self.lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: dict) -> None:
        if self.path is None:
            return
        line = json.dumps(rec, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def count_attempts(self) -> int:
        if self.path is None or not self.path.exists():
            return 0
        n = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == "attempt":
                n += 1
        return n

    def attempts_for(self, rnd: int, unit: str) -> int:
        """该 (round, unit) 在**持久台账**里已发出的尝试数。

        只看 `kind=="attempt"` 起始行，因此进程被杀时的在途请求（有 start 无 result）
        同样计入已用次数——resume 绝不会因丢失结果而突破每单元上限。
        """
        if self.path is None or not self.path.exists():
            return 0
        n = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == "attempt" and r.get("round") == rnd and r.get("unit") == unit:
                n += 1
        return n


class ResourceGuard:
    """每 `interval` 秒检查一次整机可用内存与共享容器健康；失败即写 STOP。"""

    def __init__(self, stop_file: Path | None, container: str, interval: float,
                 min_frac: float = 0.15, enabled: bool = True):
        self.stop_file = stop_file
        self.container = container
        self.interval = interval
        self.min_frac = min_frac
        self.enabled = enabled
        self.last = 0.0
        self.reason: str | None = None
        self.lock = threading.Lock()

    def check(self) -> bool:
        if not self.enabled:
            return True
        with self.lock:
            t = time.monotonic()
            if t - self.last < self.interval:
                return self.reason is None
            self.last = t
            reason = None
            try:
                mem = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
                frac = mem["MemAvailable"] / mem["MemTotal"]
            except Exception as e:  # noqa: BLE001 - 读不到就保守停止
                frac = 0.0
                reason = f"meminfo_unreadable:{type(e).__name__}"
            if frac < self.min_frac:
                reason = f"MemAvailable fraction {frac:.3f} < {self.min_frac}"
            try:
                shared = subprocess.run(
                    ["docker", "inspect", "--format",
                     "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}", self.container],
                    capture_output=True, text=True, timeout=5).stdout.strip()
            except Exception as e:  # noqa: BLE001
                shared = "inspect_failed"
                reason = f"docker_inspect_failed:{type(e).__name__}"
            if shared != "running healthy" and reason is None:
                reason = f"shared container state {shared!r}"
            if reason:
                self.reason = reason
                if self.stop_file is not None:
                    self.stop_file.write_text(reason + "\n", encoding="utf-8")
            return self.reason is None


def post_review(url: str, token: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return json.load(resp)


def worker_call(url, token, payload, timeout):
    """返回 (target_index->decision, body, status)。"""
    try:
        body = post_review(url, token, payload, timeout)
    except urllib.error.HTTPError as e:
        return None, None, f"http_{e.code}"
    except urllib.error.URLError as e:
        return None, None, f"urlerror_{getattr(e, 'reason', 'unknown')}"
    except TimeoutError:
        return None, None, "timeout"
    except Exception as e:  # noqa: BLE001
        return None, None, f"exception_{type(e).__name__}"

    items = body.get("results") if isinstance(body, dict) else None
    n = len(payload["candidates"])
    if not isinstance(items, list) or len(items) != n:
        return None, body, "result_count_mismatch"
    idx = {}
    for x in items:
        if not isinstance(x, dict) or type(x.get("target_index")) is not int or x["target_index"] not in range(n):
            return None, body, "target_index_invalid"
        if x["target_index"] in idx:
            return None, body, "target_index_duplicate"
        if x.get("decision") not in VALID:
            return None, body, "decision_invalid"
        idx[x["target_index"]] = x["decision"]
    if set(idx) != set(range(n)):
        return None, body, "target_mapping_mismatch"
    return dict(idx), body, "ok"


def aggregate(out: Path, rounds: int, units_by_round: dict) -> None:
    """从 raw/roundN/*.json 聚合脱敏 decisions/units（可在中断后单独重跑）。"""
    for rnd in range(1, rounds + 1):
        raw_dir = out / "raw" / f"round{rnd}"
        decisions, unit_rows, failed_units = {}, [], []
        for unit in units_by_round.get(rnd, []):
            f = raw_dir / f"{unit['unit']}.json"
            if not f.exists():
                failed_units.append({"unit": unit["unit"], "split": unit["split"], "last_status": "no_record"})
                continue
            rec = json.loads(f.read_text(encoding="utf-8"))
            last_status = rec["attempts"][-1]["status"] if rec["attempts"] else "no_attempt"
            unit_rows.append({"unit": unit["unit"], "split": unit["split"], "success": rec["success"],
                              "attempts": len(rec["attempts"]), "last_status": last_status,
                              "candidate_count": len(unit["candidates"])})
            if rec["success"]:
                for r in rec["results"]:
                    decisions[r["key"]] = r["decision"]
            else:
                failed_units.append({"unit": unit["unit"], "split": unit["split"], "last_status": last_status})
        (out / "decisions").mkdir(parents=True, exist_ok=True)
        (out / "units").mkdir(parents=True, exist_ok=True)
        (out / "decisions" / f"round{rnd}.json").write_text(
            json.dumps(decisions, ensure_ascii=False, indent=1), encoding="utf-8")
        with (out / "units" / f"round{rnd}.jsonl").open("w", encoding="utf-8") as fh:
            for row in unit_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{now()}] round{rnd} 聚合：决策 {len(decisions)} 框，失败图 {len(failed_units)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-url", default="http://127.0.0.1:18398/review")
    ap.add_argument("--token-file", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", default="holdout")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--retry-backoff", type=float, default=20.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ledger", type=Path, default=None, help="append-only 调用台账 JSONL")
    ap.add_argument("--stop-file", type=Path, default=None)
    ap.add_argument("--shared-container", default="yoloe_service_test")
    ap.add_argument("--resource-interval", type=float, default=60.0)
    ap.add_argument("--no-resource-check", action="store_true",
                    help="仅供本地单元测试：关闭 /proc/meminfo 与共享容器检查；服务器正式运行不得使用")
    ap.add_argument("--budget-cap", type=int, default=0, help="总尝试上限；0 = 4*N")
    ap.add_argument("--finalize-only", action="store_true", help="只聚合已有 raw 结果")
    args = ap.parse_args()

    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    items = json.loads((args.assets / "items.json").read_text(encoding="utf-8"))
    units = sorted([u for u in items if u["split"] in splits], key=lambda u: u["unit"])
    if args.limit:
        units = units[: args.limit]
    n_units = len(units)
    units_by_round = {r: units for r in range(1, args.rounds + 1)}
    (args.out / "raw").mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        aggregate(args.out, args.rounds, units_by_round)
        return

    if args.stop_file is not None and args.stop_file.exists():
        raise SystemExit(f"STOP 文件已存在：{args.stop_file.read_text(encoding='utf-8').strip()[:200]}")
    if args.concurrency > 2:
        raise SystemExit("并发必须 <= 2")

    ledger = Ledger(args.ledger)
    guard = ResourceGuard(args.stop_file, args.shared_container, args.resource_interval,
                          enabled=not args.no_resource_check)
    token = load_token(args.token_file) if not args.dry_run else "dry-run"
    cap = args.budget_cap or 4 * n_units
    prior_attempts = ledger.count_attempts()
    if prior_attempts >= cap:
        raise SystemExit(f"台账已有 {prior_attempts} 次尝试，达到/超过上限 {cap}，拒绝继续")

    attempts_total = prior_attempts
    done_total = 0
    lock = threading.Lock()
    ledger.append({"kind": "run_start", "at": now(), "splits": sorted(splits), "rounds": args.rounds,
                   "units": n_units, "cap": cap, "prior_attempts": prior_attempts,
                   "concurrency": args.concurrency, "max_retries": args.max_retries})
    print(f"[{now()}] 评测开始：{n_units} 图 / splits={sorted(splits)} / rounds={args.rounds} "
          f"concurrency={args.concurrency} 台账已用 {prior_attempts} 上限 {cap}", flush=True)

    stopped_reason = {"reason": None}
    try:
        for rnd in range(1, args.rounds + 1):
            raw_dir = args.out / "raw" / f"round{rnd}"
            raw_dir.mkdir(parents=True, exist_ok=True)

            def one(unit):
                nonlocal attempts_total, done_total
                rec_path = raw_dir / f"{unit['unit']}.json"
                if args.resume and rec_path.exists():
                    try:
                        saved = json.loads(rec_path.read_text(encoding="utf-8"))
                        if saved.get("success"):
                            return saved
                    except Exception:  # noqa: BLE001
                        pass
                img_path = args.assets / "images" / f"{unit['unit']}.jpg"
                if not img_path.exists():
                    rec = {"unit": unit["unit"], "split": unit["split"], "success": False,
                           "attempts": [{"status": "image_missing", "elapsed": 0.0}], "results": []}
                    rec_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                    return rec
                cands = build_candidates(unit)
                payload = {"image": base64.b64encode(img_path.read_bytes()).decode("ascii"), "candidates": cands}
                attempts, decisions, body = [], None, None
                per_unit_cap = args.max_retries + 1
                already = ledger.attempts_for(rnd, unit["unit"])
                if already >= per_unit_cap:
                    # 该单元本轮的尝试额度已在持久台账中用尽（含在途/丢失结果），resume 不再调用。
                    rec = {"unit": unit["unit"], "split": unit["split"], "success": False,
                           "attempts": [{"status": "attempt_cap_reached", "elapsed": 0.0,
                                         "already": already, "cap": per_unit_cap}], "results": []}
                    rec_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                    return rec
                for attempt in range(already, per_unit_cap):
                    if args.stop_file is not None and args.stop_file.exists():
                        attempts.append({"status": "stopped", "elapsed": 0.0})
                        break
                    if not guard.check():
                        attempts.append({"status": "resource_stop", "elapsed": 0.0})
                        stopped_reason["reason"] = guard.reason
                        break
                    with lock:
                        if attempts_total >= cap:
                            attempts.append({"status": "budget_exhausted", "elapsed": 0.0})
                            break
                        attempts_total += 1
                    attempt_uid = uuid.uuid4().hex
                    ledger.append({"kind": "attempt", "phase": f"round{rnd}", "round": rnd,
                                   "unit": unit["unit"], "attempt": attempt, "n_candidates": len(cands),
                                   "attempt_uid": attempt_uid, "at": now()})
                    t0 = time.monotonic()
                    decisions, body, status = worker_call(args.worker_url, token, payload, args.timeout)
                    elapsed = round(time.monotonic() - t0, 2)
                    ledger.append({"kind": "attempt_result", "phase": f"round{rnd}", "round": rnd,
                                   "unit": unit["unit"], "attempt": attempt, "status": status,
                                   "attempt_uid": attempt_uid, "elapsed": elapsed, "at": now()})
                    attempts.append({"status": status, "elapsed": elapsed, "at": now()})
                    if status == "ok":
                        break
                    if attempt < args.max_retries:
                        time.sleep(args.retry_backoff)
                results = []
                if decisions is not None:
                    for i, c in enumerate(cands):
                        results.append({"target_index": i, "key": unit["candidates"][i]["key"],
                                        "decision": decisions[i]})
                rec = {"unit": unit["unit"], "split": unit["split"], "success": decisions is not None,
                       "attempts": attempts, "results": results}
                if body is not None:
                    rec["raw_body"] = body
                tmp = rec_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                tmp.replace(rec_path)
                with lock:
                    done_total += 1
                    n = done_total
                if n % 20 == 0 or n == n_units:
                    print(f"[{now()}] round{rnd} 进度 {n}/{n_units} unit={unit['unit']} "
                          f"ok={decisions is not None} 台账尝试 {attempts_total}/{cap}", flush=True)
                return rec

            if args.dry_run:
                print(f"[{now()}] dry-run round{rnd}：只校验候选编号")
                for u in units:
                    assert [c["target_index"] for c in build_candidates(u)] == list(range(len(u["candidates"])))
                print(f"[{now()}] dry-run round{rnd} 通过 {len(units)} 图")
                continue

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(one, units))

            if args.stop_file is not None and args.stop_file.exists():
                print(f"[{now()}] STOP：{args.stop_file.read_text(encoding='utf-8').strip()[:200]}", flush=True)
                break
    finally:
        aggregate(args.out, args.rounds, units_by_round)
        ledger.append({"kind": "run_end", "at": now(), "attempts_total": attempts_total,
                       "stopped_reason": stopped_reason["reason"]})
        print(f"[{now()}] 结束：累计尝试 {min(attempts_total, cap)}/{cap}"
              + (f"，停止原因 {stopped_reason['reason']}" if stopped_reason["reason"] else ""), flush=True)


if __name__ == "__main__":
    main()
