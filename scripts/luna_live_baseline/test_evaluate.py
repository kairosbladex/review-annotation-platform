#!/usr/bin/env python3
"""本地单元测试：evaluate.py 的候选编号映射、台账、重试、STOP、预算上限。

用本地假 worker（返回**逆序** target_index）验证客户端不是按返回顺序而是按编号映射；
不需要模型、网络或凭据。运行：
    python3 scripts/luna_live_baseline/test_evaluate.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
EVAL = HERE / "evaluate.py"


class FakeWorker:
    def __init__(self):
        self.fail_first = 0
        self.calls = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                if self.path != "/review":
                    self.send_response(404); self.end_headers(); return
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                outer.calls += 1
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    self.send_response(503); self.send_header("Content-Type", "application/json")
                    self.end_headers(); self.wfile.write(b'{"error":"busy"}'); return
                cands = payload["candidates"]
                n = len(cands)
                # 故意逆序返回，检验客户端按 target_index 而非位置映射。
                results = [{"target_index": i,
                            "decision": "true" if i % 2 == 0 else "false",
                            "observed_object": "x", "reasoning": "r", "image_limitation": "n"}
                           for i in reversed(range(n))]
                body = json.dumps({"results": results}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def make_assets(root: Path, units=3):
    (root / "images").mkdir(parents=True, exist_ok=True)
    items, labels = [], {}
    for u in range(units):
        uid = f"unit{u:02d}"
        n = (u % 3) + 1
        cands = []
        for i in range(n):
            key = f"{uid}:{i:03d}"
            cands.append({"key": key, "class_name": "车", "bbox": [0.1 + 0.01 * i, 0.1, 0.2 + 0.01 * i, 0.2]})
            labels[key] = {"label": "true" if i % 2 == 0 else "false", "class_name": "车",
                           "source": "test", "split": "dev"}
        items.append({"unit": uid, "group": uid, "source": "test", "image": f"local:{uid}.jpg",
                      "sha256": "x", "candidates": cands, "split": "dev"})
        Image.new("RGB", (64 + u, 48), (100, 120, 140)).save(root / "images" / f"{uid}.jpg", quality=90)
    (root / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    (root / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")


class EvalTests(unittest.TestCase):
    def setUp(self):
        # 用 TemporaryDirectory + addCleanup，测试结束自动删除本轮临时目录。
        self._tmpdir = tempfile.TemporaryDirectory(prefix="eval-test-")
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.assets = self.tmp / "assets"
        make_assets(self.assets, units=3)
        self.token = self.tmp / "token"
        self.token.write_text("CODEX_REVIEW_WORKER_TOKEN=test-token\n", encoding="utf-8")
        self.worker = FakeWorker()
        self.url = f"http://127.0.0.1:{self.worker.port}/review"

    def tearDown(self):
        self.worker.stop()

    def run_eval(self, out, ledger, stop_file=None, extra=None):
        cmd = [sys.executable, str(EVAL),
               "--worker-url", self.url, "--token-file", str(self.token),
               "--assets", str(self.assets), "--labels", str(self.assets / "labels.json"),
               "--out", str(out), "--splits", "dev", "--rounds", "2", "--concurrency", "2",
               "--max-retries", "1", "--ledger", str(ledger), "--no-resource-check",
               "--retry-backoff", "0.1"]
        if stop_file is not None:
            cmd += ["--stop-file", str(stop_file)]
        if extra:
            cmd += extra
        return subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    def test_numbering_mapping_ledger_and_retry(self):
        out = self.tmp / "out1"
        ledger = self.tmp / "ledger1.jsonl"
        self.worker.fail_first = 1  # 第一个请求 503，触发一次重试
        r = self.run_eval(out, ledger)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        for rnd in (1, 2):
            dec = json.loads((out / "decisions" / f"round{rnd}.json").read_text(encoding="utf-8"))
            # 每个候选 i 的结论应为 true(i 偶)/false(i 奇)，与返回顺序无关。
            items = json.loads((self.assets / "items.json").read_text(encoding="utf-8"))
            for u in items:
                for i, c in enumerate(u["candidates"]):
                    self.assertEqual(dec[c["key"]], "true" if i % 2 == 0 else "false",
                                     f"round{rnd} {c['key']} 映射错误")
        lines = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines()]
        attempts = [l for l in lines if l.get("kind") == "attempt"]
        results = [l for l in lines if l.get("kind") == "attempt_result"]
        self.assertEqual(len(attempts), 7, "3 图×2 轮 + 1 次重试 = 7 次尝试")
        self.assertEqual(len(results), 7)
        failed = [l for l in results if l.get("status") == "http_503"]
        self.assertEqual(len(failed), 1, "应有 1 次 503 记录")
        self.assertTrue(any(l.get("kind") == "run_end" for l in lines))
        # 每个成功图的 attempt 数应能反映重试。
        units_r1 = [json.loads(l) for l in (out / "units" / "round1.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(u["success"] for u in units_r1))

    def test_stop_file_blocks(self):
        out = self.tmp / "out2"
        ledger = self.tmp / "ledger2.jsonl"
        stop = self.tmp / "STOP"
        stop.write_text("test stop\n", encoding="utf-8")
        before = self.worker.calls
        r = self.run_eval(out, ledger, stop_file=stop)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.worker.calls, before, "STOP 存在时不应发起请求")
        self.assertFalse(ledger.exists(), "STOP 时不应写台账")

    def test_budget_cap(self):
        out = self.tmp / "out3"
        ledger = self.tmp / "ledger3.jsonl"
        r = self.run_eval(out, ledger, extra=["--budget-cap", "1"])
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        lines = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines()]
        attempts = [l for l in lines if l.get("kind") == "attempt"]
        self.assertEqual(len(attempts), 1, "预算上限 1 时只允许 1 次尝试")
        units = [json.loads(l) for l in (out / "units" / "round1.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(u["last_status"] == "budget_exhausted" for u in units),
                        "应有图因预算耗尽被记录")

    def test_resume_does_not_reset_budget(self):
        out = self.tmp / "out4"
        ledger = self.tmp / "ledger4.jsonl"
        r1 = self.run_eval(out, ledger)
        self.assertEqual(r1.returncode, 0, r1.stderr[-2000:])
        n1 = sum(1 for l in ledger.read_text(encoding="utf-8").splitlines() if '"kind": "attempt"' in l)
        self.worker.calls = 0
        # 再跑一次（resume）：应跳过已成功图，不新增尝试、不再发请求。
        cmd = [sys.executable, str(EVAL), "--worker-url", self.url, "--token-file", str(self.token),
               "--assets", str(self.assets), "--labels", str(self.assets / "labels.json"),
               "--out", str(out), "--splits", "dev", "--rounds", "2", "--concurrency", "2",
               "--max-retries", "1", "--ledger", str(ledger), "--no-resource-check", "--resume"]
        r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        self.assertEqual(r2.returncode, 0, r2.stderr[-2000:])
        n2 = sum(1 for l in ledger.read_text(encoding="utf-8").splitlines() if '"kind": "attempt"' in l)
        self.assertEqual(n1, n2, "resume 不应新增尝试（已成功图被跳过）")
        self.assertEqual(self.worker.calls, 0, "resume 不应再发请求")

    def test_per_unit_cap_across_resume(self):
        """跨进程恢复：台账里已有 2 次（含在途）尝试的单元，resume 不得再调用。"""
        make_assets(self.assets, units=2)  # 本用例只需 unit00 / unit01
        out = self.tmp / "out5"
        ledger = self.tmp / "ledger5.jsonl"
        # 模拟上一进程对 unit00 round1 已发出 2 次尝试（例如 1 次失败 + 1 次在途被杀）。
        pre = [
            {"kind": "attempt", "phase": "round1", "round": 1, "unit": "unit00", "attempt": 0,
             "n_candidates": 1, "at": "2026-10-11T00:00:00+08:00"},
            {"kind": "attempt", "phase": "round1", "round": 1, "unit": "unit00", "attempt": 1,
             "n_candidates": 1, "at": "2026-10-11T00:00:30+08:00"},
        ]
        ledger.write_text("\n".join(json.dumps(x) for x in pre) + "\n", encoding="utf-8")
        self.worker.calls = 0
        cmd = [sys.executable, str(EVAL), "--worker-url", self.url, "--token-file", str(self.token),
               "--assets", str(self.assets), "--labels", str(self.assets / "labels.json"),
               "--out", str(out), "--splits", "dev", "--rounds", "1", "--concurrency", "2",
               "--max-retries", "1", "--ledger", str(ledger), "--no-resource-check", "--resume",
               "--budget-cap", "100", "--retry-backoff", "0.1"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        # unit00 额度已用尽：不得再调用；unit01 正常调用一次。
        self.assertEqual(self.worker.calls, 1, "只应调用 unit01")
        rec00 = json.loads((out / "raw" / "round1" / "unit00.json").read_text(encoding="utf-8"))
        self.assertEqual(rec00["attempts"][0]["status"], "attempt_cap_reached")
        self.assertFalse(rec00["success"])
        lines = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines()]
        unit00_attempts = [l for l in lines if l.get("kind") == "attempt" and l.get("unit") == "unit00"]
        self.assertEqual(len(unit00_attempts), 2, "unit00 不得新增尝试")
        unit01_attempts = [l for l in lines if l.get("kind") == "attempt" and l.get("unit") == "unit01"]
        self.assertEqual(len(unit01_attempts), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
