#!/usr/bin/env python3
"""本地单元测试：ledger_util 的顺序配对。

重点回归：进程重启后同一 (phase, unit, attempt) 会再次出现；
用 set 配对会丢掉“在途未配对”的旧尝试。这里用合成台账覆盖：
  * 重复 attempt 编号（重启）
  * 在途未配对必须保留
  * 额外尝试按 (round, unit) 计数
  * attempt_uid 优先精确配对
运行：python3 scripts/luna_live_baseline/test_ledger.py
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ledger_util", HERE / "ledger_util.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def att(unit, attempt=0, rnd=1, uid=None):
    r = {"kind": "attempt", "phase": f"round{rnd}", "round": rnd, "unit": unit, "attempt": attempt}
    if uid:
        r["attempt_uid"] = uid
    return r


def res(unit, attempt=0, rnd=1, status="ok", uid=None):
    r = {"kind": "attempt_result", "phase": f"round{rnd}", "round": rnd, "unit": unit,
         "attempt": attempt, "status": status}
    if uid:
        r["attempt_uid"] = uid
    return r


class LedgerTests(unittest.TestCase):
    def test_restart_duplicate_attempt_numbers_keep_inflight(self):
        records = [
            att("A", 0, uid="u1"),          # 第一次（在途被殺）
            att("A", 0, uid="u2"),          # 重启后重试
            res("A", 0, uid="u2"),          # 重启后成功
            att("B", 0, uid="u3"),
            res("B", 0, uid="u3"),
        ]
        s = L.summarize(records)
        self.assertEqual(s["attempts"], 3)
        self.assertEqual(s["attempt_results"], 2)
        self.assertEqual(len(s["in_flight"]), 1, "在途旧尝试不能被重启成功覆盖")
        self.assertEqual(s["extra_attempts"], 1, "A 单元有 1 次额外尝试")
        self.assertEqual(s["units_over_cap"], [])
        self.assertEqual(s["in_flight"][0]["attempt_uid"], "u1")

    def test_no_uid_fifo_pairing(self):
        # 无 attempt_uid 时按 FIFO 配对，仍保留在途计数（旧的 set 方案会丢）。
        records = [
            att("A", 0),
            att("A", 0),
            res("A", 0),
        ]
        s = L.summarize(records)
        self.assertEqual(s["attempts"], 2)
        self.assertEqual(len(s["in_flight"]), 1)
        self.assertEqual(s["extra_attempts"], 1)

    def test_expected_274_plus_2_shape(self):
        # 2 轮 × 2 图 = 4 次唯一成功；另有 2 次中断重试 → 6 次尝试、2 次额外。
        records = []
        for rnd in (1, 2):
            for u in ("A", "B"):
                records += [att(u, 0, rnd, uid=f"{u}{rnd}a"), res(u, 0, rnd, uid=f"{u}{rnd}a")]
        records += [att("A", 0, 1, uid="A1x")]   # 中断的旧尝试（在途）
        records += [att("A", 0, 1, uid="A1y"), res("A", 0, 1, uid="A1y")]  # 重试成功
        s = L.summarize(records)
        self.assertEqual(s["attempts"], 6)
        self.assertEqual(s["attempt_results"], 5)
        self.assertEqual(len(s["in_flight"]), 1)
        self.assertEqual(s["extra_attempts"], 2)

    def test_unknown_uid_does_not_match_another_attempt(self):
        s = L.summarize([att("A", uid="first"), res("A", uid="unknown")])
        self.assertEqual(s["orphan_results"], 1)
        self.assertEqual(s["attempt_results"], 0)
        self.assertEqual(len(s["in_flight"]), 1)

    def test_over_cap_detected(self):
        records = [att("A", i, uid=f"x{i}") for i in range(3)]
        s = L.summarize(records)
        self.assertEqual(len(s["units_over_cap"]), 1)
        self.assertEqual(s["units_over_cap"][0]["attempts"], 3)
        self.assertNotIn("unit", s["units_over_cap"][0])
        self.assertEqual(s["extra_attempts"], 2)

    def test_hash_units(self):
        records = [att("A", 0, uid="u"), res("A", 0, uid="u")]
        hashed = L.summarize(records, hash_units=True)
        raw = L.summarize(records, hash_units=False)
        self.assertIn("unit_hash", hashed["calls"][0])
        self.assertNotIn("unit", hashed["calls"][0])
        self.assertEqual(raw["calls"][0]["unit"], "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
