#!/usr/bin/env python3
"""本地单元测试：score_baseline 的交叉表、派生指标、翻转与聚类区间。

用手工可算的小样本核对口径，并验证聚类 bootstrap 在同种子下可复现。
运行：python3 scripts/luna_live_baseline/test_score.py
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("score_baseline", HERE / "score_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load()


class ScoreTests(unittest.TestCase):
    def test_cross_tab_and_derive(self):
        labels = {
            "a:0": {"label": "true", "class_name": "车", "split": "holdout"},
            "a:1": {"label": "true", "class_name": "车", "split": "holdout"},
            "a:2": {"label": "false", "class_name": "车", "split": "holdout"},
            "b:0": {"label": "false", "class_name": "人员", "split": "holdout"},
        }
        decisions = {"a:0": "true", "a:1": "false", "a:2": "true"}  # b:0 缺失
        tab, per_class = S.cross_tab(labels, decisions, list(labels))
        d = S.derive(tab)
        self.assertEqual(d["n"], 3)
        self.assertEqual(d["gold_true"], 2)
        self.assertEqual(d["gold_false"], 1)
        # right = true&是 + 非true&非是 = 1 + 0 = 1
        self.assertEqual(d["right"], 1)
        self.assertAlmostEqual(d["accuracy"], 1 / 3, places=4)
        # true_recall = 1/2
        self.assertAlmostEqual(d["true_recall"], 0.5, places=4)
        # false_release_rate = 1/1
        self.assertAlmostEqual(d["false_release_rate"], 1.0, places=4)
        self.assertEqual(d["uncertain"], 0)
        self.assertEqual(per_class["人员"][("gold_false", "missing")], 1)

    def test_flips(self):
        a = {"k1": "true", "k2": "false", "k3": "uncertain"}
        b = {"k1": "true", "k2": "true", "k3": "uncertain"}
        f = S.flips(a, b)
        self.assertEqual(f["compared"], 3)
        self.assertEqual(f["raw_flips"], 1)       # k2 three-value change
        self.assertEqual(f["decision_flips"], 1)  # k2 true/not-true change

    def test_bootstrap_deterministic(self):
        labels = {f"u{i}:0": {"label": "true" if i % 2 == 0 else "false", "class_name": "车", "split": "holdout"}
                  for i in range(10)}
        decisions = {k: ("true" if i % 3 else "false") for i, k in enumerate(labels)}
        key_to_unit = {k: k.split(":")[0] for k in labels}
        a = S.cluster_bootstrap(labels, decisions, list(labels), key_to_unit, iters=200, seed=7)
        b = S.cluster_bootstrap(labels, decisions, list(labels), key_to_unit, iters=200, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(a["n_clusters"], 10)

    def test_group_bootstrap_coverage(self):
        labels = {f"u{i}:0": {"label": "true" if i % 2 == 0 else "false", "class_name": "车", "split": "holdout"}
                  for i in range(6)}
        decisions = {k: ("true" if i % 3 else "false") for i, k in enumerate(labels)}
        key_to_unit = {k: k.split(":")[0] for k in labels}
        # u0,u1 同航次 gA；u2 航次 gB；u3..u5 未映射（None）
        unit_to_group = {"u0": "gA", "u1": "gA", "u2": "gB", "u3": None, "u4": None, "u5": None}
        r = S.cluster_bootstrap_groups(labels, decisions, list(labels), key_to_unit, unit_to_group, iters=50, seed=1)
        self.assertEqual(r["coverage"]["units_covered"], 3)
        self.assertEqual(r["coverage"]["units_total"], 6)
        self.assertEqual(r["coverage"]["boxes_covered"], 3)
        self.assertEqual(r["coverage"]["groups"], 2)
        self.assertIsNotNone(r["accuracy_ci95"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
