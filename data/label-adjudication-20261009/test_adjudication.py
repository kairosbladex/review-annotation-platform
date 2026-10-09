"""复核包挑选规则、更正清单和本机页面的定向测试。"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

import adjudication
import label_server


def _labels_and_items():
    labels = {
        "a": {"label": "false", "class_name": "车", "split": "dev", "source": "design"},
        "b": {"label": "false", "class_name": "人员", "split": "dev", "source": "design"},
        "c": {"label": "true", "class_name": "车", "split": "dev", "source": "design"},
        "d": {"label": "true", "class_name": "人员", "split": "dev", "source": "design"},
        "e": {"label": "false", "class_name": "光斑", "split": "holdout", "source": "design"},
        "f": {"label": "true", "class_name": "违建", "split": "dev", "source": "design"},
    }
    items = [{
        "unit": "unit1",
        "candidates": [
            {"key": key, "class_name": labels[key]["class_name"], "bbox": [0.2, 0.3, 0.4, 0.5]}
            for key in labels
        ],
    }]
    return labels, items


def _runs():
    return {
        "p000": {"a": "true", "b": "true", "c": "false", "d": "false", "e": "true", "f": "uncertain"},
        "p000b": {"a": "uncertain", "b": "false", "c": "false", "d": "uncertain", "e": "true", "f": "false"},
        "p001": {"a": "true", "b": "true", "c": "false", "d": "false", "f": "false"},
    }


class SelectTests(unittest.TestCase):
    def test_rules_keep_uncertain_and_missing_runs_distinct(self):
        records = adjudication.select_records(*_labels_and_items(), _runs())
        chosen = {record["key"]: record["rule"] for record in records}
        self.assertEqual(chosen, {
            "a": adjudication.RULE_HUMAN_FALSE_GE2_TRUE,
            "b": adjudication.RULE_HUMAN_FALSE_GE2_TRUE,
            "c": adjudication.RULE_HUMAN_TRUE_ALL_FALSE,
            "e": adjudication.RULE_HUMAN_FALSE_GE2_TRUE,
        })

    def test_order_is_stable(self):
        first = [record["key"] for record in adjudication.select_records(*_labels_and_items(), _runs())]
        second = [record["key"] for record in adjudication.select_records(*_labels_and_items(), _runs())]
        self.assertEqual(first, second)

    def test_public_queue_has_no_model_fields(self):
        records = adjudication.select_records(*_labels_and_items(), _runs())
        row = adjudication.queue_row(records[0])
        self.assertEqual(set(row), set(adjudication.CSV_FIELDS))
        self.assertFalse(set(adjudication.API_FIELDS) & {"raw", "rule", "source", "split", "p000", "p000b", "p001"})


class CorrectionTests(unittest.TestCase):
    def test_lists_only_real_label_changes(self):
        rows = [
            {"key": "keep", "human_label": "false", "decision": "维持原标签", "class_name": "车", "unit": "u", "bbox": "[]"},
            {"key": "same-yes", "human_label": "true", "decision": "改为是", "class_name": "车", "unit": "u", "bbox": "[]"},
            {"key": "same-no", "human_label": "false", "decision": "改为不是", "class_name": "车", "unit": "u", "bbox": "[]"},
            {"key": "unclear", "human_label": "false", "decision": "看不清", "class_name": "车", "unit": "u", "bbox": "[]"},
            {"key": "empty", "human_label": "true", "decision": "", "class_name": "车", "unit": "u", "bbox": "[]"},
            {"key": "to-yes", "human_label": "false", "decision": "改为是", "class_name": "人员", "unit": "u", "bbox": "[1]", "note": "清楚", "reviewed_at": "t"},
            {"key": "to-no", "human_label": "true", "decision": "改为不是", "class_name": "车", "unit": "u", "bbox": "[2]", "note": "", "reviewed_at": "t2"},
        ]
        changed = adjudication.correction_rows(rows)
        self.assertEqual([row["key"] for row in changed], ["to-yes", "to-no"])
        self.assertEqual(changed[0]["corrected_label"], "true")
        self.assertEqual(changed[1]["corrected_label"], "false")

    def test_writer_does_not_touch_gold_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "labels.json"
            gold.write_text('{"keep": true}\n', encoding="utf-8")
            before = gold.read_bytes()
            out = root / "corrections.csv"
            adjudication.write_corrections([
                {"key": "to-yes", "human_label": "false", "decision": "改为是", "class_name": "车", "unit": "u", "bbox": "[1]", "note": "", "reviewed_at": ""}
            ], out)
            self.assertEqual(gold.read_bytes(), before)
            self.assertIn("to-yes", out.read_text(encoding="utf-8-sig"))
            with self.assertRaises(ValueError):
                adjudication.write_corrections([], root / "labels.json")


class ViewTests(unittest.TestCase):
    def test_zoom_stays_inside_image_and_draws_box(self):
        self.assertEqual(adjudication.zoom_window(100, 80, [0.0, 0.0, 0.05, 0.05])[:2], (0, 0))
        image = Image.new("RGB", (320, 200), (20, 20, 20))
        crop, thumb, meta = adjudication.render_views(image, [0.45, 0.4, 0.55, 0.6])
        self.assertGreater(crop.size[0], 0)
        self.assertEqual(meta["image_w"], 320)
        self.assertTrue(any(pixel[0] > 240 and pixel[1] > 180 for pixel in crop.getdata()))
        self.assertTrue(any(pixel[0] > 240 and pixel[1] > 180 for pixel in thumb.getdata()))


class PageTests(unittest.TestCase):
    def test_page_and_api_hide_model_conclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            for name in ("images", "crops", "thumbs"):
                (package / name).mkdir()
            image = Image.new("RGB", (40, 30), (30, 30, 30))
            for relative in ("images/unit1.jpg", "crops/box_a.jpg", "thumbs/box_a.jpg"):
                image.save(package / relative, "JPEG")
            (package / "index.html").write_text(
                (Path(__file__).resolve().parent / "index.html").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (package / "selection.json").write_text('{"raw": {"p000": "true"}}\n', encoding="utf-8")
            adjudication.write_csv(package / "adjudication.csv", [{
                "key": "box:a",
                "class_name": "车",
                "human_label": "false",
                "unit": "unit1",
                "bbox": "[0.1, 0.1, 0.2, 0.2]",
                "image_file": "images/unit1.jpg",
                "crop_file": "crops/box_a.jpg",
                "thumb_file": "thumbs/box_a.jpg",
                "image_w": "40",
                "image_h": "30",
                "box_w_px": "4",
                "box_h_px": "3",
                "decision": "",
                "note": "",
                "reviewed_at": "",
            }])
            handler = type("BoundHandler", (label_server.Handler,), {"package": package, "csv_path": package / "adjudication.csv"})
            server = label_server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            try:
                page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode("utf-8")
                for text in ("维持原标签", "改为是", "改为不是", "看不清", "局部放大", "整图缩略图", "原人工标签"):
                    self.assertIn(text, page)
                for text in ("p000", "p000b", "p001", "selection.json", "模型"):
                    self.assertNotIn(text, page)
                rows = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/rows").read())["rows"]
                self.assertEqual(set(rows[0]), set(adjudication.API_FIELDS))
                self.assertNotIn("raw", rows[0])
                self.assertEqual(urllib.request.urlopen(f"http://127.0.0.1:{port}/crops/box_a.jpg").status, 200)
                for blocked in ("/selection.json", "/summary.json", "/adjudication.csv", "/images/"):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}{blocked}")
                    self.assertEqual(caught.exception.code, 404)
                body = json.dumps({"key": "box:a", "decision": "改为是", "note": "车在框里"}).encode()
                request = urllib.request.Request(f"http://127.0.0.1:{port}/api/label", data=body, headers={"Content-Type": "application/json"})
                self.assertEqual(json.loads(urllib.request.urlopen(request).read())["ok"], True)
                saved = adjudication.read_csv(package / "adjudication.csv")
                self.assertEqual(saved[0]["decision"], "改为是")
                self.assertEqual(adjudication.correction_rows(saved)[0]["corrected_label"], "true")
                self.assertFalse((package / "labels.json").exists())
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
