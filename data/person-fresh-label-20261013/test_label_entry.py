#!/usr/bin/env python3
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_server


ROOT = Path(__file__).resolve().parent


def test_queue_is_empty_and_unlabeled():
    payload = label_server.load_queue(ROOT / "samples.json")
    assert payload["eligible"] == []
    assert payload["manual_boxes"] == []
    assert payload["show_model_answer"] is False
    text = (ROOT / "index.html").read_text(encoding="utf-8")
    for label in ("是人员", "不是人员", "看不清"):
        assert label in text
    assert "模型答案" in text
    assert "rawDecision" not in text
    header = (ROOT / "annotations.csv").read_text(encoding="utf-8").strip()
    assert header == "candidate_id,label,note,labeled_at"


def test_server_rejects_label_without_a_box():
    handler = type("Bound", (label_server.Handler,), {"package": ROOT, "csv_path": ROOT / "annotations.csv"})
    server = label_server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/queue") as response:
            body = json.loads(response.read().decode("utf-8"))
        assert body["eligible"] == []
        assert body["show_model_answer"] is False
        assert [item["group"] for item in body["blocked_groups"]] == [
            "2026-10-11至2026-10-13/人员类型",
            "2026-10-11至2026-10-13/车辆类型",
            "6755b1e-business两条输入存档",
        ]
        request = urllib.request.Request(
            f"http://{host}:{port}/api/label",
            data=json.dumps({"candidate_id": "made-up", "label": "是人员"}).encode("utf-8"),
            method="POST",
        )
        try:
            urllib.request.urlopen(request)
            raise AssertionError("空队列不应接受标签")
        except urllib.error.HTTPError as exc:
            assert exc.code == 409
        page = urllib.request.urlopen(f"http://{host}:{port}/").read().decode("utf-8")
        assert "当前没有可标注的框" in page
        assert (ROOT / "annotations.csv").read_text(encoding="utf-8").strip().count("\n") == 0
    finally:
        server.shutdown()
        thread.join(timeout=3)


if __name__ == "__main__":
    test_queue_is_empty_and_unlabeled()
    test_server_rejects_label_without_a_box()
    print("ok")
