#!/usr/bin/env python3
import json
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_server


ROOT = Path(__file__).resolve().parent
OFFICIAL_SAMPLES = (ROOT / "samples.json").read_bytes()
OFFICIAL_CSV = (ROOT / "annotations.csv").read_bytes()
FRAME = (320, 200)
BOX = {"x": 40, "y": 30, "width": 80, "height": 50}
SHOT = Path("/tmp/person-label-synth-check.png")


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
        server.server_close()
        thread.join(timeout=3)


def assert_official_untouched():
    assert (ROOT / "samples.json").read_bytes() == OFFICIAL_SAMPLES
    assert (ROOT / "annotations.csv").read_bytes() == OFFICIAL_CSV
    assert json.loads(OFFICIAL_SAMPLES)["eligible"] == []
    assert json.loads(OFFICIAL_SAMPLES)["manual_boxes"] == []
    assert OFFICIAL_CSV.decode("utf-8").strip() == "candidate_id,label,note,labeled_at"


def _item(candidate_id, **changes):
    item = {
        "candidate_id": candidate_id,
        "group": "flight-synth",
        "class_name": "人员",
        "box_origin": "initial_review",
        "bbox": dict(BOX),
        "frame_width": FRAME[0],
        "frame_height": FRAME[1],
        "image_file": f"images/{candidate_id}.png",
        "source_evidence": {
            "task_id": "task-synth",
            "stream_id": "stream-1",
            "frame_id": "frame-7",
            "pts_ns": 1000,
            "binding_id": f"bind-{candidate_id}",
        },
    }
    for key, value in changes.items():
        if value is None:
            item.pop(key, None)
        else:
            item[key] = value
    return item


def _queue(directory, eligible, manual=None, blocked=None):
    payload = {
        "purpose": "本机合成验收，不进入正式队列。",
        "labels": ["是人员", "不是人员", "看不清"],
        "show_model_answer": False,
        "prefill": False,
        "eligible": eligible,
        "manual_boxes": manual or [],
        "blocked_groups": blocked or [],
    }
    (directory / "samples.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    csv_path = directory / "annotations.csv"
    if not csv_path.exists():
        csv_path.write_text("candidate_id,label,note,labeled_at\n", encoding="utf-8")
    return csv_path


def _png(path, size=FRAME, color=(30, 90, 160)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class _Server:
    def __init__(self, package):
        handler = type("BoundHandler", (label_server.Handler,), {
            "package": package,
            "csv_path": package / "annotations.csv",
        })
        self.httpd = label_server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.httpd.server_address
        self.base = f"http://{host}:{port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=3)


def _get(base, path):
    with urllib.request.urlopen(base + path) as response:
        return json.loads(response.read().decode("utf-8"))


def _post(base, payload):
    request = urllib.request.Request(
        base + "/api/label",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _prepare_admission(directory):
    fake = {
        "verified": True,
        "frame_binding": "same_frame_verified",
        "same_frame_verified": True,
        "bytes_differ": True,
        "byte_different": True,
        "nearby_time": True,
        "time_delta": 1,
        "mad": 22,
        "similarity": 0.99,
        "image_similarity": 1,
    }
    good_mad = _item("good-mad")
    good_mad["source_evidence"] = {**good_mad["source_evidence"], "mad": 22}
    pts_zero = _item("pts-zero", source_evidence={
        "task_id": "task-synth", "stream_id": "stream-1", "pts_ns": 0, "binding_id": "bind-zero",
    })
    listed = _item("list-box", bbox=[40, 30, 80, 50])
    pts_over_fake_frame = _item("pts-over-fake-frame", source_evidence={
        "task_id": "task-synth", "stream_id": "stream-1", "frame_id": "verified",
        "pts_ns": 50, "binding_id": "bind-pts",
    })
    items = [
        good_mad,
        pts_zero,
        listed,
        pts_over_fake_frame,
        _item("no-image", image_file=None),
        _item("missing-file", image_file="images/missing-file.png"),
        _item("unreadable"),
        _item("traversal", image_file="../secret.png"),
        _item("size-mismatch"),
        _item("no-bbox", bbox=None),
        _item("outside", bbox={"x": 300, "y": 30, "width": 80, "height": 50}),
        _item("negative", bbox={"x": -1, "y": 0, "width": 10, "height": 10}),
        _item("no-frame", frame_width=None),
        _item("blank-group", group="  "),
        _item("no-source", source_evidence=None),
        _item("fake-signals", source_evidence=fake),
        _item("binding-verified", source_evidence={
            "task_id": "task-synth", "stream_id": "stream-1", "frame_id": "frame-7",
            "binding_id": "verified",
        }),
        _item("frame-verified", source_evidence={
            "task_id": "task-synth", "stream_id": "stream-1", "frame_id": "true",
            "binding_id": "bind-frame",
        }),
        _item("two-gaps", group=" ", source_evidence=None),
    ]
    _queue(directory, items, manual=[{
        "candidate_id": "hand-1",
        "group": "flight-synth",
        "box_origin": "manual",
    }])
    for item in items:
        candidate_id = item["candidate_id"]
        image = directory / "images" / f"{candidate_id}.png"
        if candidate_id in {"no-image", "missing-file", "traversal"}:
            continue
        if candidate_id == "unreadable":
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"this is not an image")
            continue
        if candidate_id == "size-mismatch":
            _png(image, size=(8, 8))
            continue
        _png(image)
    return items


def test_admission_rejects_incomplete_samples():
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        _prepare_admission(directory)
        before = (directory / "samples.json").read_bytes()
        payload = label_server.load_queue(directory / "samples.json")
        assert (directory / "samples.json").read_bytes() == before
        admitted = {item["candidate_id"] for item in payload["eligible"]}
        assert admitted == {"good-mad", "pts-zero", "list-box", "pts-over-fake-frame"}
        reasons = {item["candidate_id"]: item["reasons"] for item in payload["rejected"]}
        assert set(reasons) == {
            "no-image", "missing-file", "unreadable", "traversal", "size-mismatch",
            "no-bbox", "outside", "negative", "no-frame", "blank-group", "no-source",
            "fake-signals", "binding-verified", "frame-verified", "two-gaps",
        }
        for candidate_id in ("no-image", "missing-file", "unreadable", "traversal"):
            assert reasons[candidate_id] == ["缺少可读取图片"]
        assert reasons["size-mismatch"] == ["画面尺寸与图片不一致"]
        for candidate_id in ("no-bbox", "outside", "negative", "no-frame"):
            assert reasons[candidate_id] == ["框坐标或画面尺寸无效"]
        assert reasons["blank-group"] == ["缺少任务分组"]
        for candidate_id in ("no-source", "fake-signals", "binding-verified", "frame-verified"):
            assert reasons[candidate_id] == ["缺少可核验的来源绑定"]
        assert reasons["two-gaps"] == ["缺少任务分组", "缺少可核验的来源绑定"]
        public = label_server.public_queue(payload, directory / "annotations.csv")
        by_id = {item["candidate_id"]: item for item in public["eligible"]}
        assert by_id["good-mad"]["label"] == ""
        assert by_id["good-mad"]["bbox"] == BOX
        assert set(by_id["good-mad"]["source"]) == {
            "task_id", "stream_id", "frame_id", "pts_ns", "binding_id",
        }
        assert "mad" not in by_id["good-mad"]["source"]
        assert 22 not in by_id["good-mad"]["source"].values()
        assert by_id["pts-zero"]["source"]["frame_id"] == ""
        assert by_id["pts-zero"]["source"]["pts_ns"] == 0
        assert by_id["pts-over-fake-frame"]["source"]["frame_id"] == ""
        assert by_id["pts-over-fake-frame"]["source"]["pts_ns"] == 50
        assert by_id["list-box"]["bbox"] == BOX
        assert public["manual_boxes"] == [{
            "candidate_id": "hand-1", "group": "flight-synth", "box_origin": "manual",
        }]
        assert public["show_model_answer"] is False
        leaked = json.dumps(public, ensure_ascii=False)
        for key in ("rawDecision", "model_decision", "gold", "human_label", "reasoning", "confidence"):
            assert key not in leaked
        assert [item["candidate_id"] for item in payload["manual_boxes"]] == ["hand-1"]


def test_illegal_ids_and_prefill_do_not_open_the_queue():
    numbered = _item("ok")
    numbered["candidate_id"] = 12
    cases = [
        ([_item("")], "待标样本缺少唯一编号"),
        ([_item("   ")], "待标样本缺少唯一编号"),
        ([numbered], "待标样本缺少唯一编号"),
        ([_item("same"), _item("same")], "待标样本缺少唯一编号"),
        ([_item("filled", label="是人员")], "待标样本不能预填标签"),
        ([_item("poison", model_decision="true")], "待标样本含有模型结论或旧标签"),
        ([_item("drawn", box_origin="manual")], "待标框必须是初审原框"),
        ([_item("car", class_name="车")], "待标框的初审类别必须是人员"),
    ]
    manual_cases = [
        ([_item("kept")], [{"candidate_id": "kept", "box_origin": "manual"}], "人工补框不能与初审框使用同一编号"),
        ([_item("kept")], [{"candidate_id": "hand", "box_origin": "manual", "label": "不是人员"}], "人工补框不能预填标签"),
    ]
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        for eligible, message in cases:
            _queue(directory, eligible)
            try:
                label_server.load_queue(directory / "samples.json")
            except label_server.QueueError as exc:
                assert message in str(exc)
            else:
                raise AssertionError(message)
        for eligible, manual, message in manual_cases:
            _queue(directory, eligible, manual)
            try:
                label_server.load_queue(directory / "samples.json")
            except label_server.QueueError as exc:
                assert message in str(exc)
            else:
                raise AssertionError(message)


def test_server_saves_three_labels_and_rejects_the_rest():
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        shutil.copyfile(ROOT / "index.html", directory / "index.html")
        colors = {"synth-yes": (180, 40, 40), "synth-no": (40, 140, 70), "synth-unclear": (40, 70, 170)}
        items = [_item(candidate_id) for candidate_id in colors]
        items.append(_item("no-proof", source_evidence={
            "verified": True, "mad": 22, "nearby_time": True, "bytes_differ": True,
        }))
        _queue(directory, items, manual=[{
            "candidate_id": "hand-1", "group": "flight-synth", "box_origin": "manual",
        }])
        for candidate_id, color in colors.items():
            _png(directory / "images" / f"{candidate_id}.png", color=color)
        _png(directory / "images" / "no-proof.png")
        server = _Server(directory)
        try:
            body = _get(server.base, "/api/queue")
            assert [item["candidate_id"] for item in body["eligible"]] == [
                "synth-yes", "synth-no", "synth-unclear",
            ]
            assert all(item["label"] == "" for item in body["eligible"])
            assert body["rejected"][0]["candidate_id"] == "no-proof"
            page = urllib.request.urlopen(server.base + "/").read().decode("utf-8")
            assert "当前没有可标注的框" in page
            assert 'id="photo"' in page and 'id="box"' in page
            for candidate_id, label in (
                ("no-proof", "是人员"),
                ("hand-1", "不是人员"),
                ("missing-id", "看不清"),
                ("synth-yes", "也许"),
            ):
                status, payload = _post(server.base, {"candidate_id": candidate_id, "label": label})
                assert status == 409, payload
            assert "不能标注" in _post(server.base, {"candidate_id": "no-proof", "label": "是人员"})[1]["error"]
            assert "没有这个初审框" in _post(server.base, {"candidate_id": "missing-id", "label": "是人员"})[1]["error"]
            expected = {
                "synth-yes": "是人员",
                "synth-no": "不是人员",
                "synth-unclear": "看不清",
            }
            for candidate_id, label in expected.items():
                status, payload = _post(server.base, {"candidate_id": candidate_id, "label": label})
                assert status == 200 and payload["ok"] is True
            status, _ = _post(server.base, {"candidate_id": "synth-yes", "label": "是人员"})
            assert status == 200
            again = _get(server.base, "/api/queue")
            found = {item["candidate_id"]: item["label"] for item in again["eligible"]}
            assert found == expected
            rows = label_server.read_annotations(directory / "annotations.csv")
            assert {(row["candidate_id"], row["label"]) for row in rows} == set(expected.items())
            assert len(rows) == 3
            saved = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
            assert all(item.get("label", "") in ("", None) for item in saved["eligible"])
        finally:
            server.close()


def _assert_box(page, filename):
    page.wait_for_function(
        """(name) => {
            const img = document.querySelector('#photo');
            return img && img.complete && img.naturalWidth === 320 && img.naturalHeight === 200
                && img.src.endsWith(name);
        }""",
        arg=filename,
    )
    img = page.locator("#photo").bounding_box()
    box = page.locator("#box").bounding_box()
    assert img and box and img["width"] > 0
    assert abs((box["x"] - img["x"]) - img["width"] * 40 / 320) <= 2
    assert abs((box["y"] - img["y"]) - img["height"] * 30 / 200) <= 2
    assert abs(box["width"] - img["width"] * 80 / 320) <= 2
    assert abs(box["height"] - img["height"] * 50 / 200) <= 2


def test_browser_empty_rejected_and_three_labels():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "平均绝对差" not in readme and "不超过 22" not in readme
    assert "字节不同" in readme and "verified" in readme
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        empty = directory / "empty"
        rejected = directory / "rejected"
        labeled = directory / "labeled"
        for path in (empty, rejected, labeled):
            path.mkdir()
            shutil.copyfile(ROOT / "index.html", path / "index.html")
        shutil.copyfile(ROOT / "samples.json", empty / "samples.json")
        (empty / "annotations.csv").write_text("candidate_id,label,note,labeled_at\n", encoding="utf-8")
        _queue(rejected, [_item("no-proof", source_evidence={"verified": True, "mad": 10})])
        _png(rejected / "images" / "no-proof.png")
        colors = {"synth-yes": (180, 40, 40), "synth-no": (40, 140, 70), "synth-unclear": (40, 70, 170)}
        _queue(labeled, [_item(candidate_id) for candidate_id in colors], manual=[{
            "candidate_id": "hand-1", "group": "flight-synth", "box_origin": "manual",
        }])
        for candidate_id, color in colors.items():
            _png(labeled / "images" / f"{candidate_id}.png", color=color)
        servers = [_Server(empty), _Server(rejected), _Server(labeled)]
        notes = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page(viewport={"width": 1280, "height": 900})
                    page.goto(servers[0].base + "/")
                    page.wait_for_function(
                        "() => document.querySelector('#count').textContent.includes('当前没有可标注的初审框')"
                    )
                    assert page.locator("#viewer").is_hidden()
                    for name in ("是人员", "不是人员", "看不清"):
                        assert page.get_by_role("button", name=name, exact=True).is_disabled()
                    assert "2026-10-11至2026-10-13/人员类型" in page.locator("body").inner_text()
                    notes.append("empty: viewer hidden, buttons disabled")

                    page.goto(servers[1].base + "/")
                    page.wait_for_function(
                        "() => document.querySelector('#count').textContent.includes('当前没有可标注的初审框')"
                    )
                    assert page.locator("#viewer").is_hidden()
                    assert page.get_by_role("button", name="是人员", exact=True).is_disabled()
                    assert "no-proof" not in page.locator("body").inner_text()
                    notes.append("rejected: not shown, buttons disabled")

                    page.goto(servers[2].base + "/")
                    _assert_box(page, "synth-yes.png")
                    color = page.locator("#photo").evaluate(
                        """el => {
                            const canvas = document.createElement('canvas');
                            canvas.width = 1; canvas.height = 1;
                            canvas.getContext('2d').drawImage(el, 0, 0, 1, 1);
                            return Array.from(canvas.getContext('2d').getImageData(0, 0, 1, 1).data).slice(0, 3);
                        }"""
                    )
                    assert color == [180, 40, 40]
                    assert page.locator("#saved").inner_text() == "尚未标注"
                    assert "人工补框 1 个" in page.locator("#manual").inner_text()
                    assert "模型答案" not in page.locator("#meta").inner_text()
                    choices = (
                        ("synth-yes", "是人员", "button.yes"),
                        ("synth-no", "不是人员", "button.no"),
                        ("synth-unclear", "看不清", "button.unclear"),
                    )
                    for candidate_id, label, _selector in choices:
                        _assert_box(page, f"{candidate_id}.png")
                        assert page.locator("#saved").get_attribute("data-candidate-id") == candidate_id
                        with page.expect_response("**/api/label") as posted:
                            page.get_by_role("button", name=label, exact=True).click()
                        assert posted.value.status == 200
                        page.wait_for_function(
                            """([id, label]) => {
                                const saved = document.querySelector('#saved');
                                return saved.dataset.candidateId === id && saved.dataset.label === label
                                    && saved.textContent === '已保存：' + label;
                            }""",
                            arg=[candidate_id, label],
                        )
                        if candidate_id != "synth-unclear":
                            page.locator("#next").click()
                    page.reload()
                    page.wait_for_function(
                        "() => document.querySelector('#saved').dataset.label === '是人员'"
                    )
                    for candidate_id, label, selector in choices:
                        _assert_box(page, f"{candidate_id}.png")
                        assert page.locator("#saved").get_attribute("data-candidate-id") == candidate_id
                        assert page.locator("#saved").get_attribute("data-label") == label
                        assert "active" in (page.locator(selector).get_attribute("class") or "")
                        if candidate_id != "synth-unclear":
                            page.locator("#next").click()
                    page.screenshot(path=str(SHOT), full_page=True)
                    mobile = browser.new_page(viewport={"width": 390, "height": 844})
                    mobile.goto(servers[2].base + "/")
                    _assert_box(mobile, "synth-yes.png")
                    assert mobile.locator("#saved").inner_text() == "已保存：是人员"
                    mobile.close()
                    confirmed = _get(servers[2].base, "/api/queue")
                    assert {item["candidate_id"]: item["label"] for item in confirmed["eligible"]} == {
                        "synth-yes": "是人员",
                        "synth-no": "不是人员",
                        "synth-unclear": "看不清",
                    }
                    notes.append("saved-reload: synth-yes=是人员 synth-no=不是人员 synth-unclear=看不清")
                    notes.append("mobile: box aligned, saved label still shown")
                finally:
                    browser.close()
        finally:
            for server in servers:
                server.close()
        print("; ".join(notes))


if __name__ == "__main__":
    assert_official_untouched()
    test_queue_is_empty_and_unlabeled()
    test_server_rejects_label_without_a_box()
    test_admission_rejects_incomplete_samples()
    test_illegal_ids_and_prefill_do_not_open_the_queue()
    test_server_saves_three_labels_and_rejects_the_rest()
    test_browser_empty_rejected_and_three_labels()
    assert_official_untouched()
    print("ok")
