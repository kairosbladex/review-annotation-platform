"""Cross-repo contract checks; set PV_MATCHER_ROOT to the AR checkout."""
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

from fastapi.testclient import TestClient
from PIL import Image
import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/pv_component_workbench.py"
spec = importlib.util.spec_from_file_location("pv_workbench", PATH)
workbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workbench)


@pytest.fixture
def matcher_root():
    value = os.environ.get("PV_MATCHER_ROOT")
    if not value:
        pytest.skip("Set PV_MATCHER_ROOT to run the cross-repo PV tests")
    return Path(value)


@pytest.fixture
def snapshot(tmp_path, matcher_root):
    # Synthetic source is intentionally not the user-confirmed target ID.
    image = io.BytesIO()
    Image.new("RGB", (100, 100)).save(image, "JPEG")
    raw = image.getvalue()
    photos = tmp_path / "photos"
    photos.mkdir()
    (photos / "event1.jpg").write_bytes(raw)
    row = dict(id="event1", target_location_id="matrix", target_location="Test matrix",
               task_id="flight1", source={"id":"source1"}, task_info="DO_NOT_LEAK_OLD_PREDICTION",
               request=dict(image_width=100,image_height=100,bboxes_xyxy=[[20,20,30,30],[40,40,50,50]],
                            photo_camera={"image_sha256":hashlib.sha256(raw).hexdigest()}),
               blocks=[dict(id="panel1",matrixId="matrix",code="code1",blockRow=1,blockColumn=1,
                    lngLatAlt=json.dumps({"geometry":{"coordinates":[[[113.,34.,100.],
                           [113.00001,34.,100.],[113.00001,34.00001,100.],[113.,34.00001,100.],
                           [113.,34.,100.]]]}}))])
    source = tmp_path / "source.json"
    workbench.write(source, [row])
    root = tmp_path / "snapshot"
    workbench.freeze(source, photos, root, matcher_root)
    return root


def test_freeze_preserves_slots_and_separates_old_labels(snapshot):
    manifest = workbench.verify(snapshot)
    assert manifest["detection_count"] == 2
    payload = workbench.read(snapshot / manifest["cases"][0]["input"])
    assert [d["index"] for d in payload["detections"]] == [0,1]
    assert "DO_NOT_LEAK" not in json.dumps(payload)
    assert all(g["status"] == "unknown" for g in workbench.read(snapshot/"gold/labels.json"))


def test_frozen_data_tamper_rejected(snapshot):
    (snapshot / "images/event1.jpg").write_bytes(b"replaced")
    with pytest.raises(ValueError,match="Frozen asset changed"):
        workbench.verify(snapshot)


def test_existing_snapshot_never_overwritten(snapshot, matcher_root):
    with pytest.raises(ValueError,match="already exists"):
        workbench.freeze("unused", "unused", snapshot, matcher_root)


def test_unknown_gold_not_counted_as_correct(snapshot, matcher_root):
    report = workbench.read(workbench.assess(snapshot, matcher_root))
    assert report["development_gold_count"] == 0
    assert report["unknown_gold_count"] == 2
    assert report["states"]["unavailable"] == 2
    assert not report["p2_allowed"] and not report["blind_test"]["passed"]


def test_local_ui_api_and_calibration_staleness(snapshot, matcher_root):
    client = TestClient(workbench.create_app(snapshot, matcher_root))
    assert client.get("/").status_code == 200
    config = client.get("/api/cases").json()
    data = client.get("/api/cases/event1").json()
    assert "gold" not in data and "task_info" not in data
    assert client.get("/api/cases/no-such-case").status_code == 404
    calibration = dict(image_sha256=data["input"]["image_sha256"],
                       layout_sha256=data["input"]["layout_sha256"], planes=[])
    path = "/api/cases/event1/calibration"
    assert client.post(path,json=calibration).status_code == 403
    headers = {"X-Workbench-Token":config["token"]}
    first = client.post(path,json=calibration,headers=headers)
    second = client.post(path,json=calibration,headers=headers)
    assert first.status_code == 200 and first.json() == second.json()
    assert len(list((snapshot/"runs").glob("event1-*.json"))) == 1
    calibration["layout_sha256"] = "old"
    assert client.post(path,json=calibration,headers=headers).status_code == 400
    assert client.get("/api/cases",headers={"Host":"untrusted.example"}).status_code == 400


def test_http_route_has_same_fail_closed_contract(snapshot, matcher_root):
    client = TestClient(workbench.create_app(snapshot, matcher_root))
    body = workbench.read(snapshot/"inputs/event1.json")
    body["image_base64"] = base64.b64encode((snapshot/"images/event1.jpg").read_bytes()).decode()
    response = client.post("/api/v1/pv/component-match",json=body)
    assert response.status_code == 200
    assert [r["index"] for r in response.json()["results"]] == [0,1]
    assert all(r["status"] == "unavailable" for r in response.json()["results"])
    assert not response.json()["business_write_enabled"]
