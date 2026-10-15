"""Run against the unmodified /app files in the frozen live image, offline only."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest import mock

os.environ.update(
    API_KEY="offline-regression-key", MYSQL_ENABLED="false", MQTT_HOST="",
    MQTT_USERNAME="", MQTT_PASSWORD="", MINIO_ENDPOINT="", MINIO_ACCESS_KEY="",
    MINIO_SECRET_KEY="", LLM_API_KEY="", DEVICE="cpu", REVIEW_YOLOE_DEVICE="cpu",
    NVIDIA_VISIBLE_DEVICES="void", CUDA_VISIBLE_DEVICES="", CODEX_REVIEW_ENABLED="false",
    CLEAN_SOURCE_REVIEW_ENABLED="false", REVIEW_TWO_STATE_COMPAT="false",
    CODEX_REVIEW_WORKER_URL="http://worker.invalid", CODEX_REVIEW_WORKER_TOKEN="synthetic-worker-token",
    VL_COMPARE_ENGINE="registered", VL_COMPARE_BASE_URL="http://model.invalid",
    VL_COMPARE_BASELINE_IS_CURRENT="false", VL_COMPARE_SEMANTIC_TIMEOUT="1",
    OMP_NUM_THREADS="1", OPENCV_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
)

import cv2
import httpx
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api import review_agent_routes as review
from api import vl_compare as vl
from history_compare_runtime.services import history_compare as history

logging.disable(logging.CRITICAL)
cv2.setNumThreads(1)
API_HEADERS = {"X-API-Key": "offline-regression-key"}
SOURCE_HASHES = {
    "/app/api/review_agent_routes.py": "a56fb3bc6e9fea5ef3049960d1270052e4d2beeaf3a3903b76fa1aef4181bf00",
    "/app/api/vl_compare.py": "e7d904fc30007d0e5dd5a84502396d58e79f64692c2d50c057040eddda2ffabd",
    "/app/history_compare_runtime/services/history_compare.py": "5884b3a4c20106875ec8fba7e3927896eb5f40e1f8fe4447888a87577ec165eb",
    "/app/history_compare_runtime/services/history_compare_legacy.py": "2271d5e75f02ac26ba5074a1394c9ce28cd9e084e11393a56d20d91e4cae9975",
    "/app/tests/test_review_agent_routes.py": "a3b2739fe4c87f686af2a9e8ab9a0ef9ba33c3d75612ad6322b6574d8f811c7d",
    "/app/tests/test_mask_for_draw.py": "f31042ce63200fbb039e9cf2883e3e0d8914c053218326c701a8647c04fc842b",
}


def png(image):
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def source(image, name):
    return {"source_type": "base64", "base64_data": base64.b64encode(png(image)).decode(), "image_id": name}


def textured_scene():
    image = np.random.default_rng(42).integers(30, 210, size=(480, 640, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)


def model_response(body, status=200):
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://model.invalid"))


class LiveReviewHTTP(unittest.TestCase):
    def setUp(self):
        self.stack = __import__("contextlib").ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(mock.patch.object(review, "REVIEW_AGENT_UPLOAD_DIR", root / "uploads"))
        self.stack.enter_context(mock.patch.object(review, "REVIEW_AGENT_RESULT_DIR", root / "results"))
        self.stack.enter_context(mock.patch.dict(os.environ, CODEX_REVIEW_ENABLED="true"))
        self.stack.enter_context(mock.patch.object(review, "get_reviewer", side_effect=AssertionError("real reviewer forbidden")))
        self.stack.enter_context(mock.patch.object(review, "get_annotation_store", side_effect=AssertionError("annotation DB forbidden")))
        app = FastAPI()
        app.include_router(review.router, prefix="/api/v1/review-agent")
        self.client = self.stack.enter_context(TestClient(app, raise_server_exceptions=False))
        self.image = png(np.zeros((120, 160, 3), dtype=np.uint8))
        self.boxes = [{"class_name": name, "bbox": [0.1, 0.1, 0.4, 0.4], "confidence": 0.8}
                      for name in ("人员", "车", "卡车")]

    def post(self, *, extra=None, boxes=None, clean=False):
        files = {"image": ("synthetic.png", self.image, "image/png")}
        if clean:
            files["clean_image"] = ("synthetic-clean.png", self.image, "image/png")
        return self.client.post("/api/v1/review-agent/review", files=files,
                                data={"detections": json.dumps(self.boxes if boxes is None else boxes), **(extra or {})})

    @staticmethod
    def decisions():
        return [{"target_index": i, "decision": value, "reasoning": "合成判断", "observed_object": "合成目标",
                 "image_limitation": ""} for i, value in enumerate(("true", "false", "uncertain"))]

    def test_luna_roundtrip_preserves_raw_states_and_target_order(self):
        rows = self.decisions()
        with mock.patch.object(httpx.HTTPTransport, "handle_request", autospec=True, return_value=model_response({"results": rows[::-1]})) as upstream:
            response = self.post()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([r["rawDecision"] for r in body["results"]], ["true", "false", "uncertain"])
        self.assertEqual([r["original_class"] for r in body["results"]], ["人员", "车", "卡车"])
        self.assertEqual([r["businessAction"] for r in body["results"]], ["keep", "candidate_drop", "candidate_drop"])
        self.assertEqual((body["summary"]["trueCount"], body["summary"]["falseCount"]), (1, 2))
        payload = json.loads(upstream.call_args.args[1].content)
        self.assertEqual(base64.b64decode(payload["image"]), self.image)
        self.assertEqual([c["target_index"] for c in payload["candidates"]], [0, 1, 2])

    def test_worker_timeout_is_failure_not_false(self):
        with mock.patch.object(httpx.HTTPTransport, "handle_request", side_effect=httpx.ReadTimeout("synthetic timeout")):
            response = self.post()
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["success"])
        self.assertNotIn("results", response.json())

    def test_duplicate_target_is_failure_not_false(self):
        rows = self.decisions()
        rows[2]["target_index"] = 1
        with mock.patch.object(httpx.HTTPTransport, "handle_request", return_value=model_response({"results": rows})):
            response = self.post()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("results", response.json())

    def test_invalid_box_stops_before_worker(self):
        boxes = [{"class_name": "人员", "bbox": [0.6, 0.1, 0.4, 0.4]}]
        with mock.patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("invalid input reached worker")):
            response = self.post(boxes=boxes)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["success"])

    def test_missing_clean_binding_input_stops_before_worker(self):
        with mock.patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("invalid binding reached worker")):
            response = self.post(extra={"clean_frame_id": "synthetic-frame"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["errorCode"], "codex_invalid_frame_binding")

    def test_unverified_clean_worker_result_cannot_claim_binding(self):
        with mock.patch.object(httpx.HTTPTransport, "handle_request", return_value=model_response({"results": self.decisions()})):
            response = self.post(extra={"clean_frame_id": "synthetic-frame"}, clean=True)
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["success"])


class LiveHistoryHTTP(unittest.TestCase):
    def setUp(self):
        self.stack = __import__("contextlib").ExitStack()
        self.addCleanup(self.stack.close)
        self.store = self.stack.enter_context(mock.patch.object(vl._CompareStore, "save", return_value=None))
        self.notify = self.stack.enter_context(mock.patch.object(vl, "_maybe_notify", side_effect=AssertionError("notification forbidden")))
        self.stack.enter_context(mock.patch.object(httpx.AsyncClient, "get", side_effect=AssertionError("remote image download forbidden")))
        app = FastAPI()
        app.include_router(vl.router, prefix="/api/v1")
        self.client = self.stack.enter_context(TestClient(app, raise_server_exceptions=False))
        self.baseline = textured_scene()
        self.current = self.baseline.copy()
        self.current[100:170, 150:240] = [15, 35, 70]

    def post(self, current=None, *, many=False, headers=True):
        current = self.current if current is None else current
        if many:
            url = "/api/v1/vl-compare/many-to-many"
            body = {"baseline_images": [source(self.baseline, "old")], "compare_images": [source(current, "new")],
                    "pairing_mode": "by_order"}
        else:
            url = "/api/v1/vl-compare/one-to-many"
            body = {"current_image": source(current, "new"), "history_images": [source(self.baseline, "old")]}
        body.update(task_description="对比光伏板变化", notify_mqtt=False)
        return self.client.post(url, json=body, headers=API_HEADERS if headers else {})

    def test_auth_required(self):
        response = self.post(headers=False)
        self.assertEqual(response.status_code, 401)
        self.store.assert_not_called()

    def test_identical_images_no_change_without_model(self):
        with mock.patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("unchanged image reached model")):
            response = self.post(self.baseline.copy())
        self.assertEqual(response.status_code, 200)
        item = response.json()["comparisons"][0]
        self.assertEqual((item["status"], item["polygons"], item["diff_summary"]), ("matched", [], "没有明显变化"))

    def test_real_geometry_and_grouped_summary_reach_http_response(self):
        body = {"regions": [{"id": 1, "decision": "keep", "summary": "当前图新增光伏板"}],
                "changes": [{"type": "新增", "count": 1, "detail": "当前图屋面新增蓝色光伏板"}]}
        with mock.patch.object(httpx.HTTPTransport, "handle_request", return_value=model_response({"message": {"content": json.dumps(body)}})):
            response = self.post()
        self.assertEqual(response.status_code, 200)
        item = response.json()["comparisons"][0]
        self.assertEqual(item["status"], "matched")
        self.assertEqual(len(item["polygons"]), 1)
        center = np.mean(item["polygons"][0], axis=0)
        self.assertLess(abs(center[0] - 195), 25)
        self.assertLess(abs(center[1] - 135), 25)
        self.assertIn("新增1处", item["diff_summary"])
        self.notify.assert_not_called()

    def test_many_to_many_keeps_reference_direction(self):
        body = {"regions": [{"id": 1, "decision": "keep", "summary": "当前图新增光伏板"}],
                "changes": [{"type": "新增", "count": 1, "detail": "当前图屋面新增蓝色光伏板"}]}
        with mock.patch.object(httpx.HTTPTransport, "handle_request", return_value=model_response({"message": {"content": json.dumps(body)}})):
            response = self.post(many=True)
        self.assertEqual(response.status_code, 200)
        item = response.json()["pairs"][0]
        self.assertIn("对比照片屋面新增", item["diff_summary"])
        self.assertNotIn("当前图", item["diff_summary"])
        self.assertEqual((item["baseline_index"], item["compare_index"]), (0, 0))

    def test_model_timeout_does_not_report_unchanged(self):
        with mock.patch.object(httpx.HTTPTransport, "handle_request", side_effect=httpx.ReadTimeout("synthetic timeout")):
            response = self.post()
        self.assertEqual(response.status_code, 200)
        item = response.json()["comparisons"][0]
        self.assertEqual(item["status"], "not_found")
        self.assertNotIn("没有明显变化", item["diff_summary"])

    def test_too_small_image_is_422_before_model(self):
        with mock.patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("small image reached model")):
            response = self.post(np.zeros((16, 16, 3), dtype=np.uint8))
        self.assertEqual(response.status_code, 422)

    def test_invalid_base64_image_returns_client_error_before_model(self):
        bad = {"source_type": "base64", "base64_data": base64.b64encode(b"invalid synthetic image").decode()}
        body = {"current_image": bad, "history_images": [source(self.baseline, "old")],
                "task_description": "对比光伏板变化", "notify_mqtt": False}
        with mock.patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("invalid image reached model")):
            response = self.client.post("/api/v1/vl-compare/one-to-many", json=body, headers=API_HEADERS)
        self.assertIn(response.status_code, (400, 422), "invalid image must not become HTTP 500")

    def test_invalid_http_image_is_422_before_model(self):
        body = {"current_image": {"source_type": "http", "url": "http://image.invalid/synthetic.png"},
                "history_images": [], "notify_mqtt": False}
        download = httpx.Response(200, content=b"invalid synthetic image", request=httpx.Request("GET", "http://image.invalid"))
        with mock.patch.object(httpx.AsyncClient, "get", return_value=download), \
             mock.patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("invalid image reached model")):
            response = self.client.post("/api/v1/vl-compare/one-to-many", json=body, headers=API_HEADERS)
        self.assertEqual(response.status_code, 422)

    def test_invalid_local_image_is_422_before_model(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as file:
            file.write(b"invalid synthetic image")
            file.flush()
            body = {"current_image": {"source_type": "local", "path": file.name},
                    "history_images": [], "notify_mqtt": False}
            with mock.patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("invalid image reached model")):
                response = self.client.post("/api/v1/vl-compare/one-to-many", json=body, headers=API_HEADERS)
        self.assertEqual(response.status_code, 422)

    def test_valid_local_image_is_streamed_without_path_read_bytes(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as file:
            file.write(png(self.baseline))
            file.flush()
            body = {"current_image": {"source_type": "local", "path": file.name},
                    "history_images": [], "notify_mqtt": False}
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("local files must not be preloaded")):
                response = self.client.post("/api/v1/vl-compare/one-to-many", json=body, headers=API_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"]["pair_count"], 0)

    def test_local_file_access_failure_is_not_image_format_error(self):
        with tempfile.TemporaryDirectory() as directory:
            body = {"current_image": {"source_type": "local", "path": str(Path(directory) / "missing.png")},
                    "history_images": [], "notify_mqtt": False}
            response = self.client.post("/api/v1/vl-compare/one-to-many", json=body, headers=API_HEADERS)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("无法读取图片", response.text)
        self.assertNotIn("没有明显变化", response.text)

    def test_unrelated_decoder_dependency_failure_is_not_success(self):
        body = {"current_image": source(self.baseline, "synthetic"), "history_images": [], "notify_mqtt": False}
        with mock.patch.object(Image, "open", side_effect=ModuleNotFoundError("synthetic dependency failure", name="unrelated_dependency")):
            response = self.client.post("/api/v1/vl-compare/one-to-many", json=body, headers=API_HEADERS)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("没有明显变化", response.text)


class LiveHistoryEngine(unittest.TestCase):
    def test_translation_and_brightness_are_not_changes(self):
        baseline = textured_scene()
        current = cv2.warpAffine(baseline, np.float32([[1, 0, 8], [0, 1, 5]]), (640, 480))
        for image in (current, baseline + 15):
            with self.subTest(kind="transformation"):
                result = history.compare_geometry(baseline, image)
                self.assertEqual(result["status"], "matched")
                self.assertEqual(result["polygons"], [])

    def test_two_changes_keep_distinct_geometry(self):
        baseline = textured_scene()
        current = baseline.copy()
        current[100:170, 150:240] = [15, 35, 70]
        current[280:350, 390:490] = [15, 35, 70]
        result = history.compare_geometry(baseline, current)
        self.assertEqual(len(result["polygons"]), 2)
        centers = [np.mean(polygon, axis=0) for polygon in result["polygons"]]
        self.assertTrue(any(abs(x - 195/640) < 0.04 and abs(y - 135/480) < 0.04 for x, y in centers))
        self.assertTrue(any(abs(x - 440/640) < 0.04 and abs(y - 315/480) < 0.04 for x, y in centers))

    def test_invalid_image_rejected(self):
        with self.assertRaises(ValueError):
            history.decode_image(b"invalid synthetic image")

    def test_second_batch_change_is_not_lost(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"status": "matched", "polygons": [[[.1, .1], [.4, .1], [.4, .4]]] * 9,
                "_aligned": image, "_target": image, "diff_summary": "initial"}
        batches = [[{"id": i+1, "decision": "ignore", "summary": "无关"} for i in range(8)],
                   [{"id": 1, "decision": "keep", "summary": "新增光伏板"}]]
        responses = [model_response({"message": {"content": json.dumps({"regions": rows})}}) for rows in batches]
        with mock.patch.object(httpx.HTTPTransport, "handle_request", side_effect=responses) as upstream:
            outcome = history.filter_task_changes(data, base_url="http://model.invalid", model="synthetic", timeout=1,
                                                  task_description="只比较光伏板")
        self.assertEqual(upstream.call_count, 2)
        self.assertEqual(len(data["polygons"]), 1)
        self.assertIsNone(outcome["error"])


def load_existing(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def main():
    if os.getenv("REGRESSION_CANDIDATE_HASHES"):
        candidate = json.loads(Path(os.environ["REGRESSION_CANDIDATE_HASHES"]).read_text())
        allowed = {"/app/api/vl_compare.py", "/app/history_compare_runtime/services/history_compare.py"}
        if set(candidate) != allowed:
            raise SystemExit("candidate must replace exactly the two reviewed runtime files")
        SOURCE_HASHES.update(candidate)
    for path, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit("source digest mismatch: " + path)
    suite = unittest.TestSuite()
    suite.addTests(load_existing("/app/tests/test_review_agent_routes.py"))
    suite.addTests(load_existing("/app/tests/test_mask_for_draw.py"))
    for cls in (LiveReviewHTTP, LiveHistoryHTTP, LiveHistoryEngine):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    import torch
    started = time.monotonic()
    real_connect = socket.socket.connect
    def forbid_network(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("real network forbidden in regression")
        return real_connect(sock, address)
    with mock.patch.object(socket.socket, "connect", forbid_network), \
         mock.patch.object(torch, "load", side_effect=AssertionError("model weight load forbidden")):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                      "skipped": len(result.skipped), "seconds": round(time.monotonic()-started, 3),
                      "source_hashes": SOURCE_HASHES, "network": "none", "gpu": "none",
                      "model_http": "stubbed", "shared_database": "unmounted and disabled"}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
