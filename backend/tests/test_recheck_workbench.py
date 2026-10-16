"""Exercise the real workbench GET with synthetic data at the MySQL boundary."""
import copy
import importlib.util
import json
import socket
from pathlib import Path

import pymysql
import pytest
from fastapi.testclient import TestClient


EVENT_ID = "e" * 32
PROJECT_ID = "synthetic-project"


class SyntheticDatabase:
    def __init__(self):
        self.results = {}
        self.statements = []
        self.rollbacks = 0
        self.failure = False

    def connect(self, **config):
        assert config["host"] == "synthetic.invalid"
        assert config["user"] == "synthetic-user"
        assert config["password"] == "synthetic-unused-password"
        assert config["database"] == "synthetic-db"
        if self.failure:
            raise pymysql.OperationalError("synthetic database unavailable")
        return SyntheticConnection(self)

    def rows(self, sql, args):
        statement = " ".join(sql.split())
        self.statements.append((statement, args))
        if statement in {"SET TRANSACTION READ ONLY", "START TRANSACTION READ ONLY"}:
            return []
        assert statement.startswith("SELECT "), "A GET must not write business data"
        if "FROM bo_task_event_image_recheck_result" in statement:
            assert args[0] == EVENT_ID
            return [{"recheck_result": json.dumps(self.results[args[1]]), "recheck_time": "2026-01-01"}]
        if "SELECT * FROM bo_task_event " in statement:
            assert args == (EVENT_ID, PROJECT_ID)
            return [{"id": EVENT_ID, "event_no": "synthetic-event", "upload_data_type": "synthetic-type",
                     "upload_data_type_name": "合成类型", "recheck_status": "2", "recheck_result": "{}"}]
        if "FROM bo_early_warning_customize " in statement:
            assert args == (PROJECT_ID,)
            return [{"code": "synthetic-type"}]
        if "FROM sys_annex_file" in statement and "busi_id=%s" in statement:
            assert args == (EVENT_ID,)
            return [{"id": identifier, "file_path": "/synthetic.png", "imgxy": "[]",
                     "image_resolution": "160x120", "busi_type_code": "SHOTIMG"}
                    for identifier in self.results]
        if "FROM sys_annex_source_target" in statement:
            return []
        raise AssertionError(f"Unexpected database read: {statement}")


class SyntheticConnection:
    def __init__(self, database):
        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def cursor(self):
        return SyntheticCursor(self.database)

    def rollback(self):
        self.database.rollbacks += 1


class SyntheticCursor:
    def __init__(self, database):
        self.database = database
        self.result = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, args=()):
        self.result = self.database.rows(sql, args)

    def fetchall(self):
        return copy.deepcopy(self.result)


@pytest.fixture
def workbench(monkeypatch, tmp_path):
    def forbidden_network(*_args, **_kwargs):
        raise AssertionError("Network and model calls are forbidden")

    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, forbidden_network)
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, forbidden_network)
    database = SyntheticDatabase()
    monkeypatch.setattr(pymysql, "connect", database.connect)
    for key, value in {
        "DB_HOST": "synthetic.invalid", "DB_PORT": "3306", "DB_USER": "synthetic-user",
        "DB_PASSWORD": "synthetic-unused-password", "DB_NAME": "synthetic-db",
        "PROJECT_ID": PROJECT_ID, "DATA_DIR": str(tmp_path / "private-workbench"),
        "ALLOW_SUBMIT": "false", "BEARER_TOKEN": "", "VERIFY_TLS": "true",
    }.items():
        monkeypatch.setenv("KTD_REVIEW_" + key, value)
    source = Path(__file__).resolve().parents[1] / "app/recheck_workbench.py"
    spec = importlib.util.spec_from_file_location("workbench_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        yield client, database
    assert database.rollbacks == sum(sql.startswith("SELECT ") for sql, _ in database.statements)
    assert not any((tmp_path / "private-workbench/submissions").iterdir())


def read_candidate(workbench, record):
    client, database = workbench
    database.results["synthetic-image"] = {"results": [record]}
    response = client.get(f"/workbench-api/events/{EVENT_ID}")
    assert response.status_code == 200
    assert response.json()["submitEnabled"] is False
    return response.json()["images"][0]["candidates"][0]


@pytest.mark.parametrize("raw,legacy,expected", [
    ("true", "rejected", "confirmed"),
    ("false", "confirmed", "rejected"),
    ("uncertain", "rejected", "uncertain"),
])
def test_top_level_raw_decision_controls_display(workbench, raw, legacy, expected):
    candidate = read_candidate(workbench, {
        "decision": legacy, "rawDecision": raw,
        "codexReview": {"rawDecision": "false" if raw != "false" else "true"},
    })
    assert candidate["decision"] == expected


@pytest.mark.parametrize("raw,expected", [("true", "confirmed"), ("false", "rejected"), ("uncertain", "uncertain")])
@pytest.mark.parametrize("top_level", [{}, {"rawDecision": None}])
def test_nested_raw_decision_when_top_level_absent(workbench, raw, expected, top_level):
    candidate = read_candidate(workbench, {
        "decision": "pending", **top_level, "codexReview": {"rawDecision": raw},
    })
    assert candidate["decision"] == expected


@pytest.mark.parametrize("record,expected", [
    ({"decision": "confirmed"}, "confirmed"),
    ({"decision": "rejected"}, "rejected"),
    ({"decision": "uncertain"}, "uncertain"),
    ({"decision": "corrected"}, "pending"),
    ({}, "pending"),
    ({"decision": "rejected", "rawDecision": "unknown", "codexReview": {"rawDecision": "true"}}, "rejected"),
    ({"decision": "rejected", "rawDecision": True}, "rejected"),
    ({"decision": "rejected", "rawDecision": []}, "rejected"),
    ({"decision": "rejected", "rawDecision": {}}, "rejected"),
    ({"decision": "rejected", "codexReview": "malformed"}, "rejected"),
    ({"decision": "rejected", "codexReview": {"rawDecision": []}}, "rejected"),
    ({"decision": "rejected", "reasoning": "看不清 uncertain", "reasonCode": "codex_luna_uncertain"}, "rejected"),
])
def test_legacy_and_invalid_raw_preserve_existing_contract(workbench, record, expected):
    assert read_candidate(workbench, record)["decision"] == expected


def test_multiple_images_keep_raw_results_summary_order_and_geometry(workbench):
    client, database = workbench
    first = {"results": [{"target_index": 7, "original_class": "人员", "original_bbox": [0.1, 0.2, 0.3, 0.6],
                           "decision": "rejected", "rawDecision": "uncertain", "businessAction": "candidate_drop"},
                          {"target_index": 2, "original_class": "车", "original_bbox": [0.4, 0.3, 0.8, 0.9],
                           "decision": "confirmed", "rawDecision": "true", "businessAction": "keep"}],
             "summary": {"trueCount": 1, "falseCount": 1, "uncertain": 0, "doubtfulCount": 0}}
    second = {"results": [{"original_class": "卡车", "original_bbox": [0.2, 0.1, 0.5, 0.7],
                            "decision": "rejected", "rawDecision": "false", "businessAction": "candidate_drop"}],
              "summary": {"trueCount": 0, "falseCount": 1, "uncertain": 0}}
    database.results = {"first-image": copy.deepcopy(first), "second-image": copy.deepcopy(second)}
    response = client.get(f"/workbench-api/events/{EVENT_ID}")
    assert response.status_code == 200
    images = response.json()["images"]
    assert [image["id"] for image in images] == ["first-image", "second-image"]
    assert [[candidate["decision"] for candidate in image["candidates"]] for image in images] == [["uncertain", "confirmed"], ["rejected"]]
    for image, expected in zip(images, [first, second]):
        assert image["rawResult"] == expected
        assert image["summary"] == expected["summary"]
        assert [candidate["className"] for candidate in image["candidates"]] == [row["original_class"] for row in expected["results"]]
        assert [candidate["bbox"] for candidate in image["candidates"]] == [row["original_bbox"] for row in expected["results"]]
    assert database.results == {"first-image": first, "second-image": second}


def test_database_failure_is_not_a_false_target_result(workbench):
    client, database = workbench
    database.failure = True
    response = client.get(f"/workbench-api/events/{EVENT_ID}")
    assert response.status_code == 503
    assert "images" not in response.json()
    assert "results" not in response.json()


def test_invalid_event_id_is_rejected_before_database(workbench):
    client, database = workbench
    response = client.get("/workbench-api/events/not-an-event-id")
    assert response.status_code == 400
    assert database.statements == []
