#!/usr/bin/env python3
"""Drive the isolated backend through five-class ingest and auto-annotation APIs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path("/state")
OBJECT_ROOT = ROOT / "objects" / "verify-source"
DB_PATH = ROOT / "verify.sqlite3"
BASE_URL = "http://127.0.0.1:18392"


def _request(method: str, path: str, body: dict | None = None, headers: dict | None = None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=600) as response:
            return response.status, json.loads(response.read() or b"{}")
    except HTTPError as exc:
        payload = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {path} -> {exc.code}: {payload}") from exc


def _login() -> str:
    status, body = _request(
        "POST",
        "/api/v1/auth/login",
        {
            "username": os.environ.get("ADMIN_USERNAME", "admin"),
            "password": os.environ["ADMIN_PASSWORD"],
        },
    )
    if status != 200:
        raise RuntimeError(f"login failed: {status} {body}")
    return body["access_token"]


def _image_sha256(name: str) -> str:
    return hashlib.sha256((OBJECT_ROOT / f"{name}.jpg").read_bytes()).hexdigest()


def _sample(name: str, event_id: str, source_event: str, targets: list[dict], note: str) -> dict:
    return {
        "name": name,
        "event_id": event_id,
        "source_event": source_event,
        "note": note,
        "targets": targets,
    }


def _target(index: int, class_name: str, confidence: float, bbox: list[float]) -> dict:
    return {
        "target_index": index,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,
        "reasoning": "KTD real-event verification sample",
    }


def _samples() -> list[dict]:
    return [
        _sample(
            "person-tier1",
            "verify-20261023-person-tier1",
            "1e379588d22e4580a980a067bf6a81a7",
            [_target(0, "人员", 0.8625, [0.828125, 0.358333, 0.911458, 0.488889])],
            "real person frame; detector score >= 0.85",
        ),
        _sample(
            "vehicle-tier1",
            "verify-20261023-vehicle-tier1",
            "0000e0fd01584ef89ff9a62041dff772",
            [_target(0, "车", 0.8558, [0.614583, 0.251389, 0.64375, 0.276389])],
            "real vehicle frame; detector score >= 0.85",
        ),
        _sample(
            "vehicle-tier25-confirm",
            "verify-20261023-vehicle-tier25-confirm",
            "1679d30c1c204528886cf81618dd132a",
            [
                _target(
                    0,
                    "车",
                    0.8376206755638123,
                    [
                        0.0037202380952380953,
                        0.1693121693121693,
                        0.022569444444444444,
                        0.21924603174603175,
                    ],
                )
            ],
            "real vehicle frame; human-gold true candidate, calibrated Jev-Omni P(true) >= 0.60",
        ),
        _sample(
            "vehicle-tier25-reject",
            "verify-20261023-vehicle-tier25-reject",
            "feb66a418f40409e949d0797105ff240",
            [_target(0, "车", 0.6514, [0.504712, 0.031085, 0.518104, 0.051587])],
            "real vehicle frame; human-gold false candidate, Jev-Omni P(true) <= 0.01",
        ),
        _sample(
            "tricycle-tier3-gray",
            "verify-20261023-tricycle-tier3-gray",
            "0131614611134eb6b8633658f24c5c44",
            [_target(0, "三轮车", 0.6206, [0.408333, 0.669444, 0.427083, 0.731944])],
            "real tricycle frame; Jev-Omni P(true) in gray zone",
        ),
        _sample(
            "blue-roof-tier3-forced",
            "verify-20261023-blue-roof-tier3-forced",
            "01279be3d81e4185a89ee2ea47b0fbe6",
            [_target(0, "违建", 0.8524, [0.0, 0.854167, 0.047917, 0.997222])],
            "real illegal-roof frame; score >= Tier1 but scene-dependent class forces Tier3",
        ),
        _sample(
            "road-crack-tier3",
            "verify-20261023-road-crack-tier3",
            "0010a6887fa5454da03a424b2ad01866",
            [_target(0, "道路裂纹", 0.6531, [0.480208, 0.308333, 0.573958, 0.320833])],
            "real road-crack frame; class outside Tier1 allowlist",
        ),
        _sample(
            "vehicle-tricycle-geometric-conflict",
            "verify-20261023-vehicle-tricycle-geometric-conflict",
            "0a8e5e3df22d4a9084b9584b432ae52f",
            [
                _target(0, "车", 0.8176, [0.520833, 0.65, 0.536458, 0.705556]),
                _target(1, "三轮车", 0.4436, [0.520833, 0.65, 0.536458, 0.705556]),
            ],
            "real frame; tricycle bbox controlled to overlap vehicle bbox (geometric gate fixture)",
        ),
        _sample(
            "vehicle-tier2-reject",
            "verify-20261023-vehicle-tier2-reject",
            "feb66a418f40409e949d0797105ff240",
            [_target(0, "车", 0.05, [0.504712, 0.031085, 0.518104, 0.051587])],
            "real vehicle frame; controlled detector score <= 0.12 for Tier2 boundary",
        ),
    ]


def _ingest(samples: list[dict], token: str) -> dict[str, str]:
    sample_ids: dict[str, str] = {}
    for item in samples:
        payload = {
            "event_id": item["event_id"],
            "source_service": "ktd-five-class-verify",
            "session_id": "verify-20261023",
            "image": {
                "bucket": "verify-source",
                "object_key": f"{item['source_event']}.jpg",
                "sha256": _image_sha256(item["source_event"]),
            },
            "targets": item["targets"],
            "metadata": {
                "source_manifest_event_id": item["source_event"],
                "verification_note": item["note"],
                "branch": "docs/luna-business-test-release",
                "commit": "90e9d054",
            },
        }
        status, body = _request(
            "POST",
            "/api/v1/ingest/review-events",
            payload,
            {"X-Service-Key": os.environ["SERVICE_API_KEY"]},
        )
        if status not in {200, 201}:
            raise RuntimeError(f"ingest {item['name']} failed: {status} {body}")
        sample_ids[item["name"]] = body["sample_id"]
        print(f"[ingest] {item['name']}: {status} sample={body['sample_id']}", flush=True)
    return sample_ids


def _run_job(sample_ids: list[str], token: str) -> dict:
    status, body = _request(
        "POST",
        "/api/v1/auto-annotations",
        {"sample_ids": sample_ids},
        {"Authorization": f"Bearer {token}"},
    )
    if status not in {200, 202}:
        raise RuntimeError(f"create auto annotation failed: {status} {body}")
    job_id = body["job_id"]
    print(f"[auto-annotation] job={job_id} initial={body}", flush=True)
    for _ in range(120):
        status, job = _request(
            "GET",
            f"/api/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if status == 200 and job.get("status") in {"succeeded", "failed", "canceled"}:
            print(f"[auto-annotation] final={job}", flush=True)
            return job
        time.sleep(1)
    raise RuntimeError(f"job {job_id} did not finish in 120s")


def _read_records() -> list[dict]:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT s.source_event_id, s.metadata_json, c.id AS candidate_id,
               c.target_index, c.class_name, c.confidence,
               c.x1, c.y1, c.x2, c.y2,
               a.label, a.note,
               ae.action, ae.payload_json
        FROM samples s
        JOIN candidates c ON c.sample_id = s.id
        LEFT JOIN annotations a ON a.candidate_id = c.id
        LEFT JOIN audit_events ae
          ON ae.entity_id = c.id AND ae.action LIKE 'auto_annotation.%'
        WHERE s.source_event_id LIKE 'verify-20261023-%'
        ORDER BY s.created_at, c.target_index, ae.created_at
        """
    ).fetchall()
    connection.close()
    records = []
    for row in rows:
        payload = json.loads(row["payload_json"] or "{}")
        records.append(
            {
                "source_event_id": row["source_event_id"],
                "candidate_id": row["candidate_id"],
                "target_index": row["target_index"],
                "class_name": row["class_name"],
                "confidence": row["confidence"],
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
                "label": row["label"],
                "note": row["note"],
                "audit_action": row["action"],
                "cascade_routing": payload.get("cascade_routing"),
                "policy": payload.get("policy"),
                "jev_omni": (payload.get("cascade_routing") or {}).get("jev_omni"),
            }
        )
    return records


def _validate(records: list[dict]) -> dict:
    by_key = {(r["source_event_id"], r["target_index"]): r for r in records}
    checks = {}

    def probability(record: dict) -> float | None:
        routing = record["cascade_routing"] or {}
        nested = routing.get("jev_omni") or {}
        value = routing.get("p_true", nested.get("p_true"))
        return None if value is None else float(value)

    def check(name: str, key: tuple[str, int], *, tier: str, reason: str, label: str):
        record = by_key[key]
        routing = record["cascade_routing"] or {}
        checks[name] = {
            "passed": routing.get("tier") == tier
            and routing.get("reason_code") == reason
            and record["label"] == label,
            "tier": routing.get("tier"),
            "reason_code": routing.get("reason_code"),
            "label": record["label"],
            "jev_omni": record["jev_omni"],
            "p_true": probability(record),
        }

    check(
        "person_tier1",
        ("verify-20261023-person-tier1", 0),
        tier="tier1_auto_confirm",
        reason="tier1_high_confidence_auto_confirm",
        label="true",
    )
    check(
        "vehicle_tier1",
        ("verify-20261023-vehicle-tier1", 0),
        tier="tier1_auto_confirm",
        reason="tier1_high_confidence_auto_confirm",
        label="true",
    )
    check(
        "vehicle_tier25_confirm",
        ("verify-20261023-vehicle-tier25-confirm", 0),
        tier="tier2_5_jev_omni_confirm",
        reason="tier2_5_jev_omni_calibrated_confirm",
        label="true",
    )
    check(
        "vehicle_tier25_reject",
        ("verify-20261023-vehicle-tier25-reject", 0),
        tier="tier2_5_jev_omni_reject",
        reason="tier2_5_jev_omni_calibrated_reject",
        label="false",
    )
    check(
        "tricycle_tier3_gray",
        ("verify-20261023-tricycle-tier3-gray", 0),
        tier="tier3_vlm_verify",
        reason="tier3_routed_to_l20_vlm",
        label="true",
    )
    check(
        "blue_roof_tier3_forced",
        ("verify-20261023-blue-roof-tier3-forced", 0),
        tier="tier3_vlm_verify",
        reason="tier3_scene_dependent_forced_vlm",
        label="true",
    )
    check(
        "road_crack_tier3",
        ("verify-20261023-road-crack-tier3", 0),
        tier="tier3_vlm_verify",
        reason="tier3_road_crack_never_auto_confirmed",
        label="true",
    )
    for index in (0, 1):
        check(
            f"geometric_conflict_{index}",
            ("verify-20261023-vehicle-tricycle-geometric-conflict", index),
            tier="tier3_vlm_verify",
            reason="tier3_geometric_conflict",
            label="false",
        )
    check(
        "vehicle_tier2_reject",
        ("verify-20261023-vehicle-tier2-reject", 0),
        tier="tier2_auto_reject",
        reason="tier2_low_confidence_auto_reject",
        label="false",
    )

    routing_complete = [
        bool(record["cascade_routing"])
        and all(
            key in record["cascade_routing"]
            for key in ("tier", "requires_vlm", "reason_code", "class_name")
        )
        for record in records
    ]
    blue = checks["blue_roof_tier3_forced"]
    tier25_confirm = checks["vehicle_tier25_confirm"]
    tier25_reject = checks["vehicle_tier25_reject"]
    p_confirm = tier25_confirm["p_true"]
    p_reject = tier25_reject["p_true"]
    checks["routing_fields_complete"] = {
        "passed": bool(records) and all(routing_complete),
        "records": len(records),
        "complete": sum(routing_complete),
    }
    checks["blue_roof_forced_tier3_100pct"] = {
        "passed": blue["passed"] and blue["tier"] == "tier3_vlm_verify",
        "observed": blue["tier"],
    }
    checks["tier25_confirm_threshold"] = {
        "passed": p_confirm is not None and float(p_confirm) >= 0.60,
        "p_true": p_confirm,
    }
    checks["tier25_reject_threshold"] = {
        "passed": p_reject is not None and float(p_reject) <= 0.01,
        "p_true": p_reject,
    }
    checks["tier1_no_jev_call"] = {
        "passed": checks["person_tier1"]["jev_omni"] is None
        and checks["vehicle_tier1"]["jev_omni"] is None,
    }
    return checks


def main() -> int:
    token = _login()
    samples = _samples()
    sample_ids = _ingest(samples, token)
    job = _run_job(list(sample_ids.values()), token)
    records = _read_records()
    checks = _validate(records)
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": BASE_URL,
        "job": job,
        "sample_ids": sample_ids,
        "checks": checks,
        "records": records,
    }
    output = Path("/state/logs/five_class_e2e.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "record_count": len(records)}, ensure_ascii=False, indent=2))
    if job.get("status") != "succeeded":
        return 2
    return 0 if all(item.get("passed") for item in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
