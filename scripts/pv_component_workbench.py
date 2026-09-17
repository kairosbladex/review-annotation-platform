#!/usr/bin/env python3
"""Frozen PV replay and a loopback-only anchor workbench; no business database.

Run --help. The matcher implementation belongs to the AR service and is loaded
explicitly with --matcher-root; it is not copied into this repository.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_matcher(root):
    path = Path(root).resolve() / "pv_component_match.py"
    spec = importlib.util.spec_from_file_location("pv_component_match", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def revision(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def freeze(source, photos, output, matcher_root):
    """Fail before creating output if any source image/hash/box is inconsistent."""
    matcher = load_matcher(matcher_root)
    source, photos, output = Path(source), Path(photos), Path(output)
    if output.exists():
        raise ValueError("Output already exists: snapshots are immutable; choose a new directory")
    rows = read(source)
    cases, payloads, gold = [], [], []
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate event IDs")
    for row in rows:
        event_id = row["id"]
        if not event_id.isalnum():
            raise ValueError("Invalid event ID")
        image_path = photos / (event_id + ".jpg")
        request = row["request"]
        if sha(image_path) != request["photo_camera"]["image_sha256"]:
            raise ValueError("Source image fingerprint mismatch: " + event_id)
        with Image.open(image_path) as image:
            if image.size != (request["image_width"], request["image_height"]):
                raise ValueError("Source image size mismatch")
            if image.getexif().get(274, 1) != 1:
                raise ValueError("Original source orientation must be normalized explicitly")
        blocks = row["blocks"]
        # Local ENU tangent coordinates are sufficient for this bounded rooftop;
        # retain exact source geometry separately for provenance.
        geometry = [json.loads(b["lngLatAlt"])["geometry"]["coordinates"][0] for b in blocks]
        lon0, lat0, z0 = geometry[0][0]
        origin = [lon0, lat0, z0]
        components = []
        for block, polygon in zip(blocks, geometry):
            if len(polygon) == 5 and polygon[0] == polygon[-1]:
                polygon = polygon[:-1]
            if len(polygon) != 4 or block["matrixId"] != row["target_location_id"]:
                raise ValueError("Unexpected component geometry/matrix")
            vertices = [[math.radians(p[0]-lon0)*6378137*math.cos(math.radians(lat0)),
                         math.radians(p[1]-lat0)*6378137, p[2]-z0] for p in polygon]
            components.append(matcher.Component(id=block["id"], matrix_id=block["matrixId"],
                code=block["code"], row=block["blockRow"], column=block["blockColumn"],
                vertices=vertices).model_dump(mode="json"))
        detections = [{"index": i, "bbox_xyxy": box} for i, box in enumerate(request["bboxes_xyxy"])]
        payload = {"image_sha256": sha(image_path), "image_width": request["image_width"],
                   "image_height": request["image_height"], "layout_sha256": matcher.digest(components),
                   "components": components, "detections": detections}
        # Validate the SAME typed request used by the HTTP route, including image bytes.
        matcher.MatchRequest(**payload, image_base64=base64.b64encode(image_path.read_bytes()).decode())
        cases.append({"id": event_id, "source_id": row["source"]["id"], "task_id": row["task_id"],
                      "matrix_id": row["target_location_id"], "matrix_name": row["target_location"],
                      "flight_group": row["task_id"], "split": "development", "origin_lon_lat_alt": origin,
                      "image": "images/" + event_id + ".jpg", "input": "inputs/" + event_id + ".json",
                      "count": len(detections)})
        payloads.append(payload)
        for detection in detections:
            confirmed = event_id == "4e95d41793e143969a61cbc66d285343" and detection["index"] == 1
            gold.append({"case_id": event_id, "index": detection["index"], "split": "development",
                         "status": "confirmed" if confirmed else "unknown",
                         "component_id": "4028ee97a0a8029801a0a81dad230278" if confirmed else None,
                         "source": "user_confirmed_component_identity" if confirmed else None})
    output.mkdir(parents=True)
    for name in ("images", "inputs", "provenance", "calibrations", "runs", "gold"):
        (output / name).mkdir()
    for case, payload, row in zip(cases, payloads, rows):
        shutil.copyfile(photos / (case["id"] + ".jpg"), output / case["image"])
        write(output / case["input"], payload)
        # Old predictions and descriptions never enter inference inputs or UI bootstrap.
        write(output / "provenance" / (case["id"] + ".json"), row)
    write(output / "gold/labels.json", gold)
    hashes = {str(p.relative_to(output)): sha(p) for name in ("images", "inputs", "provenance", "gold")
              for p in (output / name).iterdir()}
    manifest = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_sha256": sha(source), "review_repo_head": revision(Path(__file__).resolve().parents[1]),
                "matcher_repo_head": revision(matcher_root),
                "matcher_source_sha256": sha(Path(matcher_root) / "pv_component_match.py"),
                "case_count": len(cases), "detection_count": sum(c["count"] for c in cases),
                "confirmed_gold_count": sum(g["status"] == "confirmed" for g in gold),
                "hashes": hashes, "cases": cases,
                "gates": {"p1": "needs_independent_human_anchors", "p2": "blocked_by_p1",
                          "p3": "insufficient_independent_gold", "business_write_enabled": False}}
    write(output / "manifest.json", manifest)
    return manifest


def verify(snapshot):
    root = Path(snapshot).resolve()
    manifest = read(root / "manifest.json")
    for name, expected in manifest["hashes"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("Frozen asset changed: " + name)
    cases = manifest["cases"]
    if len(cases) != manifest["case_count"]:
        raise ValueError("Case count mismatch")
    count = 0
    for case in cases:
        payload = read(root / case["input"])
        if len(payload["detections"]) != case["count"]:
            raise ValueError("Detection count mismatch")
        count += case["count"]
    if count != manifest["detection_count"]:
        raise ValueError("Total count mismatch")
    return manifest


def create_app(snapshot, matcher_root):
    root = Path(snapshot).resolve()
    manifest = verify(root)
    matcher = load_matcher(matcher_root)
    matcher_hash = sha(Path(matcher_root) / "pv_component_match.py")
    cases = {c["id"]: c for c in manifest["cases"]}
    app = FastAPI(title="光伏组件锚点工作台（本地验证）")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.include_router(matcher.router)
    token = secrets.token_urlsafe(32)

    def case_input(case_id):
        if case_id not in cases:
            raise HTTPException(404, "Unknown case")
        try:
            verify(root)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        case = cases[case_id]
        return case, read(root / case["input"])

    @app.get("/")
    def page():
        return FileResponse(Path(__file__).with_suffix(".html"))

    @app.get("/api/cases")
    def index():
        return {"cases": list(cases.values()), "token": token, "gates": manifest["gates"]}

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str):
        case, payload = case_input(case_id)
        path = root / "calibrations" / (case_id + ".json")
        return {"case": case, "input": payload, "calibration": read(path) if path.exists() else None}

    @app.get("/api/cases/{case_id}/image")
    def image(case_id: str):
        case, _ = case_input(case_id)
        return FileResponse(root / case["image"])

    @app.post("/api/cases/{case_id}/calibration")
    def calibrate(case_id: str, body: dict, x_workbench_token: str | None = Header(default=None)):
        if x_workbench_token != token:
            raise HTTPException(403, "Local workbench token required")
        case, payload = case_input(case_id)
        try:
            calibration = matcher.Calibration.model_validate(body)
            raw = (root / case["image"]).read_bytes()
            req = matcher.MatchRequest(**payload, image_base64=base64.b64encode(raw).decode(),
                                       calibration=calibration)
            result = matcher.match_components(req)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        # Every attempt is retained; identical retries reuse the same receipt.
        evidence_key = matcher.digest({"request": body, "input": payload,
                                       "matcher": matcher_hash})
        receipt = {"calibration": body, "result": result,
                   "matcher_source_sha256": matcher_hash}
        attempt = root / "runs" / (case_id + "-" + evidence_key + ".json")
        if not attempt.exists():
            write(attempt, receipt)
        write(root / "calibrations" / (case_id + ".json"), body)
        return {"receipt": str(attempt), "result": result}

    return app


def replay(snapshot, matcher_root):
    root = Path(snapshot)
    manifest = verify(root)
    matcher = load_matcher(matcher_root)
    results = []
    for case in manifest["cases"]:
        payload = read(root / case["input"])
        calibration = root / "calibrations" / (case["id"] + ".json")
        req = matcher.MatchRequest(**payload,
            image_base64=base64.b64encode((root / case["image"]).read_bytes()).decode(),
            calibration=read(calibration) if calibration.exists() else None)
        results.append({"case_id": case["id"], "result": matcher.match_components(req)})
    report = {"case_count": len(results), "detection_count": sum(len(c["result"]["results"]) for c in results),
              "matcher_source_sha256": sha(Path(matcher_root) / "pv_component_match.py"),
              "cases": results, "blind_test_passed": False, "business_write_enabled": False}
    destination = root / "runs" / ("replay-" + matcher.digest(report) + ".json")
    if not destination.exists():
        write(destination, report)
    return destination


def assess(snapshot, matcher_root):
    """Development evidence only. Never promote the known sample to blind gold."""
    root = Path(snapshot)
    manifest = verify(root)
    replay_path = replay(root, matcher_root)
    report = read(replay_path)
    predictions = {(c["case_id"], r["index"]): r for c in report["cases"] for r in c["result"]["results"]}
    gold = read(root / "gold/labels.json")
    expected_slots = {(c["id"], d["index"]) for c in manifest["cases"]
                      for d in read(root / c["input"])["detections"]}
    if set(predictions) != expected_slots or {(g["case_id"],g["index"]) for g in gold} != expected_slots:
        raise ValueError("Lost or extra prediction/gold slots")
    known = [g for g in gold if g["status"] == "confirmed"]
    correct = sum(predictions[(g["case_id"],g["index"])]["component_id"] == g["component_id"] for g in known)
    states = {s: sum(r["status"] == s for r in predictions.values())
              for s in ("matched", "ambiguous", "unavailable")}
    # Neighbor checks must be on a verified horizontal AND vertical neighbor,
    # rather than inferred from the target label or fit points themselves.
    neighbor_checks = {}
    for g in known:
        case = next(c for c in manifest["cases"] if c["id"] == g["case_id"])
        components = {c["id"]: c for c in read(root / case["input"])["components"]}
        target = components[g["component_id"]]
        result = next(c["result"] for c in report["cases"] if c["case_id"] == g["case_id"])
        checks = [a for p in result["registration"] if p["status"] == "valid"
                  for a in p["quality"]["anchors"] if a["role"] == "check"]
        neighbors = [components[a["component_id"]] for a in checks]
        neighbor_checks[g["case_id"]] = {
            "horizontal": any(c["row"] == target["row"] and abs(c["column"]-target["column"]) == 1 for c in neighbors),
            "vertical": any(c["column"] == target["column"] and abs(c["row"]-target["row"]) == 1 for c in neighbors)}
    blockers = []
    if not known or correct != len(known): blockers.append("KNOWN_COMPONENT_NOT_MATCHED")
    if not neighbor_checks or not all(all(v.values()) for v in neighbor_checks.values()):
        blockers.append("INDEPENDENT_NEIGHBOR_CHECKS_MISSING")
    result = {"replay": str(replay_path), "case_count": manifest["case_count"],
              "detection_count": manifest["detection_count"], "states": states,
              "development_gold_count": len(known), "development_correct": correct,
              "unknown_gold_count": len(gold)-len(known), "neighbor_checks": neighbor_checks,
              "p1_geometry_ready": not blockers, "p1_blockers": blockers,
              "p1_business_acceptance_passed": False,
              "blind_test": {"passed": False, "confirmed_count": 0,
                             "reason": "CURRENT_SNAPSHOT_IS_DEVELOPMENT_ONLY"},
              "p2_allowed": not blockers, "business_write_enabled": False}
    matcher = load_matcher(matcher_root)
    destination = root / "runs" / ("assessment-" + matcher.digest(result) + ".json")
    if not destination.exists(): write(destination, result)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "verify", "replay", "assess", "serve"])
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--matcher-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--photos", type=Path)
    parser.add_argument("--port", type=int, default=8196)
    args = parser.parse_args()
    if args.command != "verify" and not args.matcher_root:
        parser.error("--matcher-root is required")
    if args.command == "freeze":
        if not args.source or not args.photos:
            parser.error("freeze requires --source and --photos")
        manifest = freeze(args.source, args.photos, args.snapshot, args.matcher_root)
        print(json.dumps({k: manifest[k] for k in ("case_count", "detection_count", "confirmed_gold_count")}))
    elif args.command == "verify":
        manifest = verify(args.snapshot)
        print(json.dumps({"verified": True, "cases": manifest["case_count"], "boxes": manifest["detection_count"]}))
    elif args.command == "replay":
        print(replay(args.snapshot, args.matcher_root))
    elif args.command == "assess":
        print(assess(args.snapshot, args.matcher_root))
    else:
        import uvicorn
        uvicorn.run(create_app(args.snapshot, args.matcher_root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
