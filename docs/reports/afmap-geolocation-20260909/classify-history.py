"""Match retained original requests to immutable DB coordinates and stored image boxes."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
snapshot = json.loads((root / "history-input-snapshot.json").read_text())
recovered = json.loads((root / "history-recovered-requests.json").read_text())
by_event = defaultdict(list)
orphan_rows = []
for row in snapshot["rows"]:
    if row["task_id"] is None: orphan_rows.append(row["id"])
    else: by_event[row["busi_id"]].append(row)

def signature(features):
    return sorted((f["properties"]["index"], *f["geometry"]["coordinates"][:2]) for f in features)

requests = defaultdict(list)
for record in recovered["records"]:
    req = record["request"]
    match = re.search(r"/([a-f0-9]{32})\.log(?:\?|$)", req.get("log_url", ""))
    if not match or not req.get("shot_time_ms"): continue
    when = datetime.fromtimestamp(req["shot_time_ms"] / 1000, timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    requests[(match[1], when)].append(record)

classified = []
candidates = []
for event_id, rows in by_event.items():
    row = rows[0]
    base = {"eventId": event_id, "taskId": row["task_id"], "eventTime": row["event_time"],
            "attachmentIds": [r["id"] for r in rows]}
    try:
        features = json.loads(row["ar_geo_coordinates"])
        metadata = [f.get("properties", {}).get("geolocation") for f in features]
        if metadata and all(metadata):
            status = "already_time_corrected" if all(m.get("location_status") == "time_corrected" for m in metadata) else "new_quality_managed"
            classified.append(dict(base, classification=status)); continue
        assert features and all(not m for m in metadata), "mixed_metadata"
        sig = signature(features)
        matches = []
        seen = set()
        for rec in requests[(row["task_id"], row["event_time"])]:
            response = rec["response"]
            original = sorted((p["index"], p["lon"], p["lat"]) for p in response["results"] if p.get("ok"))
            if sig != original: continue
            key = json.dumps(rec["request"], sort_keys=True)
            if key not in seen: matches.append(rec); seen.add(key)
        if not matches:
            status = "original_request_not_retained" if str(row["event_time"])[:10] < "2026-09-04" else "original_request_not_matched"
            classified.append(dict(base, classification=status)); continue
        assert len(matches) == 1, "ambiguous_original_requests"
        rec = matches[0]; req = rec["request"]
        assert req.get("point_mode", "center") == "center", "unsupported_anchor_mode"
        assert all(req.get(k) is not None for k in ("image_width", "image_height", "focal_mm", "sensor_w_mm", "sensor_h_mm", "zoom")), "missing_original_parameters"
        n = len(req["bboxes_xyxy"])
        original_results = rec["response"]["results"]
        assert rec["response"]["count"] == n and len(original_results) == n, "incomplete_original_response"
        assert {p["index"] for p in original_results} == set(range(n)), "incomplete_original_indices"
        mappings = {}
        for annex in rows:
            assert signature(json.loads(annex["ar_geo_coordinates"])) == sig, "attachment_coordinate_mismatch"
            boxes = json.loads(annex["imgxy"])
            assert isinstance(boxes, list) and len(boxes) == n, "missing_or_incomplete_boxes"
            available = defaultdict(list)
            for i, bbox in enumerate(req["bboxes_xyxy"]): available[tuple(bbox)].append(i)
            mapping = []
            for box in boxes:
                # 旧附件可缺尺寸，但已经恢复的原请求给出了当时使用的精确帧尺寸。
                assert box.get("frame_width") in (None, 0, req["image_width"]) and box.get("frame_height") in (None, 0, req["image_height"]), "frame_size_conflict"
                # 原Java调用按x/y周围100像素构造定位框，保持已验证历史协议。
                bbox = (max(box["x"] - 50, 0), max(box["y"] - 50, 0),
                        min(box["x"] + 50, req["image_width"]), min(box["y"] + 50, req["image_height"]))
                assert len(available[bbox]) == 1, "ambiguous_or_unmatched_box"
                mapping.append(available[bbox].pop())
            mappings[annex["id"]] = mapping
        candidates.append(dict(base, original=rec, storedIndexToRequestIndex=mappings))
        classified.append(dict(base, classification="recoverable_original_request", targetCount=n))
    except (AssertionError, TypeError, ValueError, KeyError) as exc:
        classified.append(dict(base, classification="input_incomplete_or_ambiguous", reason=str(exc)[:180]))

output = {"checkedAt": datetime.now(timezone.utc).isoformat(), "snapshotAt": snapshot["checkedAt"],
          "snapshotSha256": hashlib.sha256((root / "history-input-snapshot.json").read_bytes()).hexdigest(),
          "sourceLogFirstDate": "2026-09-04", "snapshotRows": len(snapshot["rows"]),
          "eventsWithTaskAndStoredLocation": len(by_event), "rowsWithoutTaskAssociation": orphan_rows,
          "counts": dict(Counter(c["classification"] for c in classified)), "events": classified,
          "candidates": candidates, "databaseChanged": False}
(root / "history-classification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: output[k] for k in ("snapshotAt", "snapshotRows", "eventsWithTaskAndStoredLocation", "counts")}, ensure_ascii=False))
