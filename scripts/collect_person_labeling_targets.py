#!/usr/bin/env python3
"""从 09-24 测试环境全量快照中挑出人员候选框，生成高清取图任务（在服务器运行）。

只使用已核验的视频/原图匹配（clean_usable_for_inference）；无高清来源、已有人工金标、
同图重复的框只计数不入包。输出 targets.jsonl / jobs.jsonl，供 extract_hires_review_views.py 使用。
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

CODEX = Path("/REDACTED_RUNTIME_ROOT/codex-ktd-real-monitoring-20260924")
CLASS_NAME = "人员"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def norm_box(cand: dict) -> list[float]:
    w, h = cand["frame_size"]
    x1, y1, x2, y2 = cand["bbox_px"]
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True,
                        help="已有人工金标 jsonl（sample_id/image_sha256/bbox，归一化）")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    gold_overlap = {row["full_candidate_id"] for row in read_jsonl(CODEX / "input/gold_overlap_map.jsonl")}
    matches = {}
    for name in ("still_matches.jsonl", "video_matches.jsonl"):
        for row in read_jsonl(CODEX / "output" / name):
            if row.get("clean_usable_for_inference") and row["event_id"] not in matches:
                matches[row["event_id"]] = row

    people = [row for row in read_jsonl(CODEX / "input/inference_manifest.jsonl") if row["class_name"] == CLASS_NAME]
    targets, jobs, skipped, kept_boxes = [], {}, collections.Counter(), []
    for cand in sorted(people, key=lambda r: (r["event_time"], r["candidate_id"])):
        box = norm_box(cand)
        if cand["candidate_id"] in gold_overlap or any(
                (g["sample_id"] == cand["event_id"] or g["image_sha256"] == cand["image_sha256"])
                and iou(g["bbox"], box) >= 0.5 for g in gold):
            skipped["already_gold"] += 1
            continue
        # 同一张图被重复建成多个事件时，只保留一次
        if any(k["image_sha256"] == cand["image_sha256"] and iou(k["box"], box) >= 0.5 for k in kept_boxes):
            skipped["duplicate_image_box"] += 1
            continue
        match = matches.get(cand["event_id"])
        if match is None:
            skipped["no_verified_hires_source"] += 1
            continue
        kept_boxes.append({"image_sha256": cand["image_sha256"], "box": box})
        eid = cand["event_id"]
        job = jobs.setdefault(eid, {
            "event_id": eid, "source_kind": match["source_kind"],
            "source_path": "codex/" + match["source_local_path"], "matched_time": match.get("matched_raw_time"),
            "homography": match["homography_annotated_to_source"], "clean_frame": "codex/" + match["clean_frame"],
            "low_clean": "codex/" + match["clean_frame"], "low_annotated": f"codex/work/annotated/{eid}.jpg",
            "candidates": []})
        slug = cand["candidate_id"].replace(":", "-")
        job["candidates"].append({"candidate_id": cand["candidate_id"], "slug": slug, "bbox_px": cand["bbox_px"]})
        targets.append({
            "candidate_id": cand["candidate_id"], "event_id": eid, "task_id": cand["task_id"],
            "event_time": cand["event_time"], "class_name": CLASS_NAME,
            "detector_confidence": cand["detector_confidence"], "bbox_px": cand["bbox_px"],
            "bbox_norm": box, "frame_size": cand["frame_size"], "image_sha256": cand["image_sha256"],
            "source_kind": match["source_kind"], "clean_match_status": match.get("status"), "slug": slug,
        })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "targets.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in targets), encoding="utf-8")
    (args.out / "jobs.jsonl").write_text(
        "".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs.values()), encoding="utf-8")
    print(json.dumps({"person_candidates": len(people), "targets": len(targets), "events": len(jobs),
                      "tasks": len({t["task_id"] for t in targets}), "skipped": skipped,
                      "by_source": collections.Counter(t["source_kind"] for t in targets),
                      "by_frame": collections.Counter("x".join(map(str, t["frame_size"])) for t in targets)},
                     ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
