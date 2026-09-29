#!/usr/bin/env python3
"""从新事件筛选结果中挑出 v3 作用范围内的候选，生成高清取图任务（在服务器工作目录运行）。

v3 作用范围：四视图两路一致确认为车，且支撑面为 building_roof/unknown。
不按部件结果分层，范围内候选全部入包；模型字段只写入 selected.jsonl 供审计。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="claude-vsg-new-events 工作目录")
    args = parser.parse_args()
    root = args.root

    screened = []
    for path in sorted((root / "out").glob("*/screen.jsonl")):
        screened += read_jsonl(path)
    ids = [row["candidate_id"] for row in screened]
    if len(ids) != len(set(ids)):
        raise SystemExit("筛选结果中有重复候选")
    selected = [
        row for row in screened
        if row.get("binary_outcome") == "confirmed" and row.get("surface") in ("building_roof", "unknown")
    ]
    selected.sort(key=lambda row: (row["event_id"], row["candidate_id"]))

    video_matches = {row["event_id"]: row for row in read_jsonl(root / "video-root/output/video_matches.jsonl")}
    jobs: dict[str, dict] = {}
    for row in selected:
        event = row["event_id"]
        if event not in jobs:
            if row["clean_source_kind"] == "still_original":
                jobs[event] = {
                    "event_id": event, "source_kind": "still", "source_path": f"frames/clean/{event}.jpg",
                    "matched_time": None, "homography": IDENTITY, "clean_frame": None, "candidates": [],
                }
            else:
                match = video_matches[event]
                if match.get("status") != "strong" or not match.get("clean_usable_for_inference"):
                    raise SystemExit(f"视频匹配不可用：{event}")
                jobs[event] = {
                    "event_id": event, "source_kind": "video",
                    "source_path": "video-root/" + match["source_local_path"],
                    "matched_time": match["matched_raw_time"],
                    "homography": match["homography_annotated_to_source"],
                    "clean_frame": "video-root/" + match["clean_frame"], "candidates": [],
                }
        jobs[event]["candidates"].append({
            "candidate_id": row["candidate_id"],
            "slug": row["candidate_id"].replace(":", "-"),
            "bbox_px": row["bbox_px"],
        })

    out = root / "package"
    out.mkdir(exist_ok=True)
    (out / "jobs.jsonl").write_text("".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs.values()), encoding="utf-8")
    (out / "selected.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    print(json.dumps({"screened": len(screened), "selected": len(selected), "events": len(jobs),
                      "video_events": sum(job["source_kind"] == "video" for job in jobs.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
