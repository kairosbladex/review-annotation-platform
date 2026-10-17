#!/usr/bin/env python3
"""把测试环境复检“确认”（放过）的目标对应到高清原片，生成高清取图任务（在服务器运行）。

输入：
  --recheck-events  bo_task_event 只读导出（id/event_time/recheck_time/recheck_result）
  --annex           无快照高清源事件的 sys_annex_file 只读导出（PRIMARYIMG 为无人机原图时直接使用）
来源：
  snapshot  09-24 快照事件：按归一化框 IoU 对应到清单候选，使用已核验的视频/原图匹配（单应矩阵）。
  original  事件主图即 4032×3024 原图：使用复检记录中的 recoveredBbox（原图归一化坐标），单位矩阵。
无高清来源的目标只记录，不入包。
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

CODEX = Path("/REDACTED_RUNTIME_ROOT/codex-ktd-real-monitoring-20260924")
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
CURRENT_PROMPT = "2cbeee9e"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recheck-events", type=Path, required=True)
    parser.add_argument("--annex", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True, help="原图事件下载目录（primaryimg/shotimg）")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = {row["event_id"]: row for row in read_jsonl(CODEX / "input/source_manifest.jsonl")}
    matches = {}
    for name in ("still_matches.jsonl", "video_matches.jsonl"):
        for row in read_jsonl(CODEX / "output" / name):
            if row.get("clean_usable_for_inference") and row["event_id"] not in matches:
                matches[row["event_id"]] = row
    annex = collections.defaultdict(dict)
    for row in json.loads(args.annex.read_text(encoding="utf-8"))["annex"]:
        annex[row["busi_id"]][row["busi_type_code"]] = row

    targets, jobs, skipped = [], {}, collections.Counter()
    for event in json.loads(args.recheck_events.read_text(encoding="utf-8")):
        eid = event["id"]
        result = json.loads(event["recheck_result"])
        details = result.get("results") or []
        for index, item in enumerate(result["resultList"]):
            if item["decision"] != "confirmed":
                continue
            clean_source = (details[index].get("cleanSource") if index < len(details) else None) or {}
            version = "current_clean_source_qwen" if str(clean_source.get("promptSha256", "")).startswith(CURRENT_PROMPT) \
                else "older"
            target = {
                "target_id": f"{eid}:r{index:03d}", "event_id": eid, "event_no": event.get("event_no"),
                "recheck_time": event["recheck_time"], "recheck_version": version,
                "class_name": item["originalClass"], "recheck_reasoning": item.get("reasoning"),
                "original_bbox_norm": item["originalBbox"],
            }
            if eid in matches and eid in manifest:
                frame_w, frame_h = manifest[eid]["candidates"][0]["frame_size"]
                best, best_iou = None, 0.0
                for cand in manifest[eid]["candidates"]:
                    x1, y1, x2, y2 = cand["bbox_px"]
                    score = iou(item["originalBbox"], [x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h])
                    if score > best_iou:
                        best, best_iou = cand, score
                if best is None or best_iou < 0.5:
                    skipped["snapshot_bbox_unmatched"] += 1
                    continue
                match = matches[eid]
                target.update(source="snapshot", candidate_id=best["candidate_id"], bbox_match_iou=round(best_iou, 4),
                              bbox_px=best["bbox_px"], frame_size=[frame_w, frame_h])
                job = jobs.setdefault(eid, {
                    "event_id": eid, "source_kind": match["source_kind"],
                    "source_path": "codex/" + match["source_local_path"], "matched_time": match.get("matched_raw_time"),
                    "homography": match["homography_annotated_to_source"], "clean_frame": "codex/" + match["clean_frame"],
                    "low_clean": "codex/" + match["clean_frame"], "low_annotated": f"codex/work/annotated/{eid}.jpg",
                    "candidates": []})
            elif clean_source.get("geometryRecovery", {}).get("status") == "recovered" \
                    and "aiEventImg" not in annex[eid].get("PRIMARYIMG", {}).get("file_path", "aiEventImg"):
                geo = clean_source["geometryRecovery"]
                width, height = geo["cleanSize"]
                bx = geo["recoveredBbox"]
                target.update(source="original", candidate_id=target["target_id"], bbox_match_iou=None,
                              bbox_px=[bx[0] * width, bx[1] * height, bx[2] * width, bx[3] * height],
                              frame_size=[width, height])
                job = jobs.setdefault(eid, {
                    "event_id": eid, "source_kind": "still", "source_path": f"original/primaryimg/{eid}.jpg",
                    "matched_time": None, "homography": IDENTITY, "clean_frame": None,
                    "low_clean": None, "low_annotated": None, "candidates": []})
            else:
                skipped["no_hires_source"] += 1
                continue
            slug = target["target_id"].replace(":", "-")
            if any(c["candidate_id"] == target["candidate_id"] for c in job["candidates"]):
                skipped["duplicate_box_in_event"] += 1
                continue
            job["candidates"].append({"candidate_id": target["candidate_id"], "slug": slug, "bbox_px": target["bbox_px"]})
            target["slug"] = slug
            targets.append(target)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "targets.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in targets), encoding="utf-8")
    (args.out / "jobs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs.values()), encoding="utf-8")
    print(json.dumps({"targets": len(targets), "events": len(jobs), "skipped": skipped,
                      "by_source": collections.Counter(t["source"] for t in targets),
                      "by_version": collections.Counter(t["recheck_version"] for t in targets),
                      "by_class": collections.Counter(t["class_name"] for t in targets)}, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
