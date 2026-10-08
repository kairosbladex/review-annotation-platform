#!/usr/bin/env python3
"""给“视频事件由复核服务自己找回干净帧”（clean_video_sources）的离线回放打分。

人员用 person-labeling-20261008 的人工标签，车辆用 random-traffic 高清标注包的人工标签（只算“车”类框）。
对照组：旧版单图复核（线上视频事件实际走的路径）；人员另对照“已核验高清原片”回放（找帧理想上限）。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

TRUE_LABELS = {"真目标", "真车"}
FALSE_LABELS = {"误检", "非车"}


def load(out_dir: Path) -> tuple[dict[str, str], dict[str, dict]]:
    """返回 {candidate_id: decision} 与 {event_id: clean_source_video 审计}。"""
    decisions, events = {}, {}
    for path in sorted(out_dir.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        resp = item["response"]
        for cid, res in zip(item["candidate_ids"], resp.get("results") or []):
            decisions[cid] = res["decision"]
        if "clean_source_video" in resp:
            # _elapsed：整个复核请求耗时（含找帧与模型推理），秒
            events[item["event_id"]] = dict(resp["clean_source_video"] or {}, _elapsed=item.get("elapsed"))
    return decisions, events


def score(labels: dict[str, bool], decisions: dict[str, str]) -> dict:
    c = collections.Counter()
    for cid, truth in labels.items():
        if cid in decisions:
            c[(truth, decisions[cid])] += 1
    pos = sum(v for (t, _), v in c.items() if t)
    neg = sum(v for (t, _), v in c.items() if not t)
    right = c[(True, "confirmed")] + c[(False, "rejected")]
    wrong = c[(True, "rejected")] + c[(False, "confirmed")]
    return {"n": pos + neg, "right": right, "wrong": wrong, "manual": pos + neg - right - wrong,
            "true_target": {"total": pos, "confirmed": c[(True, "confirmed")], "wrongly_rejected": c[(True, "rejected")]},
            "false_target": {"total": neg, "rejected": c[(False, "rejected")], "wrongly_confirmed": c[(False, "confirmed")]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True, help="selection.json：events + labels")
    parser.add_argument("--video", type=Path, required=True, help="带 clean_video_sources 的回放输出目录")
    parser.add_argument("--vehicle-legacy", type=Path, required=True, help="车辆事件不带视频的旧版回放输出目录")
    parser.add_argument("--person-single", type=Path, required=True)
    parser.add_argument("--person-clean", type=Path, required=True)
    parser.add_argument("--snapshot-status", type=Path, required=True,
                        help="{event_id: 快照离线找帧状态}（clean_usable_for_inference 为真记 usable）")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sel = json.loads(args.selection.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot_status.read_text(encoding="utf-8"))
    video, video_events = load(args.video)
    vehicle_legacy, _ = load(args.vehicle_legacy)
    person_single, _ = load(args.person_single)
    person_clean, _ = load(args.person_clean)
    event_of = {cid: eid for eid, cids in sel["events"].items() for cid in cids}

    report = {"events": len(sel["events"]), "replayed_events": len(video_events)}
    for group, legacy in (("person", person_single), ("vehicle", vehicle_legacy)):
        labels = {cid: lab["label"] in TRUE_LABELS for cid, lab in sel["labels"].items()
                  if lab["group"] == group and lab["label"] in TRUE_LABELS | FALSE_LABELS}
        matched = {cid: t for cid, t in labels.items()
                   if video_events.get(event_of[cid], {}).get("status") == "matched"}
        unmatched = {cid: t for cid, t in labels.items() if cid not in matched}
        g = {"labels": collections.Counter(lab["label"] for lab in sel["labels"].values() if lab["group"] == group),
             "video_path": score(labels, video), "legacy_single_image": score(labels, legacy),
             "matched_boxes": len(matched),
             "video_path_on_matched": score(matched, video), "legacy_on_matched": score(matched, legacy),
             "video_path_on_unmatched": score(unmatched, video), "legacy_on_unmatched": score(unmatched, legacy)}
        if group == "person":
            g["hires_clean_upper_bound"] = score(labels, person_clean)
        # 同一个框：新路径相对旧版是变好、变差还是一样
        delta = collections.Counter()
        for cid, truth in labels.items():
            if cid not in video or cid not in legacy:
                continue
            def ok(d: str) -> str:
                return "right" if d == ("confirmed" if truth else "rejected") else (
                    "wrong" if d in ("confirmed", "rejected") else "manual")
            delta[f"{ok(legacy[cid])}->{ok(video[cid])}"] += 1
        g["legacy_to_video_per_box"] = delta
        report[group] = g

    status = collections.Counter()
    for eid, audit in video_events.items():
        status[(audit.get("status"), audit.get("reasonCode"), snapshot.get(eid, "absent"))] += 1
    report["event_match_vs_snapshot"] = {" | ".join(map(str, k)): v for k, v in status.most_common()}
    times = sorted(a["_elapsed"] for a in video_events.values() if a.get("_elapsed"))
    if times:
        report["request_seconds"] = {"median": round(times[len(times) // 2], 1),
                                     "p90": round(times[int(len(times) * 0.9)], 1), "max": round(times[-1], 1)}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=dict), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1, default=dict))


if __name__ == "__main__":
    main()
