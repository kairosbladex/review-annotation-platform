#!/usr/bin/env python3
"""把测试环境复检放过目标的高清人工标签并入复检训练集的构建输入。

输出 candidate_pairs.jsonl / gold_labels.jsonl，供 prepare_ktd_clean_source_qwen_dataset.py 生成数据集。
约定与现有训练集一致：截图（annotated）用平台事件截图，干净帧用高清原片（视频最佳对齐帧或无人机原图），
框按单应矩阵投影到干净帧。

规则：
  - 现有 142 条原样保留，划分不变；
  - 新样本排除评测集（eval_set 非空）与“看不清”；与旧样本同事件且投影框 IoU ≥ 0.5 的视为重复并丢弃；
  - 按飞行任务划分：已有任务/事件沿用原划分；新任务中含误检的按误检数从少到多放入验证集，
    直到验证集误检 ≥ 20%；不含误检的新任务按哈希约 25% 放入验证集；
  - 只纳入 --classes 指定的类别（默认“车”）：训练脚本按“类别|标签”等权抽样，
    只有单一标签或极少样本的类别（如只有误检的卡车）会被过度重复、学成整类驳回，另存 other_class_*.jsonl；
  - 误检全部保留，真目标每个划分按 pos_per_neg × 误检数抽样（同事件优先），其余写入备用池。
    训练脚本每轮抽样数等于训练行数且各“类别|标签”等权，真目标放太多会拉长每轮并让小类（如人员）被过度重复。
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image

LABEL_TO_GOLD = {"真目标": "true", "误检": "false"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8")


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if not target.exists():
            try:
                target.hardlink_to(item)
            except OSError:
                shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True, help="labeling-package 目录")
    parser.add_argument("--annotated-dir", type=Path, required=True, help="含 codex/work/annotated 与 original/shotimg")
    parser.add_argument("--matches-dir", type=Path, required=True, help="09-24 快照 video/still_matches 所在目录")
    parser.add_argument("--event-task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pos-per-neg", type=int, default=3)
    parser.add_argument("--classes", default="车", help="逗号分隔，纳入训练的类别")
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"输出目录非空：{args.out}")

    old_pairs = read_jsonl(args.old_root / "candidate_pairs.jsonl")
    old_gold = read_jsonl(args.old_root / "gold_labels.jsonl")
    event_task = json.loads(args.event_task.read_text(encoding="utf-8"))
    with (args.package / "annotations.csv").open(encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    results = {row["candidate_id"]: row for row in read_jsonl(args.package / "results.jsonl")}
    targets = {row["candidate_id"]: row for row in read_jsonl(args.package / "targets.jsonl")}
    match_status = {}
    for name in ("still_matches.jsonl", "video_matches.jsonl"):
        for row in read_jsonl(args.matches_dir / name):
            if row.get("clean_usable_for_inference") and row["event_id"] not in match_status:
                match_status[row["event_id"]] = row["status"]

    old_gold_by_id = {row["candidate_id"]: row for row in old_gold}
    old_by_event = collections.defaultdict(list)
    task_partition, event_partition = {}, {}
    for pair in old_pairs:
        old_by_event[pair["event_id"]].append(pair)
        task_partition[pair["source_task_id"]] = pair["partition"]
        event_partition[pair["event_id"]] = pair["partition"]

    audit = collections.Counter()
    rows = []
    for label in labels:
        if label["eval_set"]:
            audit["excluded_eval_set"] += 1
            continue
        if label["label"] not in LABEL_TO_GOLD:
            audit["excluded_unclear"] += 1
            continue
        in_scope = label["class_name"] in set(args.classes.split(","))
        result = results[label["candidate_id"]]
        width, height = result["source_size"]
        xs = [p[0] for p in result["projected_polygon"]]
        ys = [p[1] for p in result["projected_polygon"]]
        clean_bbox = [min(xs), min(ys), max(xs), max(ys)]
        norm = [clean_bbox[0] / width, clean_bbox[1] / height, clean_bbox[2] / width, clean_bbox[3] / height]
        duplicate = False
        for old in old_by_event.get(label["event_id"], []):
            if not old.get("projected_clean_bbox_px"):
                continue
            cw, ch = old["clean_frame_size"]
            b = old["projected_clean_bbox_px"]
            if iou(norm, [b[0] / cw, b[1] / ch, b[2] / cw, b[3] / ch]) >= 0.5:
                duplicate = True
                break
        if duplicate:
            audit["excluded_duplicate_of_existing"] += 1
            continue
        rows.append({"label": label, "result": result, "target": targets[label["candidate_id"]],
                     "task": event_task[label["event_id"]], "clean_bbox": clean_bbox, "in_scope": in_scope})

    other = [r for r in rows if not r["in_scope"]]
    rows = [r for r in rows if r["in_scope"]]
    audit["excluded_other_class"] = len(other)

    # 按任务划分
    fp_by_task = collections.Counter(r["task"] for r in rows if r["label"]["label"] == "误检")
    new_tasks = sorted({r["task"] for r in rows} - set(task_partition))
    for row in rows:
        known = task_partition.get(row["task"]) or event_partition.get(row["label"]["event_id"])
        if known and event_partition.get(row["label"]["event_id"], known) != known:
            raise SystemExit(f"任务与事件的既有划分冲突：{row['label']['event_id']}")
    total_fp = sum(fp_by_task.values())
    val_fp = sum(fp_by_task[t] for t, p in task_partition.items() if p == "validation")
    for task in sorted((t for t in new_tasks if fp_by_task[t]), key=lambda t: (fp_by_task[t], t)):
        if val_fp < 0.2 * total_fp:
            task_partition[task] = "validation"
            val_fp += fp_by_task[task]
        else:
            task_partition[task] = "train"
    for task in new_tasks:
        if task not in task_partition:
            bucket = int(hashlib.sha256(f"{args.seed}:{task}".encode()).hexdigest(), 16) % 4
            task_partition[task] = "validation" if bucket == 0 else "train"
    for row in rows + other:
        row["partition"] = event_partition.get(row["label"]["event_id"]) or task_partition.get(row["task"], "train")

    # 正负样本比例
    rng = random.Random(args.seed)
    selected, reserve = [], []
    for partition in ("train", "validation"):
        part = [r for r in rows if r["partition"] == partition]
        negatives = [r for r in part if r["label"]["label"] == "误检"]
        positives = [r for r in part if r["label"]["label"] == "真目标"]
        fp_events = {r["label"]["event_id"] for r in negatives}
        rng.shuffle(positives)
        positives.sort(key=lambda r: r["label"]["event_id"] not in fp_events)
        quota = args.pos_per_neg * len(negatives)
        selected += negatives + positives[:quota]
        reserve += positives[quota:]

    args.out.mkdir(parents=True, exist_ok=True)
    for sub in ("event_annotated_frames", "event_clean_frames"):
        link_tree(args.old_root / sub, args.out / sub)
    (args.out / "released_annotated_frames").mkdir(exist_ok=True)
    (args.out / "released_clean_frames").mkdir(exist_ok=True)

    def to_pair(row: dict, candidate_id: str) -> dict:
        label, result, target = row["label"], row["result"], row["target"]
        event = label["event_id"]
        source = "codex/work/annotated" if target["source"] == "snapshot" else "original/shotimg"
        annotated_rel = f"released_annotated_frames/{event}.jpg"
        clean_rel = f"released_clean_frames/{event}.jpg"
        if not (args.out / annotated_rel).exists():
            shutil.copy2(args.annotated_dir / source / f"{event}.jpg", args.out / annotated_rel)
        if not (args.out / clean_rel).exists():
            shutil.copy2(args.package / "frames" / f"{event}.jpg", args.out / clean_rel)
        with Image.open(args.out / annotated_rel) as image:
            annotated_size = list(image.size)
        with Image.open(args.out / clean_rel) as image:
            if list(image.size) != list(result["source_size"]):
                raise SystemExit(f"高清帧尺寸不一致：{event}")
        return {
            "partition": row["partition"], "candidate_id": candidate_id, "event_id": event,
            "source_task_id": row["task"], "class_name": label["class_name"],
            "human_label": LABEL_TO_GOLD[label["label"]], "bbox_px": target["bbox_px"],
            "annotated_image_size": annotated_size, "annotated_frame": annotated_rel,
            "has_clean_source": True,
            "clean_match_status": match_status.get(event, "strong") if target["source"] == "snapshot" else "strong",
            "clean_source_kind": "hires_video_frame" if result["source_kind"] == "video" else
            ("hires_still_matched" if target["source"] == "snapshot" else "original_primary"),
            "clean_frame": clean_rel, "clean_frame_size": list(result["source_size"]),
            "projected_clean_polygon_px": result["projected_polygon"],
            "projected_clean_bbox_px": row["clean_bbox"],
            "origin_candidate_id": label["candidate_id"], "target_id": label["target_id"],
            "recheck_version": label["recheck_version"], "recheck_time": label["recheck_time"],
            "label_source": "wxl_hires_web_20260930", "roi_ncc": label["roi_ncc"] or None,
        }

    selected.sort(key=lambda r: (r["partition"], r["label"]["event_id"], r["label"]["candidate_id"]))
    new_pairs = [to_pair(row, f"R{index:03d}") for index, row in enumerate(selected, start=1)]
    reserve.sort(key=lambda r: (r["partition"], r["label"]["event_id"], r["label"]["candidate_id"]))
    reserve_pairs = [to_pair(row, f"RS{index:03d}") for index, row in enumerate(reserve, start=1)]

    def gold_of(pair: dict) -> dict:
        return {k: pair[k] for k in ("candidate_id", "human_label", "class_name", "partition", "event_id")}

    write_jsonl(args.out / "candidate_pairs.jsonl", old_pairs + new_pairs)
    write_jsonl(args.out / "gold_labels.jsonl", old_gold + [gold_of(p) for p in new_pairs])
    write_jsonl(args.out / "reserve_candidate_pairs.jsonl", reserve_pairs)
    write_jsonl(args.out / "reserve_gold_labels.jsonl", [gold_of(p) for p in reserve_pairs])
    other.sort(key=lambda r: (r["label"]["class_name"], r["label"]["event_id"], r["label"]["candidate_id"]))
    other_pairs = [to_pair(row, f"RO{index:03d}") for index, row in enumerate(other, start=1)]
    write_jsonl(args.out / "other_class_candidate_pairs.jsonl", other_pairs)
    write_jsonl(args.out / "other_class_gold_labels.jsonl", [gold_of(p) for p in other_pairs])
    summary = {
        "audit": dict(audit),
        "new_usable_after_dedupe": len(rows),
        "new_selected": dict(collections.Counter((p["partition"], p["human_label"]) for p in new_pairs)),
        "reserve": dict(collections.Counter((p["partition"], p["human_label"]) for p in reserve_pairs)),
        "old": dict(collections.Counter((p["partition"], old_gold_by_id[p["candidate_id"]]["human_label"]) for p in old_pairs)),
        "new_task_partitions": {t: task_partition[t] for t in new_tasks},
        "new_selected_by_class": dict(collections.Counter((p["class_name"], p["human_label"]) for p in new_pairs)),
        "other_class": dict(collections.Counter((p["class_name"], p["human_label"]) for p in other_pairs)),
        "classes": args.classes, "pos_per_neg": args.pos_per_neg, "seed": args.seed,
    }
    summary = {k: ({"/".join(kk) if isinstance(kk, tuple) else kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v)
               for k, v in summary.items()}
    (args.out / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
