#!/usr/bin/env python3
"""Local-only human gold audit. No network, database, model or image writes."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LABELS = {"true": "true", "false": "false", "uncertain": "uncertain",
          "真目标": "true", "真车": "true", "误检": "false", "非车": "false",
          "屋顶设施": "false", "看不清": "uncertain"}
HUMAN_ORIGINS = {"human", "human_gold", "manual", "human_confirmed", "wxl_web"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_rows(path: Path, rows) -> None:
    path.write_text("".join(dumps(r) + "\n" for r in rows), encoding="utf-8")


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Union:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    total = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / total if total else 0


def discover(root: Path):
    paths = []
    for p in (root / "data").rglob("*"):
        if p.suffix not in {".csv", ".jsonl"} or not p.is_file():
            continue
        if any(x.startswith("recheck-gold-consolidated-") for x in p.parts):
            continue
        if any(x in p.parts for x in ("sealed-copy", "agent-runs")):
            continue  # Do not open sealed test answers or training checkpoints.
        if re.search(r"gold|human|annotation|^labels", p.name) or p.name == "rows.jsonl" and "frozen" in p.parts:
            paths.append(p)
    return sorted(paths)


def snapshot(path: Path, root: Path):
    content = path.read_bytes()  # One read: parse exactly the bytes whose hash is recorded.
    fingerprint = sha(content)
    name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    if path.suffix == ".csv":
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        rows = []
        for row in reader:
            rows.append((reader.line_num, row))
    else:
        rows = [(n, json.loads(line)) for n, line in enumerate(content.decode("utf-8-sig").splitlines(), 1) if line.strip()]
    return {"path": name, "sha256": fingerprint, "bytes": len(content), "rows": len(rows)}, [
        {"ref": f"{name}:{n}@{fingerprint}", "source_file": name, "source_sha256": fingerprint,
         "source_line": n, "source_batch": str(path.parent.name), "raw": r} for n, r in rows]


def normalize(record, root: Path, evidence):
    r = record["raw"]
    meta = r.get("meta") or {}
    original_label = r.get("human_label", r.get("label"))
    actor = r.get("labeled_by", r.get("reviewer_id"))
    note = str(r.get("human_note", r.get("note", r.get("human_comment", ""))) or "")
    reasons = []
    if not actor or not str(actor).strip():
        reasons.append("missing_labeled_by")
    if note.strip().lower().startswith("auto annotation"):
        reasons.append("auto_annotation_note")
    for field in ("label_origin", "label_source"):
        if field in r and str(r[field]).strip().lower() not in HUMAN_ORIGINS:
            reasons.append("non_human_" + field)
    label = LABELS.get(str(original_label).strip().lower())
    if label is None:
        reasons.append("missing_or_unknown_label")
    timestamp = r.get("labeled_at", r.get("reviewed_at"))
    if not timestamp:
        reasons.append("missing_labeled_at")
    if reasons:
        return None, reasons
    class_name = r.get("class_name", r.get("candidate_class"))
    box = r.get("bbox", r.get("bbox_norm"))
    if box is None and all(r.get("bbox_norm_"+k) not in (None, "") for k in ("x1", "y1", "x2", "y2")):
        box = [r["bbox_norm_"+k] for k in ("x1", "y1", "x2", "y2")]
    try:
        box = [float(x) for x in box]
        if len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError()
    except (TypeError, ValueError):
        return None, ["missing_or_invalid_normalized_box"]
    if not class_name:
        return None, ["missing_class"]
    digest = r.get("image_sha256", r.get("source_image_sha256", r.get("original_image_sha256")))
    image_path = None
    possible = [r.get("canonical_source_image"), r.get("source_image"), r.get("original_image_path")]
    if r.get("sample_id"):
        possible.extend(str(p) for p in (root / Path(record["source_file"]).parent / "images").glob(str(r["sample_id"]) + ".*"))
    for name in possible:
        if not name:
            continue
        p = Path(name)
        candidates = [p] if p.is_absolute() else [root / Path(record["source_file"]).parent / p, root / p]
        found = next((x for x in candidates if x.is_file()), None)
        if found:
            actual = sha(found.read_bytes())
            if digest and actual != digest:
                return None, ["image_hash_mismatch"]
            digest, image_path = actual, str(found)
            break
    if not digest or not re.fullmatch(r"[a-f0-9]{64}", str(digest)):
        return None, ["missing_image_hash"]
    aliases = []
    src = r.get("source") if isinstance(r.get("source"), dict) else {}
    task = r.get("source_task_id", r.get("task_id"))
    capture = r.get("source_capture_group", r.get("capture_group"))
    event = r.get("source_event_id", r.get("event_id", src.get("source_event_id")))
    if task:
        aliases.append("task:" + str(task))
    if capture:
        aliases.append(("task:" if r.get("source_group_kind") == "task_id" else "capture:") + str(capture))
    if event:
        aliases.append("event:" + str(event))
    if not aliases:
        return None, ["missing_task_capture_and_event"]
    # Shared image content also connects groups, even when event IDs differ.
    aliases.append("image:" + digest)
    exposure = []
    path = record["source_file"]
    if "codex" in path or r.get("codex_prompt_version"):
        exposure.append("prior_prompt_review_batch")
    if "qwen" in path or "train" in path or r.get("training_eligible"):
        exposure.append("qwen_or_training_batch")
    if "vehicle-surface" in path:
        exposure.append("vehicle_surface_evaluation")
    for key in (str(r.get("candidate_id", "")), str(r.get("sample_id", "")), digest):
        exposure.extend(evidence.get(key, []))
    # Historic labels with incomplete exposure lineage cannot establish a fresh test.
    if not exposure:
        exposure.append("exposure_history_unverified")
    return {"ref": record["ref"], "class_name": class_name, "label": label, "bbox": box,
            "image_sha256": digest, "source_image": image_path,
            "image_hash_verified_locally": image_path is not None, "group_aliases": sorted(set(aliases)),
            "exposure_reasons": sorted(set(exposure)),
            "provenance": {**{k: v for k, v in record.items() if k != "raw"},
                           "original_row": "source_rows.jsonl#" + record["ref"],
                           "original_label": original_label, "labeled_by": actor,
                           "labeled_at": timestamp, "note": note,
                           "candidate_id": r.get("candidate_id", r.get("canonical_id")),
                           "image_hash_verified_locally": image_path is not None}}, []


def consolidate(records, root: Path, evidence, old_registry=()):
    accepted, excluded = [], []
    for record in records:
        item, reasons = normalize(record, root, evidence)
        if item is None:
            excluded.append({**{k: v for k, v in record.items() if k != "raw"},
                             "original_row": "source_rows.jsonl#" + record["ref"], "reasons": reasons})
        else:
            accepted.append(item)
    grouped = collections.defaultdict(list)
    duplicates = Union()
    groups = Union()
    for index, item in enumerate(accepted):
        duplicates.find(index)
        grouped[(item["image_sha256"], item["class_name"])].append(index)
        aliases = item["group_aliases"]
        for alias in aliases:
            groups.join(aliases[0], alias)
    for indices in grouped.values():
        for pos, a in enumerate(indices):
            for b in indices[pos+1:]:
                if iou(accepted[a]["bbox"], accepted[b]["bbox"]) >= 0.5:
                    duplicates.join(a, b)
    components = collections.defaultdict(list)
    for index, item in enumerate(accepted):
        components[duplicates.find(index)].append(item)
    prior = {alias: row for row in old_registry for alias in row["aliases"]}
    group_members = collections.defaultdict(set)
    for item in accepted:
        group_members[groups.find(item["group_aliases"][0])].update(item["group_aliases"])
    assignments, registry = {}, list(old_registry)
    for key, members in sorted(group_members.items()):
        existing = {prior[x]["group_id"] for x in members if x in prior}
        if len(existing) > 1:
            raise ValueError("新增来源连接了两个已冻结分组，须人工裁决；不会重切")
        group_id = next(iter(existing)) if existing else "group-" + sha(dumps(sorted(members)).encode())[:20]
        split = next((prior[x]["split"] for x in members if x in prior), "development")
        if split != "development":
            raise ValueError("历史已看样本连接到锁定测试组，须人工裁决；不会改动冻结切分")
        assignments[key] = (group_id, split)
        new_aliases = members - prior.keys()
        if new_aliases:
            registry.append({"group_id": group_id, "split": split, "aliases": sorted(new_aliases),
                             "reason": "historical_exposure_or_unverified_history"})
    gold, conflicts, dedup = [], [], []
    for items in components.values():
        items = sorted(items, key=lambda x: x["ref"])
        base = items[0]
        row = {k: base[k] for k in ("class_name", "image_sha256", "bbox")}
        group_id, split = assignments[groups.find(base["group_aliases"][0])]
        row.update(gold_id=sha(dumps(row).encode()), group_id=group_id, split=split,
                   group_aliases=sorted({a for x in items for a in x["group_aliases"]}),
                   exposure_reasons=sorted({e for x in items for e in x["exposure_reasons"]}),
                   source_images=sorted({x["source_image"] for x in items if x["source_image"]}),
                   provenance=[x["provenance"] for x in items],
                   duplicate_policy="same_image_class_iou_ge_0.5_connected_component")
        labels = sorted({x["label"] for x in items})
        if len(labels) > 1:
            row.update(labels=labels, status="requires_human_adjudication")
            conflicts.append(row)
        else:
            row["label"] = labels[0]
            gold.append(row)
        if len(items) > 1:
            dedup.append({"gold_id": row["gold_id"], "source_refs": [x["ref"] for x in items],
                          "removed_duplicates": len(items)-1, "conflict": len(labels) > 1})
    return gold, excluded, conflicts, dedup, registry


def statistics(gold):
    result = {}
    for cls in sorted({r["class_name"] for r in gold}):
        entries = [r for r in gold if r["class_name"] == cls]
        counts = {label: sum(r["label"] == label for r in entries) for label in ("true", "false", "uncertain")}
        group_counts = {label: len({r["group_id"] for r in entries if r["label"] == label}) for label in counts}
        test = [r for r in entries if r["split"] == "locked_test"]
        test_groups = {label: len({r["group_id"] for r in test if r["label"] == label}) for label in counts}
        # User's thresholds are group counts, not box counts.
        tier = "可评测" if min(test_groups["true"], test_groups["false"]) >= 100 else (
            "仅可开发" if min(group_counts["true"], group_counts["false"]) >= 30 else "未充分验证")
        result[cls] = {"samples": counts, "independent_groups": group_counts,
                       "all_groups": len({r["group_id"] for r in entries}),
                       "locked_test_groups": test_groups, "tier": tier,
                       "gap": "缺少未被使用过的锁定测试任务组；真、假各需100组"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--extra-source", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--append", action="store_true", help="保留历史来源快照与分组，仅追加；不重切")
    args = parser.parse_args()
    root, out = args.source_root.resolve(), args.out.resolve()
    if out.exists() and not args.append:
        raise SystemExit("输出已存在；禁止重切。追加使用 --append")
    sources, records, evidence = [], [], collections.defaultdict(list)
    exposure_sources = []
    # Converted training conversations mark exposure; they never supply gold labels.
    for path in sorted((root / "data").glob("*qwen*/all.jsonl")):
        receipt, items = snapshot(path, root)
        exposure_sources.append({**receipt, "purpose": "training_exposure_only"})
        for item in items:
            raw = item["raw"]
            for d in (raw, raw.get("meta") or {}):
                for key in ("id", "candidate_id", "image_id", "sample_id", "image_sha256"):
                    if d.get(key):
                        evidence[str(d[key])].append("qwen_training_dataset")
    for path in sorted(set(discover(root) + [p.resolve() for p in args.extra_source])):
        receipt, items = snapshot(path, root)
        sources.append(receipt)
        records.extend(items)
        if "qwen" in str(path) or "train" in str(path) or "frozen" in path.parts or "prompt-eval" in str(path):
            reason = "vehicle_surface_B_C" if "frozen" in path.parts else "prior_training_or_prompt_evaluation"
            for item in items:
                r = item["raw"]
                for d in (r, r.get("meta") or {}):
                    for key in ("id", "candidate_id", "sample_id", "image_sha256", "clean_sha256", "review_sha256"):
                        if d.get(key):
                            evidence[str(d[key])].append(reason)
    old_registry = []
    if args.append:
        old_registry = read_rows(out / "split_registry.jsonl")
        records = read_rows(out / "source_rows.jsonl") + records
    records = list({r["ref"]: r for r in records}.values())
    gold, excluded, conflicts, dedup, registry = consolidate(records, root, evidence, old_registry)
    sources_by_file = {s["path"]: {**s, "accepted_rows": 0, "excluded_rows": 0,
                                  "reason": "空文件；无人工标签可纳入"} for s in sources}
    for source, items in itertools.groupby(sorted(records, key=lambda r: r["source_file"]), key=lambda r: r["source_file"]):
        items = list(items)
        accepted_refs = {p["ref"] for r in gold + conflicts for p in r["provenance"]}
        sources_by_file.setdefault(source, {"path": source, "rows": len(items), "sha256": "historic_snapshot_preserved", "bytes": None})
        sources_by_file[source]["accepted_rows"] = sum(i["ref"] in accepted_refs for i in items)
        sources_by_file[source]["excluded_rows"] = len(items)-sources_by_file[source]["accepted_rows"]
        sources_by_file[source]["reason"] = "通过人工来源检查；仍须去重/冲突隔离" if sources_by_file[source]["accepted_rows"] else "缺少人工身份/时间/原始标签或必要溯源，逐行原因见剔除清单；空文件保持空"
    subsets = []
    for small, large in [("data/gold-export-20260903/annotations.jsonl", "data/gold-export-live-20260903/annotations.jsonl"),
                         ("data/ktd-codex-human-gold-validation-trial-20260924/human_gold_annotations.jsonl", "data/ktd-codex-human-gold-validation-20260924/human_gold_annotations.jsonl")]:
        def signatures(file):
            return {(r["raw"].get("candidate_id"), r["raw"].get("human_label"), tuple(r["raw"].get("bbox") or []), r["raw"].get("class_name")) for r in records if r["source_file"] == file}
        a, b = signatures(small), signatures(large)
        subsets.append({"small": small, "large": large, "small_unique": len(a), "large_unique": len(b), "is_label_box_subset": a <= b, "not_in_large": len(a-b)})
    deferred = [{"path": str(p.relative_to(root)), "reason": "封存答案，不打开；不纳入本期金标"}
                for p in (root / "data").glob("**/sealed-copy/*human*jsonl")]
    report = {"schema": "recheck_gold_consolidated_v1", "captured_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
              "source_root": str(root), "source_rows": len(records), "eligible_source_rows": sum(len(r["provenance"]) for r in gold+conflicts),
              "gold_samples": len(gold), "excluded_rows": len(excluded), "conflict_samples": len(conflicts),
              "removed_duplicate_rows": sum(r["removed_duplicates"] for r in dedup),
              "exclusion_reasons": dict(collections.Counter(reason for r in excluded for reason in r["reasons"])),
              "classes": statistics(gold), "subset_checks": subsets, "deferred_sources": deferred,
              "formal_evaluation_available": any(c["tier"] == "可评测" for c in statistics(gold).values()),
              "side_effects": {"model_calls": 0, "image_uploads": 0, "database_writes": 0, "event_submissions": 0},
              "limitations": ["12个回归样本的唯一完整清单未在当前入口定位；所有历史来源保守限制开发集，不会漏入测试集。",
                              "封存测试答案 sealed-copy 未打开；本次不使用旧锁定测试集做评分。",
                              "有图片哈希但本机无原图的行保留原始声明，不能声称已现场校验图片；source_images为空。",
                              "任务/连拍/事件别名和相同图片连通分组，避免跨来源泄漏；连通组计数比任务数更保守。",
                              "外发授权、模型别名实测及额度接口尚未验证；本步无需调用。"]}
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("source_rows", records), ("gold", gold), ("excluded", excluded), ("conflicts", conflicts),
                       ("deduplication", dedup), ("split_registry", registry), ("sources", list(sources_by_file.values())),
                       ("exposure_sources", exposure_sources),
                       ("splits", [{"gold_id": r["gold_id"], "group_id": r["group_id"], "split": r["split"]} for r in gold])]:
        write_rows(out / (name + ".jsonl"), rows)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    lines = ["# 复检统一金标：第0步待用户确认", "", f"快照时间：{report['captured_at']}。只读取本地文件，图片不复制、不上传。", "",
             f"来源行 {len(records)}；合格人工来源行 {report['eligible_source_rows']}；去重后金标 {len(gold)}；剔除 {len(excluded)}；冲突目标 {len(conflicts)}。", "",
             "暂无类别可正式评测。" if not report["formal_evaluation_available"] else "存在可评测类别。", "",
             "档位全部按独立组计数：锁定测试真、假各≥100组才可评测（因此各≥10组也满足）；开发真、假各≥30组才仅可开发。框数只作附加统计。", "",
             "|类别|真/假/存疑框数|真/假/存疑独立组数|总组数|锁定测试真/假组|档位|", "|---|---|---|---|---|---|"]
    for cls, c in report["classes"].items():
        fmt = lambda d: "/".join(str(d[k]) for k in ("true", "false", "uncertain"))
        lines.append(f"|{cls}|{fmt(c['samples'])}|{fmt(c['independent_groups'])}|{c['all_groups']}|{c['locked_test_groups']['true']}/{c['locked_test_groups']['false']}|{c['tier']}|")
    lines += ["", "每类主要缺口：尚无未被训练/调提示词使用的锁定测试任务组；不足30组的类别还缺开发真/假组。", "",
              "## 来源纳入和排除", "", "每行来源引用包含文件、物理行号及本次文件哈希。CSV读取一次后立即固化，后续标注变化不改变本快照。", "",
              "|来源|快照行数|合格人工行|剔除行|原因|", "|---|---|---|---|---|"]
    for s in sources_by_file.values():
        lines.append(f"|{s['path']}|{s['rows']}|{s['accepted_rows']}|{s['excluded_rows']}|{s['reason']}|")
    lines += ["", "## 子集核对", ""]
    for s in subsets:
        lines.append(f"- `{s['small']}` → `{s['large']}`：标签、同类框及候选ID子集={s['is_label_box_subset']}；{s['small_unique']}/{s['large_unique']}，不在大批内={s['not_in_large']}。")
    lines += ["", "## 口径与冻结", "", "人工身份为空、auto annotation备注、明确非人工来源都剔除；未记录标注时间、图片哈希、框或分组的行也单列剔除，绝不补造元数据。reviewer_id是旧平台人工身份字段，保留原字段并归一到labeled_by。",
              "", "按同图SHA-256、同类别、框IoU≥0.5建立连通组件去重。组件内任何标签冲突（含uncertain）全部隔离，保留所有人工原行，不自动选择。",
              "", "图片、任务、连拍和事件别名连通后固定分组；没有任务ID时退到连拍/事件。跨任务相同图片合为一个保守组。split_registry.jsonl保持追加，已冻结分组不得改动。所有本次历史样本只进入development；已看证据与无法证明未看分别记录。",
              "", "本步仅建立历史开发集。未来新组进入测试的比例/抽样规则须在用户确认第0步后、接入新金标前固定；当前脚本对历史来源没有随机分配测试的路径。",
              "", "## 文件", "", "gold.jsonl：统一金标及每次人工标注溯源；excluded.jsonl：逐行剔除原因；conflicts.jsonl：待人工裁决；deduplication.jsonl：去重关系；splits.jsonl/split_registry.jsonl：冻结清单；sources.jsonl/source_rows.jsonl：文件快照哈希与原行；exposure_sources.jsonl：仅标记训练暴露的转换文件；SHA256SUMS：产物哈希。", "",
              "## 检查范围与待确认", ""]
    lines += ["- " + x for x in report["limitations"]]
    lines += ["- 排除且未打开：" + x["path"] + "；" + x["reason"] for x in deferred]
    lines += ["", "默认素材建议：画面匹配的高清干净原图；无干净图时标明红框截图局限。依据源目录vehicle-surface-gate-eval-20260929/resolution-ab/README.md和resolution-ab-random/README.md。难例配对改善有依据，随机批收益未达到统计显著，不推广为GPT准确率结论。"]
    lines += ["", "线上后续记录使用标注平台DATABASE_URL主库新表；本步不连库、不做迁移、不启服务。模型别名和/v1/usage未验证。", "",
              "资源：无容器、GPU、后台进程、端口监听；不复制图片与模型。只保留脚本、测试、统一清单和必要证据。", "",
              "统计报告待用户确认。确认前不进入第1—4步，不调用中转站。"]
    (out / "README.md").write_text("\n".join(lines)+"\n")
    sums = [f"{sha(p.read_bytes())}  {p.name}" for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("\n".join(sums)+"\n")
    print(json.dumps({k: report[k] for k in ("source_rows", "eligible_source_rows", "gold_samples", "excluded_rows", "conflict_samples", "classes")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
