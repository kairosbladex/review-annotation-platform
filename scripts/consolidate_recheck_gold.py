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
import random
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

LABELS = {"true": "true", "false": "false", "uncertain": "uncertain",
          "真目标": "true", "真车": "true", "误检": "false", "非车": "false",
          "屋顶设施": "false", "看不清": "uncertain"}
SEED = 20260930
TZ = ZoneInfo("Asia/Shanghai")

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
    before = path.stat()
    content = path.read_bytes()  # Parse exactly the bytes whose hash is recorded.
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise ValueError(f"读取期间源文件发生变化，请重跑：{path}")
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
         "source_line": n, "source_batch": str(path.parent.name),
         "source_mtime": datetime.fromtimestamp(before.st_mtime, TZ).isoformat(), "raw": r} for n, r in rows]


def web_package(path):
    return path.endswith("annotations.csv") and (
        "recheck-released-fp-20260930/labeling-package/" in path or
        "vehicle-surface-gate-eval-20260929/random-traffic-hires-package/" in path or
        "vehicle-surface-gate-eval-20260929/prerelease-evidence/" in path)


def population(path):
    if "recheck-released-fp" in path:
        return "released_by_recheck"
    if "random-traffic" in path or "resolution-ab-random" in path:
        return "random_traffic"
    if "vehicle-surface" in path:
        return "rooftop_hard_case"
    return "historical_review"


def exposure_keys(raw):
    keys = set()
    for d in (raw, raw.get("meta") or {}, raw.get("representative") or {},
              raw.get("source") if isinstance(raw.get("source"), dict) else {}):
        for k in ("id", "candidate_id", "canonical_id", "image_id", "sample_id", "target_id",
                  "image_sha256", "source_image_sha256", "original_image_sha256", "clean_sha256",
                  "review_sha256", "clean_frame_sha256", "annotated_frame_sha256", "image_object_key", "file_path"):
            if d.get(k):
                value = str(d[k]); keys.add(value)
                match = re.fullmatch(r"(.+):r?(\d+)", value)
                if match:
                    keys.add(f"{match[1]}:{int(match[2])}")
        for k in ("event_id", "source_event_id", "task_id", "source_task_id", "source_capture_group"):
            if d.get(k):
                keys.add("group:" + str(d[k]))
    return keys


def direct_exposure(path, raw):
    gpt = (raw.get("codex_prompt_version") or "ktd-codex-" in path or "prompt-eval" in path or
           "review-annotation-platform-prompt-gold" in path or "ktd-formal-review" in path or
           ("frozen/rows.jsonl" in path and raw.get("group") in ("B", "C")) or
           "regression" in path or "recheck-runs" in path)
    qwen = ("qwen" in path or "recheck-released-fp-20260930/dataset-source" in path or
            "resolution-ab" in path or "ktd-qwen-validation" in path)
    return (["gpt_prompt_exposed"] if gpt else []) + (["qwen_exposed"] if qwen else [])


def enrich_web(records, root, receipts):
    manifest_cache = {}
    for record in records:
        if not web_package(record["source_file"]):
            continue
        r = record["raw"]
        if not str(r.get("label") or "").strip():
            continue
        package = root / Path(record["source_file"]).parent
        # Geometric identity comes from the exact package's manifests, never labels.
        matches = []
        for name in ("results.jsonl", "targets.jsonl", "sample.jsonl", "selected.jsonl"):
            path = package / name
            if not path.is_file():
                continue
            if path not in manifest_cache:
                manifest_cache[path] = snapshot(path, root)
            receipt, items = manifest_cache[path]
            receipts[receipt["path"]] = {**receipt, "purpose": "web_geometry_provenance"}
            target = exposure_keys(r) - {"group:" + str(r.get("event_id"))}
            for item in items:
                if target & (exposure_keys(item["raw"]) - {"group:" + str(item["raw"].get("event_id"))}):
                    matches.append(item)
        geometry = {}
        for item in matches:
            geometry.update(item["raw"])
        result = next((i["raw"] for i in matches if i["source_file"].endswith("results.jsonl")), None)
        if result and result.get("projected_polygon") and result.get("source_size"):
            points = result["projected_polygon"]; w, h = result["source_size"]
            geometry["bbox"] = [min(x[0] for x in points)/w, min(x[1] for x in points)/h,
                                max(x[0] for x in points)/w, max(x[1] for x in points)/h]
            geometry["source_image"] = str(package / result["frame"])
            # The projected bbox belongs to this high-resolution frame.
            geometry.pop("original_image_sha256", None)
        geometry.pop("label", None); geometry.pop("human_label", None)
        merged = {**geometry, **r}
        if not merged.get("class_name") and "vehicle-surface-gate-eval" in record["source_file"]:
            merged["class_name"] = "车"
        if result:
            merged["bbox"] = geometry["bbox"]
            merged["source_image"] = geometry["source_image"]
            for key in ("image_sha256", "original_image_sha256", "source_image_sha256"):
                merged.pop(key, None)
        inferred = []
        if not merged.get("labeled_by"):
            merged["labeled_by"] = "wangxiaolin"; inferred.append("labeled_by")
        if not merged.get("labeled_at"):
            merged["labeled_at"] = record["source_mtime"]
            inferred.append("labeled_at")
        merged["metadata_inferred"] = "file_level" if inferred else None
        merged["inferred_fields"] = inferred
        record["normalized_input"] = merged
        record["geometry_refs"] = [i["ref"] for i in matches]


def normalize(record, root: Path, evidence):
    r = record.get("normalized_input", record["raw"])
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
        if original_label is not None and str(original_label).strip():
            raise ValueError(f"未知标签写法：{original_label!r}，来源 {record['ref']}")
        reasons.append("missing_label")
    timestamp = r.get("labeled_at", r.get("reviewed_at"))
    if not timestamp:
        reasons.append("missing_labeled_at")
    if reasons:
        return None, reasons
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        # Legacy platform exports lost tzinfo in SQLite; annotation_service uses utc_now().
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamp = parsed.astimezone(TZ).isoformat()
    except ValueError:
        return None, ["invalid_labeled_at"]
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
    exposure = direct_exposure(record["source_file"], r)
    for key in exposure_keys(r) | {digest}:
        exposure.extend(evidence.get(key, []))
    known = web_package(record["source_file"]) or bool(exposure)
    unknown = not known
    return {"ref": record["ref"], "class_name": class_name, "label": label, "bbox": box,
            "image_sha256": digest, "source_image": image_path,
            "image_hash_verified_locally": image_path is not None, "group_aliases": sorted(set(aliases)),
            "exposure": sorted(set(exposure)), "gpt_history_unknown": unknown,
            "population": population(record["source_file"]),
            "provenance": {**{k: v for k, v in record.items() if k not in ("raw", "normalized_input")},
                           "original_row": "source_rows.jsonl#" + record["ref"],
                           "original_label": original_label, "labeled_by": actor,
                           "labeled_at": timestamp, "note": note,
                           "candidate_id": r.get("candidate_id", r.get("canonical_id")),
                           "image_hash_verified_locally": image_path is not None,
                           "metadata_inferred": r.get("metadata_inferred"),
                           "inferred_fields": r.get("inferred_fields", [])}}, []


def consolidate(records, root: Path, evidence, old_registry=()):
    accepted, excluded = [], []
    for record in records:
        item, reasons = normalize(record, root, evidence)
        if item is None:
            excluded.append({**{k: v for k, v in record.items() if k not in ("raw", "normalized_input")},
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
    assignments, registry = {}, []
    eligible = []
    for key, members in sorted(group_members.items()):
        existing = {prior[x]["group_id"] for x in members if x in prior}
        if len(existing) > 1:
            raise ValueError("新增来源连接了两个已冻结分组，须人工裁决；不会重切")
        group_id = next(iter(existing)) if existing else "group-" + sha(dumps(sorted(members)).encode())[:20]
        items = [x for x in accepted if groups.find(x["group_aliases"][0]) == key]
        exposed = sorted({e for x in items for e in x["exposure"]})
        unknown = any(x["gpt_history_unknown"] for x in items)
        reason = "gpt_prompt_exposed" if "gpt_prompt_exposed" in exposed else (
            "gpt_history_unknown" if unknown else "seeded_group_random_half")
        assignments[key] = {"group_id": group_id, "split": "development", "exposure": exposed,
                            "gpt_history_unknown": unknown, "reason": reason, "aliases": sorted(members)}
        if reason == "seeded_group_random_half":
            eligible.append(key)
    random.Random(SEED).shuffle(eligible)
    for key in eligible[:len(eligible)//2]:
        assignments[key]["split"] = "locked_test_gpt"
    for row in assignments.values():
        row["valid_for"] = ["gpt"] if "qwen_exposed" in row["exposure"] else ["gpt", "qwen"]
        if row["split"] == "development":
            row["valid_for"] = []
            registry.append({**row, "status": "development_only"})
        row["status"] = "proposed_pending_user_confirmation"
    gold, conflicts, dedup = [], [], []
    for items in components.values():
        items = sorted(items, key=lambda x: x["ref"])
        base = items[0]
        row = {k: base[k] for k in ("class_name", "image_sha256", "bbox")}
        assignment = assignments[groups.find(base["group_aliases"][0])]
        group_id, split = assignment["group_id"], assignment["split"]
        row.update(gold_id=sha(dumps(row).encode()), group_id=group_id, split=split,
                   group_aliases=sorted({a for x in items for a in x["group_aliases"]}),
                   exposure=sorted({e for x in items for e in x["exposure"]}),
                   group_exposure=assignment["exposure"], valid_for=assignment["valid_for"],
                   split_status=assignment["status"], gpt_history_unknown=assignment["gpt_history_unknown"],
                   population=sorted({x["population"] for x in items}),
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
        test = [r for r in entries if r["split"] == "locked_test_gpt"]
        test_groups = {label: len({r["group_id"] for r in test if r["label"] == label}) for label in counts}
        test_counts = {label: sum(r["label"] == label for r in test) for label in counts}
        tier = "可评测" if min(test_counts["true"], test_counts["false"]) >= 100 and min(test_groups["true"], test_groups["false"]) >= 10 else (
            "仅可开发" if min(counts["true"], counts["false"]) >= 30 else "未充分验证")
        result[cls] = {"samples": counts, "independent_groups": group_counts,
                       "all_groups": len({r["group_id"] for r in entries}),
                       "locked_test_samples": test_counts, "locked_test_groups": test_groups, "tier": tier,
                       "gap": "可评测需要锁定测试真/假各100框、各10组；仅可开发需要全量真/假各30框"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--extra-source', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.source_root.resolve(), args.out.resolve()
    if out.exists():
        raise SystemExit('输出已存在；须明确授权删除旧输出后全量重建')
    sources, records, evidence = [], [], collections.defaultdict(set)
    receipts = {}
    paths = sorted(set(discover(root) + [p.resolve() for p in args.extra_source]))
    for path in paths:
        receipt, items = snapshot(path, root)
        sources.append(receipt); records.extend(items)
    # Exposure-only files are never admitted as human gold.
    exposure_paths = set((root / 'data').glob('*qwen*/all.jsonl'))
    exposure_paths.update((root / 'data').glob('*qwen*/train.jsonl'))
    exposure_paths.update((root / 'data').glob('*qwen*/validation.jsonl'))
    exposure_paths.update((root / 'data/recheck-released-fp-20260930/dataset-source').glob('*.jsonl'))
    exposure_paths.update((root / 'data/vehicle-surface-gate-eval-20260929').glob('resolution-ab*/**/*.jsonl'))
    exposure_paths.update((root / 'data').glob('ktd-codex-*/*.jsonl'))
    exposure_paths.update((root / 'data/ktd-formal-review-20260903').glob('*jsonl*'))
    exposure_paths.update((root / 'data/recheck-runs').glob('**/*.jsonl'))
    exposure_paths.update((root / 'data').glob('**/*regression*.jsonl'))
    prompt_root = root.parent / 'review-annotation-platform-prompt-gold/data/prompt-eval'
    exposure_paths.update(prompt_root.glob('snapshots/**/*.jsonl'))
    exposure_paths.update((root / 'data/prompt-eval').glob('**/*.jsonl'))
    for path in sorted(exposure_paths):
        if 'sealed-copy' in path.parts:
            continue
        receipt, items = snapshot(path, root)
        receipts[receipt['path']] = {**receipt, 'purpose': 'model_exposure_only'}
        receipts[receipt["path"]]["exposure"] = sorted({e for item in items for e in direct_exposure(str(path), item["raw"])})
        for item in items:
            flags = direct_exposure(str(path), item['raw'])
            for key in exposure_keys(item['raw']):
                evidence[key].update(flags)
    for item in records:
        for key in exposure_keys(item['raw']):
            evidence[key].update(direct_exposure(item['source_file'], item['raw']))
    enrich_web(records, root, receipts)
    gold, excluded, conflicts, dedup, registry = consolidate(records, root, evidence)
    accepted_refs = {p['ref'] for r in gold + conflicts for p in r['provenance']}
    inferred = collections.Counter(p['source_file'] for r in gold+conflicts for p in r['provenance'] if p['metadata_inferred'])
    inferred_all = collections.Counter(r['source_file'] for r in records if r.get('normalized_input', {}).get('metadata_inferred'))
    unknown = collections.defaultdict(lambda: {'groups': set(), 'gold_ids': set()})
    locked = collections.defaultdict(lambda: {'groups': set(), 'gold_ids': set(), 'populations': collections.defaultdict(set)})
    for r in gold:
        for p in r['provenance']:
            if r['gpt_history_unknown']:
                unknown[p['source_file']]['groups'].add(r['group_id'])
                unknown[p['source_file']]['gold_ids'].add(r['gold_id'])
            if r['split'] == 'locked_test_gpt':
                v = locked[p['source_file']]; v['groups'].add(r['group_id']); v['gold_ids'].add(r['gold_id'])
                v['populations'][population(p['source_file'])].add(r['gold_id'])
    locked_sources = [{ 'source': k, 'groups': len(v['groups']), 'boxes': len(v['gold_ids']),
                        'population': {p:len(ids) for p,ids in v['populations'].items()}} for k,v in sorted(locked.items())]
    unknown_sources = [{'source':k, 'groups':len(v['groups']), 'boxes':len(v['gold_ids'])} for k,v in sorted(unknown.items())]
    for s in sources:
        s['accepted_rows'] = sum(r['ref'] in accepted_refs for r in records if r['source_file']==s['path'])
        s['excluded_rows'] = s['rows'] - s['accepted_rows']
    class_conflicts = []
    for cls in sorted({r['class_name'] for r in conflicts}):
        rows = [r for r in conflicts if r['class_name']==cls]
        class_conflicts.append({'class_name':cls,'conflict_samples':len(rows),
                                'label_combinations':dict(collections.Counter('/'.join(r['labels']) for r in rows)),
                                'gold_ids':[r['gold_id'] for r in rows]})
    # Population tables are separate; no pooled accuracy is calculated.
    populations = {pop: statistics([r for r in gold if pop in r['population']]) for pop in sorted({p for r in gold for p in r['population']})}
    report = {'schema':'recheck_gold_consolidated_v2', 'captured_at':datetime.now(TZ).isoformat(),
              'timezone':'Asia/Shanghai','random_seed':SEED,'source_root':str(root),
              'source_rows':len(records),'eligible_source_rows':len(accepted_refs),
              'gold_samples':len(gold),'excluded_rows':len(excluded),'conflict_samples':len(conflicts),
              'removed_duplicate_rows':sum(r['removed_duplicates'] for r in dedup),
              'classes':statistics(gold),'population':populations,'locked_test_gpt_sources':locked_sources,
              'metadata_inferred_rows_by_source':dict(sorted(inferred_all.items())),
              'metadata_inferred_eligible_rows_by_source':dict(sorted(inferred.items())),
              'gpt_history_unknown_sources':unknown_sources, 'conflicts_by_class':class_conflicts,
              'split_status':'proposed_pending_user_confirmation',
              'unresolved_exposure_manifests':[{'source':'12个复检回归样本',
                 'reason':'未定位独立完整清单，无法核对跨来源重叠；测试候选不视为完整曝光审计通过'}],
              'exclusion_reasons':dict(collections.Counter(x for r in excluded for x in r['reasons'])),
              'side_effects':{'model_calls':0,'image_uploads':0,'database_writes':0,'event_submissions':0},
              'limitations':['12个复检回归样本的独立完整清单尚未定位；已把本地recheck-runs、prompt-eval快照与formal-review曝光并入证据；其余不明历史来源保守进开发集，待逐个确认。',
                             '封存答案sealed-copy未打开；仅进行金标清点，不运行模型或准确率评测。',
                             '同一金标可能有多个来源/population，来源贡献和population表不可相加；各自去重计数。',
                             '本机无原图但有图片哈希的行仅保留哈希声明，不声称图片已现场核对。',
                             '旧平台无时区标注时间依据annotation_service.utc_now()按UTC解析后转Asia/Shanghai；原始字符串仍留在接受行快照。']}
    exclusions = {r['ref']:r['reasons'] for r in excluded}
    snapshots = []
    for r in records:
        if r['ref'] in exclusions:
            snapshots.append({'ref':r['ref'],'source_sha256':r['source_sha256'],'reasons':exclusions[r['ref']]})
        else:
            snapshots.append({k:v for k,v in r.items() if k != 'normalized_input'})
    out.mkdir(parents=True)
    group_rows = {}
    for r in gold+conflicts:
        group_rows.setdefault(r['group_id'], {'group_id':r['group_id'],'split':r['split'], 'valid_for':r['valid_for'],
                             'exposure':r['group_exposure'],'gpt_history_unknown':r['gpt_history_unknown'],
                             'status':'proposed_pending_user_confirmation'})
    for name, rows in [('source_rows',snapshots),('gold',gold),('excluded',excluded),('conflicts',conflicts),
                       ('conflicts_by_class',class_conflicts),('deduplication',dedup),('sources',sources),
                       ('split_registry',registry),('proposed_groups',list(group_rows.values())),
                       ('exposure_sources',list(receipts.values())),
                       ('splits',[{k:r[k] for k in ('gold_id','group_id','split','valid_for','split_status')} for r in gold])]:
        write_rows(out/(name+'.jsonl'),rows)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines = ['# 复检统一金标：第0步修正重跑，待用户确认','',
             f"快照：{report['captured_at']}；时区Asia/Shanghai。来源{len(records)}行，合格人工{len(accepted_refs)}行，去重金标{len(gold)}框，剔除{len(excluded)}行，冲突{len(conflicts)}框。",'',
             '可评测：locked_test_gpt真、假各≥100框且各≥10独立组；仅可开发：全部真、假各≥30框；uncertain不计真/假。档位仅表示样本数量条件，不表示已通过模型评测。','',
             f'固定种子{SEED}。按排序后的组列表用Python random.Random(seed).shuffle，未被GPT看过且来源可判定的组前floor(N/2)进入locked_test_gpt，其余开发。GPT曝光或历史不明的整组进入开发。仅Qwen曝光不阻止GPT锁定测试；带Qwen曝光的测试组valid_for=["gpt"]，禁止评测Qwen。','',
             '全部测试切分仅为待确认提案。split_registry.jsonl只记录development，不写入、不冻结locked_test_gpt；提案见proposed_groups.jsonl与splits.jsonl。确认前不进入第1—4步，不调用中转站。','']
    def table(stats):
        result=['|类别|真/假/uncertain框|真/假/uncertain组|测试真/假框|测试真/假组|档位|','|---|---|---|---|---|---|']
        for cls,c in stats.items():
            fmt=lambda d,ks: '/'.join(str(d[k]) for k in ks)
            result.append(f"|{cls}|{fmt(c['samples'],('true','false','uncertain'))}|{fmt(c['independent_groups'],('true','false','uncertain'))}|{fmt(c['locked_test_samples'],('true','false'))}|{fmt(c['locked_test_groups'],('true','false'))}|{c['tier']}|")
        return result
    lines += table(report['classes'])
    lines += ['','## population分开清点（无准确率评测）','']
    for pop,stats in populations.items():
        lines += ['### '+pop,'']+table(stats)+['']
    lines += ['## locked_test_gpt来源清单','','同一金标若有多来源，分别贡献计数，不能把来源表直接相加。','','|来源|组|框|population框分布|','|---|---|---|---|']
    for v in locked_sources:
        lines.append(f"|{v['source']}|{v['groups']}|{v['boxes']}|{dumps(v['population'])}|")
    lines += ['','## 补齐元数据','','只对指定人工网页CSV的非空label补齐；labeled_by=wangxiaolin，labeled_at=该CSV的文件mtime，metadata_inferred=file_level。原行快照不篡改。','','|来源|补齐行数|其中通过溯源检查|','|---|---|---|']
    for k,v in sorted(inferred_all.items()):
        lines.append(f'|{k}|{v}|{inferred[k]}|')
    lines += ['','## GPT曝光无法判断，保守进开发集','','以下包含因同组其他来源不明而连带进入开发的来源。','','|来源|组|框|','|---|---|---|']
    for v in unknown_sources:
        lines.append(f"|{v['source']}|{v['groups']}|{v['boxes']}|")
    if not unknown_sources:
        lines += ['当前已纳入金标未发现额外的来源不明组；但12个回归样本独立清单未定位，跨来源重叠核验仍未完成，不能把测试候选视为完整曝光审计通过。']
    lines += ['','## 冲突隔离','','|类别|冲突框|标签组合|','|---|---|---|']
    for v in class_conflicts:
        lines.append(f"|{v['class_name']}|{v['conflict_samples']}|{dumps(v['label_combinations'])}|")
    lines += ['','冲突逐框溯源见conflicts.jsonl，分类裁决索引见conflicts_by_class.jsonl。','',
              '## 来源对账','','|来源|行数|合格人工行|剔除行|','|---|---|---|---|']
    for v in sources:
        lines.append(f"|{v['path']}|{v['rows']}|{v['accepted_rows']}|{v['excluded_rows']}|")
    lines += ['','## 检查范围','']+['- '+x for x in report['limitations']]
    lines += ['','source_rows.jsonl的剔除行仅保留ref、源哈希及原因，无原始内容。接受行保留原始快照；网页几何信息的补齐来源和哈希见geometry_refs及exposure_sources.jsonl。',
              '','资源清理：未启动容器、GPU、后台进程或端口；不复制图片与模型，只保留脚本、测试、最终清单和验证证据。原工作树保留供确认，本次未新增工作树。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    refresh_sums(out)
    print(json.dumps({k:report[k] for k in ('gold_samples','conflict_samples','classes','metadata_inferred_rows_by_source','locked_test_gpt_sources','gpt_history_unknown_sources')},ensure_ascii=False,indent=2))


def refresh_sums(out):
    (out/'SHA256SUMS').write_text('\n'.join(f'{sha(p.read_bytes())}  {p.name}' for p in sorted(out.iterdir()) if p.is_file() and p.name != 'SHA256SUMS')+'\n')


if __name__ == '__main__':
    main()
