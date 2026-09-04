"""Paired prompt scorer (v1.0.0 vs v1.1.0) — contract section 13.

Read-only, deterministic paired scoring of two final decisions JSONL files
against the fixed selection gold (snapshot pg1-bc8c818184d8). Both sides are
aligned one-to-one by canonical_id; any duplicate/missing/extra id, version or
SHA mismatch, or a non-terminal run is an explicit failure (exit 2), never a
silently dropped row or order-based alignment.

Usage:
    python3 -m tools.prompt_eval_data.paired_scorer score \
        --v10 final_decisions_v1.0.0.jsonl --v11 final_decisions_v1.1.0.jsonl \
        --report-v10 report.json --report-v11 report.json \
        --snapshot-dir data/prompt-eval/snapshots/pg1-bc8c818184d8 \
        --report-json out.json --report-md out.md --error-packet out.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID
from .common import jdumps
from .scorer import SCOPE_NOTE, _empty_buckets, _rate, judge, normalize_decision

PAIRED_SCORE_SCHEMA_VERSION = "prompt-gold-paired-score/1"
ERROR_PACKET_SCHEMA_VERSION = "prompt-gold-paired-error-packet/1"

SNAPSHOT_ID = "pg1-bc8c818184d8"
EXPECTED_MANIFEST_SHA256 = "4c15c3892c69eea10f28c3cd1fbc381ed6b34c7690922a34af189d5c2cd752b2"

# Codex prompt_eval run statuses that mean "no more work will happen"
# (backend/app/prompt_eval/service.py only ever sets running/succeeded/partial_failed).
TERMINAL_RUN_STATUSES = {"succeeded", "partial_failed"}

# Codex final_decisions rows use "succeeded"; section 8 uses "ok".
_STATUS_ALIAS = {"succeeded": "ok"}

VERSION_TAGS = {"v10": ".v1.0.0", "v11": ".v1.1.0"}

# Bucket severity for improved/regressed: lower is better. dangerous_miss is the
# worst outcome, false_accept next, then execution failures, then uncertain.
_BUCKET_RANK = {
    "correct_accept": 0,
    "correct_reject": 0,
    "uncertain_on_true": 1,
    "uncertain_on_false": 1,
    "protocol_failure": 2,
    "technical_failure": 2,
    "false_accept": 3,
    "dangerous_miss": 4,
}
_GOOD_BUCKETS = {"correct_accept", "correct_reject"}

CALLS_NOTE = (
    "876 次调用 = 2 个版本 × 438 个独立框；所有准确率分母只用单侧 438 框的 "
    "gold_true/gold_false，绝不用调用数 876。"
)


class PairedScorerError(Exception):
    """Explicit, non-silent failure of a paired-scoring invariant."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_strict(path: str | Path, what: str) -> list[dict]:
    """Paired mode does not tolerate bad lines: any parse error is fatal."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PairedScorerError(f"{what}: line {lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise PairedScorerError(f"{what}: line {lineno} is not a JSON object")
            rows.append(row)
    return rows


def verify_snapshot(snapshot_dir: str | Path, expected_manifest_sha: str | None = EXPECTED_MANIFEST_SHA256) -> dict:
    """Pin the gold: manifest SHA, snapshot_id and member file SHAs must match."""
    snap = Path(snapshot_dir)
    manifest_path = snap / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise PairedScorerError(f"snapshot manifest not found: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha is not None and manifest_sha != expected_manifest_sha:
        raise PairedScorerError(
            f"snapshot manifest SHA mismatch: got {manifest_sha}, expected {expected_manifest_sha}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_id = manifest.get("snapshot_id")
    if snapshot_id != snap.name:
        raise PairedScorerError(f"snapshot_id mismatch: manifest says {snapshot_id}, directory is {snap.name}")
    member_sha = {entry["path"]: entry["sha256"] for entry in manifest.get("files", [])}
    verified = {}
    for name in ("selection_gold.jsonl", "selection_inputs.jsonl"):
        path = snap / name
        if not path.is_file():
            raise PairedScorerError(f"snapshot member missing: {path}")
        actual = sha256_file(path)
        if member_sha.get(name) != actual:
            raise PairedScorerError(f"{name} SHA mismatch: got {actual}, manifest says {member_sha.get(name)}")
        verified[name] = actual
    return {
        "snapshot_id": snapshot_id,
        "manifest_sha256": manifest_sha,
        "gold_path": str(snap / "selection_gold.jsonl"),
        "gold_sha256": verified["selection_gold.jsonl"],
        "inputs_path": str(snap / "selection_inputs.jsonl"),
        "inputs_sha256": verified["selection_inputs.jsonl"],
    }


def verify_run_terminal(report_path: str | Path, expected_snapshot_sha: str | None) -> dict:
    """A non-terminal run is an explicit failure. Report SHA linkage is checked when present."""
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedScorerError(f"run report unreadable: {report_path}: {exc}") from exc
    status = report.get("status")
    if status not in TERMINAL_RUN_STATUSES:
        raise PairedScorerError(
            f"run is not terminal: {report_path} status={status!r} (terminal: {sorted(TERMINAL_RUN_STATUSES)})"
        )
    run_id = report.get("run_id")
    if not run_id:
        raise PairedScorerError(f"run report has no run_id: {report_path}")
    lineage = report.get("lineage") or {}
    snapshot_sha = lineage.get("snapshot_sha") or report.get("snapshot_sha")
    if expected_snapshot_sha and snapshot_sha and snapshot_sha != expected_snapshot_sha:
        raise PairedScorerError(
            f"run {run_id} snapshot SHA mismatch: report says {snapshot_sha}, expected {expected_snapshot_sha}"
        )
    return {
        "run_id": run_id,
        "status": status,
        "snapshot_sha": snapshot_sha,
        "work_item_count": report.get("work_item_count"),
        "completed_count": report.get("completed_count"),
        "failed_count": report.get("failed_count"),
        "report_path": str(report_path),
    }


def load_gold(gold_path: str | Path) -> dict[str, dict]:
    rows = load_jsonl_strict(gold_path, "selection_gold")
    gold: dict[str, dict] = {}
    for row in rows:
        cid = row.get("canonical_id")
        if not isinstance(cid, str) or not cid:
            raise PairedScorerError(f"selection_gold: row without canonical_id: {row!r:.120}")
        if cid in gold:
            raise PairedScorerError(f"selection_gold: duplicate canonical_id {cid}")
        label = row.get("label")
        if label not in ("true", "false"):
            raise PairedScorerError(f"selection_gold: {cid} has invalid label {label!r}")
        class_name = row.get("class_name")
        class_id = row.get("class_id")
        # class_name is the business key; class_id is only kept as recorded in the snapshot.
        if CLASS_NAME_TO_ID.get(class_name) != class_id or CLASS_ID_TO_NAME.get(class_id) != class_name:
            raise PairedScorerError(
                f"selection_gold: {cid} class mismatch vs fixed map: class_id={class_id!r} class_name={class_name!r}"
            )
        gold[cid] = {"canonical_id": cid, "class_id": class_id, "class_name": class_name, "label": label}
    return gold


def load_inputs(inputs_path: str | Path, gold: dict[str, dict]) -> dict[str, dict]:
    """De-identified image references for the error packet (sha256/object_key only)."""
    rows = load_jsonl_strict(inputs_path, "selection_inputs")
    refs: dict[str, dict] = {}
    for row in rows:
        cid = row.get("canonical_id")
        if cid in gold:
            refs[cid] = {"image_sha256": row.get("image_sha256"), "image_object_key": row.get("image_object_key")}
    return refs


def load_decisions(path: str | Path, side: str, gold_ids: set[str]) -> tuple[dict[str, dict], dict]:
    """Strictly validate one final decisions file and index it by canonical_id."""
    tag = VERSION_TAGS[side]
    rows = load_jsonl_strict(path, f"decisions[{side}]")
    by_cid: dict[str, dict] = {}
    run_ids: set[str] = set()
    versions: set[str] = set()
    for row in rows:
        cid = row.get("canonical_id")
        if not isinstance(cid, str) or not cid:
            raise PairedScorerError(f"{path}: decision row without canonical_id: {row!r:.120}")
        if cid in by_cid:
            raise PairedScorerError(f"{path}: duplicate canonical_id {cid} (paired mode never picks first/last)")
        version = row.get("prompt_version")
        if not isinstance(version, str) or not version.endswith(tag):
            raise PairedScorerError(
                f"{path}: {cid} prompt_version {version!r} does not end with {tag!r} (version mismatch)"
            )
        versions.add(version)
        run_id = row.get("run_id")
        if isinstance(run_id, str) and run_id:
            run_ids.add(run_id)
        status, decision = normalize_decision(
            {"execution_status": _STATUS_ALIAS.get(row.get("execution_status"), row.get("execution_status")),
             "decision": row.get("decision")}
        )
        by_cid[cid] = {
            "canonical_id": cid,
            "execution_status": status,
            "decision": decision,
            "class_name": row.get("class_name"),
            "prompt_version": version,
            "safety_action": row.get("safety_action"),
        }
    got = set(by_cid)
    missing = sorted(gold_ids - got)
    extra = sorted(got - gold_ids)
    if missing or extra:
        raise PairedScorerError(
            f"{path}: canonical_id set != selection_gold ({len(got)}/{len(gold_ids)}); "
            f"missing {len(missing)} (e.g. {missing[:3]}), extra {len(extra)} (e.g. {extra[:3]})"
        )
    if len(run_ids) > 1:
        raise PairedScorerError(f"{path}: multiple run_id values {sorted(run_ids)}; one final run per file required")
    meta = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "run_id": sorted(run_ids)[0] if run_ids else None,
        "prompt_versions": sorted(versions),
    }
    return by_cid, meta


def _score_side(gold: dict[str, dict], decisions: dict[str, dict]) -> tuple[dict, dict[str, str]]:
    """Bucket every gold record. Sets are already proven identical by load_decisions."""
    totals = _empty_buckets()
    per_class: dict[int, dict] = {}
    buckets_by_cid: dict[str, str] = {}
    for cid in sorted(gold):
        grow = gold[cid]
        drow = decisions[cid]
        if drow["class_name"] is not None and drow["class_name"] != grow["class_name"]:
            raise PairedScorerError(
                f"{cid}: decisions class_name {drow['class_name']!r} != gold class_name {grow['class_name']!r}"
            )
        status, decision = normalize_decision(drow)  # idempotent; defensive for non-CLI paths
        drow["execution_status"], drow["decision"] = status, decision
        bucket = judge(grow["label"], status, decision)
        buckets_by_cid[cid] = bucket
        totals[bucket] += 1
        brow = per_class.setdefault(
            grow["class_id"],
            {"class_id": grow["class_id"], "class_name": grow["class_name"],
             "gold_true": 0, "gold_false": 0, **_empty_buckets()},
        )
        brow["gold_true" if grow["label"] == "true" else "gold_false"] += 1
        brow[bucket] += 1
    gold_true = sum(r["gold_true"] for r in per_class.values())
    gold_false = sum(r["gold_false"] for r in per_class.values())
    for row in per_class.values():
        row["dangerous_miss_rate"] = _rate(row["dangerous_miss"], row["gold_true"])
        row["false_accept_rate"] = _rate(row["false_accept"], row["gold_false"])
    report = {
        "gold_true": gold_true,
        "gold_false": gold_false,
        **totals,
        "uncertain": totals["uncertain_on_true"] + totals["uncertain_on_false"],
        "rates": {
            "dangerous_miss_rate": _rate(totals["dangerous_miss"], gold_true),
            "false_accept_rate": _rate(totals["false_accept"], gold_false),
            "uncertain_rate": _rate(
                totals["uncertain_on_true"] + totals["uncertain_on_false"], gold_true + gold_false
            ),
        },
        "per_class": [per_class[cid] for cid in sorted(per_class)],
    }
    return report, buckets_by_cid


def _paired_analysis(gold: dict[str, dict], b10: dict[str, str], b11: dict[str, str]) -> dict:
    matrix: dict[str, dict[str, int]] = {}
    improved, regressed = [], []
    new_dangerous_miss, recovered_correct_reject = [], []
    newly_uncertain, resolved_uncertain = [], []
    uncertain_buckets = {"uncertain_on_true", "uncertain_on_false"}
    for cid in sorted(gold):
        a, b = b10[cid], b11[cid]
        matrix.setdefault(a, {}).setdefault(b, 0)
        matrix[a][b] += 1
        if _BUCKET_RANK[b] < _BUCKET_RANK[a]:
            improved.append(cid)
        elif _BUCKET_RANK[b] > _BUCKET_RANK[a]:
            regressed.append(cid)
        if b == "dangerous_miss" and a != "dangerous_miss":
            new_dangerous_miss.append(cid)
        if a == "false_accept" and b == "correct_reject":
            recovered_correct_reject.append(cid)
        if b in uncertain_buckets and a not in uncertain_buckets:
            newly_uncertain.append(cid)
        if a in uncertain_buckets and b not in uncertain_buckets:
            resolved_uncertain.append(cid)
    return {
        "transition_matrix": matrix,
        "improved": improved,
        "regressed": regressed,
        "new_dangerous_miss": new_dangerous_miss,
        "recovered_correct_reject": recovered_correct_reject,
        "uncertain": {
            "newly_uncertain": newly_uncertain,
            "resolved_uncertain": resolved_uncertain,
        },
    }


def _gate(v10: dict, v11: dict) -> dict:
    checks = {
        "v11_dangerous_miss_zero": v11["dangerous_miss"] == 0,
        "v11_dangerous_miss_not_worse": v11["dangerous_miss"] <= v10["dangerous_miss"],
        "v11_false_accept_not_worse": v11["false_accept"] <= v10["false_accept"],
        "v11_correct_accept_not_worse": v11["correct_accept"] >= v10["correct_accept"],
        "v11_correct_reject_not_worse": v11["correct_reject"] >= v10["correct_reject"],
        "v11_failures_not_worse": (
            v11["protocol_failure"] + v11["technical_failure"]
            <= v10["protocol_failure"] + v10["technical_failure"]
        ),
    }
    passed = all(checks.values())
    strictly_better = (
        v11["dangerous_miss"] < v10["dangerous_miss"]
        or v11["false_accept"] < v10["false_accept"]
        or v11["correct_accept"] > v10["correct_accept"]
        or v11["correct_reject"] > v10["correct_reject"]
        or v11["protocol_failure"] + v11["technical_failure"]
        < v10["protocol_failure"] + v10["technical_failure"]
    )
    if not passed:
        verdict = "regressed"
        recommendation = "v1.1.0 门禁未通过（见 checks），保留 v1.0.0。"
    elif strictly_better:
        verdict = "improved"
        recommendation = "v1.1.0 通过门禁且至少一项严格更优，可按流程推进（locked_test 仍需另行授权）。"
    else:
        verdict = "tie_keep_v1.0"
        recommendation = "全指标平局：不得写成提升，建议保留 v1.0.0。"
    return {"checks": checks, "passed": passed, "verdict": verdict, "recommendation": recommendation}


def build_error_packet(
    gold: dict[str, dict],
    inputs: dict[str, dict],
    d10: dict[str, dict],
    d11: dict[str, dict],
    b10: dict[str, str],
    b11: dict[str, str],
) -> list[dict]:
    """De-identified rows for prompt debugging: no gold label/note/reviewer."""
    packet = []
    for cid in sorted(gold):
        a, b = b10[cid], b11[cid]
        if a in _GOOD_BUCKETS and b in _GOOD_BUCKETS:
            continue
        packet.append(
            {
                "schema_version": ERROR_PACKET_SCHEMA_VERSION,
                "canonical_id": cid,
                "class_name": gold[cid]["class_name"],
                "image": inputs.get(cid),
                "v10": _packet_side(d10[cid], a),
                "v11": _packet_side(d11[cid], b),
                "error_buckets": {"v10": a, "v11": b},
            }
        )
    return packet


def _packet_side(drow: dict, bucket: str) -> dict:
    axes = drow.get("axes")
    return {
        "final": drow["decision"],
        "execution_status": drow["execution_status"],
        "safety_action": drow.get("safety_action"),
        "axes": {k: axes.get(k) for k in ("target_present", "class_match", "scene_match")} if axes else None,
        "reason_codes": drow.get("reason_codes"),
        "bucket": bucket,
    }


def enrich_from_attempts(decisions: dict[str, dict], attempts_path: str | Path, prompt_manifest_path: str | Path, side: str) -> None:
    """Attach axes/reason_codes from a Codex run's attempts.jsonl (latest attempt wins)."""
    tag = VERSION_TAGS[side]
    manifest_rows = json.loads(Path(prompt_manifest_path).read_text(encoding="utf-8"))
    version_by_key = {row["prompt_key"]: row["version"] for row in manifest_rows}
    best: dict[str, tuple[int, str, dict]] = {}
    for row in load_jsonl_strict(attempts_path, f"attempts[{side}]"):
        version = version_by_key.get(row.get("prompt_key"))
        if not isinstance(version, str) or not version.endswith(tag):
            continue
        cid = row.get("record_key")
        if cid not in decisions:
            continue
        key = (int(row.get("attempt_no") or 0), str(row.get("attempt_id") or ""))
        if cid in best and key <= (best[cid][0], best[cid][1]):
            continue
        parsed = row.get("parsed_result")
        if isinstance(parsed, dict):
            best[cid] = (key[0], key[1], parsed)
    for cid, (_, _, parsed) in best.items():
        decisions[cid]["axes"] = parsed
        decisions[cid]["reason_codes"] = parsed.get("reason_codes")


def paired_score(
    gold: dict[str, dict],
    d10: dict[str, dict],
    d11: dict[str, dict],
    meta10: dict,
    meta11: dict,
    inputs: dict[str, dict] | None = None,
) -> tuple[dict, list[dict]]:
    r10, b10 = _score_side(gold, d10)
    r11, b11 = _score_side(gold, d11)
    calls_v10, calls_v11 = meta10["rows"], meta11["rows"]
    report = {
        "schema_version": PAIRED_SCORE_SCHEMA_VERSION,
        "scope": "candidate_box",
        "scope_note": SCOPE_NOTE,
        "inputs": {"v10": meta10, "v11": meta11},
        "calls_vs_boxes": {
            "calls_total": calls_v10 + calls_v11,
            "calls_v10": calls_v10,
            "calls_v11": calls_v11,
            "unique_boxes": len(gold),
            "note": CALLS_NOTE,
        },
        "v10": r10,
        "v11": r11,
        "paired": _paired_analysis(gold, b10, b11),
        "gate": _gate(r10, r11),
    }
    packet = build_error_packet(gold, inputs or {}, d10, d11, b10, b11)
    return report, packet


def render_markdown(report: dict, packet: list[dict]) -> str:
    v10, v11, gate, paired = report["v10"], report["v11"], report["gate"], report["paired"]
    lines = [
        "# v1.0.0 vs v1.1.0 paired selection 评分",
        "",
        f"- scope：`{report['scope']}` —— {report['scope_note']}",
        f"- 快照：`{report['snapshot']['snapshot_id']}`，gold {v10['gold_true'] + v10['gold_false']} 框"
        f"（true {v10['gold_true']} / false {v10['gold_false']}）",
        f"- 调用数 {report['calls_vs_boxes']['calls_total']}（v1.0 {report['calls_vs_boxes']['calls_v10']} + "
        f"v1.1 {report['calls_vs_boxes']['calls_v11']}），独立框 {report['calls_vs_boxes']['unique_boxes']}：{CALLS_NOTE}",
        f"- v1.0 run_id：`{report['inputs']['v10'].get('run_id')}`；v1.1 run_id：`{report['inputs']['v11'].get('run_id')}`",
        "",
        "## 总体对比",
        "",
        "| 指标 | v1.0.0 | v1.1.0 |",
        "|---|---:|---:|",
    ]
    rows = [
        ("correct_accept", "correct_accept"),
        ("dangerous_miss", "dangerous_miss"),
        ("correct_reject", "correct_reject"),
        ("false_accept", "false_accept"),
        ("uncertain_on_true", "uncertain_on_true"),
        ("uncertain_on_false", "uncertain_on_false"),
        ("protocol_failure", "protocol_failure"),
        ("technical_failure", "technical_failure"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | {v10[key]} | {v11[key]} |")
    for label, key in [
        ("dangerous_miss_rate", "dangerous_miss_rate"),
        ("false_accept_rate", "false_accept_rate"),
        ("uncertain_rate", "uncertain_rate"),
    ]:
        r10, r11 = v10["rates"][key], v11["rates"][key]
        lines.append(
            f"| {label} | {_fmt_rate(r10)} | {_fmt_rate(r11)} |"
        )
    lines += [
        "",
        "## 逐类对比（分母为各类 gold_true/gold_false）",
        "",
        "| 类别 | gold T/F | v1.0 误杀/误收 | v1.1 误杀/误收 | v1.0 dmr/far | v1.1 dmr/far |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    c10 = {row["class_name"]: row for row in v10["per_class"]}
    c11 = {row["class_name"]: row for row in v11["per_class"]}
    for name in sorted(c10, key=lambda n: c10[n]["class_id"]):
        a, b = c10[name], c11[name]
        lines.append(
            f"| {name} | {a['gold_true']}/{a['gold_false']} | "
            f"{a['dangerous_miss']}/{a['false_accept']} | {b['dangerous_miss']}/{b['false_accept']} | "
            f"{_fmt_rate(a['dangerous_miss_rate'])}/{_fmt_rate(a['false_accept_rate'])} | "
            f"{_fmt_rate(b['dangerous_miss_rate'])}/{_fmt_rate(b['false_accept_rate'])} |"
        )
    lines += [
        "",
        "## paired 变化",
        "",
        f"- 改善 {len(paired['improved'])}：{paired['improved']}",
        f"- 退化 {len(paired['regressed'])}：{paired['regressed']}",
        f"- 新增危险误杀 {len(paired['new_dangerous_miss'])}：{paired['new_dangerous_miss']}",
        f"- 恢复的正确拒绝 {len(paired['recovered_correct_reject'])}：{paired['recovered_correct_reject']}",
        f"- 新增 UNCERTAIN {len(paired['uncertain']['newly_uncertain'])}，解除 UNCERTAIN {len(paired['uncertain']['resolved_uncertain'])}",
        "",
        "迁移矩阵（v1.0 行 → v1.1 列）：",
        "",
        "```json",
        json.dumps(paired["transition_matrix"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 门禁",
        "",
        f"- passed：**{gate['passed']}**，verdict：**{gate['verdict']}**",
        f"- checks：`{json.dumps(gate['checks'], ensure_ascii=False, sort_keys=True)}`",
        f"- 建议：{gate['recommendation']}",
        "",
        "## 口径限制",
        "",
        "- 只衡量候选框判定，不覆盖漏检，不能代表整图 recall/mAP。",
        "- 准确率分母为单侧 438 框的 gold_true/gold_false；876 是调用数不是分母。",
        "- technical_failure/protocol_error 语义为 keep，不算 true/false。",
        f"- paired_error_packet 共 {len(packet)} 行（任一侧非 correct 即入包），已去标识化（无 gold label/note/reviewer）。",
        "",
    ]
    return "\n".join(lines)


def _fmt_rate(rate: dict) -> str:
    if rate["value"] is None:
        return f"- ({rate['k']}/{rate['n']})"
    lo, hi = rate["ci95"]
    return f"{rate['value']:.4f} [{lo:.4f},{hi:.4f}] ({rate['k']}/{rate['n']})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired v1.0.0 vs v1.1.0 scorer (candidate-box scope).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score", help="paired-score two final decisions files")
    p.add_argument("--v10", required=True, help="final decisions JSONL for v1.0.0")
    p.add_argument("--v11", required=True, help="final decisions JSONL for v1.1.0")
    p.add_argument("--report-v10", required=True, help="Codex run report.json proving v1.0.0 run is terminal")
    p.add_argument("--report-v11", required=True, help="Codex run report.json proving v1.1.0 run is terminal")
    p.add_argument("--snapshot-dir", default=f"data/prompt-eval/snapshots/{SNAPSHOT_ID}")
    p.add_argument("--expect-manifest-sha", default=EXPECTED_MANIFEST_SHA256,
                   help="pinned snapshot manifest SHA-256 (default: the published pg1 manifest)")
    p.add_argument("--enrich-v10", default=None, help="Codex run dir with attempts.jsonl + prompt_manifest.json")
    p.add_argument("--enrich-v11", default=None, help="Codex run dir with attempts.jsonl + prompt_manifest.json")
    p.add_argument("--report-json", required=True)
    p.add_argument("--report-md", required=True)
    p.add_argument("--error-packet", required=True)
    args = parser.parse_args(argv)

    try:
        snapshot = verify_snapshot(args.snapshot_dir, args.expect_manifest_sha)
        gold = load_gold(snapshot["gold_path"])
        inputs = load_inputs(snapshot["inputs_path"], gold)
        term10 = verify_run_terminal(args.report_v10, snapshot["manifest_sha256"])
        term11 = verify_run_terminal(args.report_v11, snapshot["manifest_sha256"])
        d10, meta10 = load_decisions(args.v10, "v10", set(gold))
        d11, meta11 = load_decisions(args.v11, "v11", set(gold))
        for side, term, meta in (("v10", term10, meta10), ("v11", term11, meta11)):
            if meta["run_id"] and meta["run_id"] != term["run_id"]:
                raise PairedScorerError(
                    f"{side}: decisions run_id {meta['run_id']} != report run_id {term['run_id']}"
                )
        if args.enrich_v10:
            enrich_from_attempts(d10, Path(args.enrich_v10) / "attempts.jsonl", Path(args.enrich_v10) / "prompt_manifest.json", "v10")
        if args.enrich_v11:
            enrich_from_attempts(d11, Path(args.enrich_v11) / "attempts.jsonl", Path(args.enrich_v11) / "prompt_manifest.json", "v11")
        report, packet = paired_score(gold, d10, d11, meta10, meta11, inputs)
        report["snapshot"] = snapshot
        report["terminal"] = {"v10": term10, "v11": term11}
    except PairedScorerError as exc:
        print(f"PAIRED SCORE FAILED: {exc}", file=sys.stderr)
        return 2

    Path(args.report_json).write_text(jdumps(report) + "\n", encoding="utf-8")
    Path(args.report_md).write_text(render_markdown(report, packet), encoding="utf-8")
    Path(args.error_packet).write_text(
        "".join(jdumps(row) + "\n" for row in packet), encoding="utf-8"
    )
    print(
        f"paired score OK: {len(gold)} boxes, calls {report['calls_vs_boxes']['calls_total']}, "
        f"gate passed={report['gate']['passed']} verdict={report['gate']['verdict']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
