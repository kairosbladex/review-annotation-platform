#!/usr/bin/env python3
"""Build an auditable, repeatable Qwen/YOLOE evaluation report.

The report is candidate-box scoped. It never writes the annotation database and
never treats model pseudo-labels as human gold.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CLASS_MAP = {
    0: "三轮车",
    1: "两轮车",
    2: "人员",
    3: "光伏板",
    4: "光斑",
    5: "公交车",
    6: "卡车",
    7: "没戴头盔",
    8: "车",
    9: "违建",
    10: "车牌",
}
VALID_DECISIONS = {"true", "false", "uncertain"}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    rows: list[dict[str, Any]] = []
    invalid = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            invalid += 1
    return rows, invalid


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_or_abs(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def git_dirty(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _label(value: Any) -> str | None:
    value = str(value or "").strip().lower()
    return value if value in VALID_DECISIONS else None


def _metrics_for(
    gold: dict[str, dict[str, Any]], predictions: dict[str, str],
) -> dict[str, Any]:
    confusion: Counter[tuple[str, str]] = Counter()
    missing = 0
    for candidate_id, record in gold.items():
        prediction = predictions.get(candidate_id)
        if prediction is None:
            missing += 1
            continue
        confusion[(str(record["label"]), prediction)] += 1

    total = len(gold)
    matched = total - missing
    correct = sum(confusion[(label, label)] for label in ("true", "false"))
    resolved = sum(
        count for (label, prediction), count in confusion.items()
        if prediction in {"true", "false"}
    )
    dangerous_miss = confusion[("true", "false")]
    false_accept = confusion[("false", "true")]
    uncertain_true = confusion[("true", "uncertain")]
    uncertain_false = confusion[("false", "uncertain")]
    return {
        "gold_total": total,
        "matched_predictions": matched,
        "missing_predictions": missing,
        "resolved_predictions": resolved,
        "resolved_coverage": ratio(resolved, total),
        "correct": correct,
        "agreement_including_uncertain": ratio(correct, total),
        "binary_accuracy_on_resolved": ratio(correct, resolved),
        "dangerous_miss": dangerous_miss,
        "dangerous_miss_rate": ratio(dangerous_miss, sum(1 for item in gold.values() if item["label"] == "true")),
        "false_accept": false_accept,
        "false_accept_rate": ratio(false_accept, sum(1 for item in gold.values() if item["label"] == "false")),
        "uncertain_on_true": uncertain_true,
        "uncertain_on_false": uncertain_false,
        "technical_or_missing": missing,
        "confusion": {
            gold_label: {
                predicted_label: confusion[(gold_label, predicted_label)]
                for predicted_label in ("true", "false", "uncertain")
            }
            for gold_label in ("true", "false")
        },
    }


def _human_gold(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows, invalid = read_jsonl(path)
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    auto_rows = 0
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        if not str(row.get("labeled_by") or "").strip():
            auto_rows += 1
            continue
        label = _label(row.get("human_label") or row.get("label"))
        if label in {"true", "false"}:
            by_candidate[candidate_id].append({"label": label, "class_name": row.get("class_name") or "unknown"})

    conflicts = {candidate_id for candidate_id, values in by_candidate.items() if {v["label"] for v in values} > {values[0]["label"]}}
    gold: dict[str, dict[str, Any]] = {}
    for candidate_id, values in by_candidate.items():
        if candidate_id in conflicts:
            continue
        gold[candidate_id] = values[0]

    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"true": 0, "false": 0})
    for record in gold.values():
        by_class[str(record["class_name"])][str(record["label"])] += 1
    return gold, {
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "raw_rows": len(rows),
        "invalid_rows": invalid,
        "human_gold_rows": len(gold),
        "auto_rows_excluded": auto_rows,
        "conflicting_candidate_ids_excluded": len(conflicts),
        "labels": dict(Counter(record["label"] for record in gold.values())),
        "by_class": dict(sorted(by_class.items())),
    }


def _predictions(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    rows, invalid = read_jsonl(path)
    values: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row_order, attempt in enumerate(rows):
        for decision in attempt.get("decisions") or []:
            if not isinstance(decision, dict):
                continue
            candidate_id = str(decision.get("candidate_id") or "").strip()
            label = _label(decision.get("model_label"))
            if candidate_id and label:
                values[candidate_id].append((row_order, label))
    predictions = {candidate_id: entries[-1][1] for candidate_id, entries in values.items()}
    return predictions, {
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "attempt_rows": len(rows),
        "invalid_rows": invalid,
        "candidate_predictions": len(predictions),
        "duplicate_candidate_predictions": sum(max(0, len(entries) - 1) for entries in values.values()),
    }


def _service_evaluation(
    *,
    name: str,
    summary_path: Path,
    attempts_path: Path,
    gold: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = read_json(summary_path) or {}
    predictions, source = _predictions(attempts_path)
    result = {
        "name": name,
        "summary": summary,
        "source": source,
        "evaluation": _metrics_for(gold, predictions) if gold else None,
    }
    return result


def _prompt_lineage(snapshot_dir: Path | None, catalog_dir: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if snapshot_dir and snapshot_dir.is_dir():
        manifest_path = snapshot_dir / "snapshot_manifest.json"
        manifest = read_json(manifest_path) or {}
        result["snapshot"] = {
            "path": str(snapshot_dir.resolve()),
            "snapshot_id": manifest.get("snapshot_id") or snapshot_dir.name,
            "schema_version": manifest.get("schema_version"),
            "manifest_sha256": sha256_file(manifest_path),
            "selection_rows": sum(1 for _ in (snapshot_dir / "selection_inputs.jsonl").open(encoding="utf-8")) if (snapshot_dir / "selection_inputs.jsonl").is_file() else None,
            "locked_test_rows": sum(1 for _ in (snapshot_dir / "locked_test_inputs.jsonl").open(encoding="utf-8")) if (snapshot_dir / "locked_test_inputs.jsonl").is_file() else None,
        }
    if catalog_dir and catalog_dir.is_dir():
        catalog_path = catalog_dir / "prompts/ktd/bundles/ktd-three-axis-v1.0.0/catalog.json"
        catalog = read_json(catalog_path) or {}
        acceptance_path = catalog_dir / "prompts/ktd/ACCEPTANCE_KTD-PROMPT-CATALOG.md"
        result["grok_catalog"] = {
            "path": str(catalog_dir.resolve()),
            "commit": git_head(catalog_dir),
            "dirty": git_dirty(catalog_dir),
            "catalog_sha256": catalog.get("sha256") or sha256_file(catalog_path),
            "approval_status": catalog.get("approval_status"),
            "acceptance_sha256": sha256_file(acceptance_path),
        }
    return result


def _prompt_eval_report(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.is_file():
        return None
    report = read_json(path) or {}
    prompts = {}
    for prompt_key, values in (report.get("prompts") or {}).items():
        prompts[str(prompt_key)] = {
            key: values.get(key)
            for key in (
                "total_cases",
                "valid_json_count",
                "valid_json_rate",
                "structural_failures",
                "scored",
                "correct",
                "accuracy",
                "confidence_interval_95",
            )
            if key in values
        }
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "run_id": report.get("run_id"),
        "mode": report.get("mode"),
        "status": report.get("status"),
        "lineage": report.get("lineage"),
        "generation": {
            key: (report.get("generation") or {}).get(key)
            for key in ("stream", "think", "options")
        },
        "canonical_case_count": report.get("canonical_case_count"),
        "attempt_count": report.get("attempt_count"),
        "dispatch_message_count": report.get("dispatch_message_count"),
        "observed_max_concurrency": report.get("observed_max_concurrency"),
        "selection_gate": report.get("selection_gate"),
        "prompts": prompts,
    }


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    gold, gold_summary = _human_gold(args.gold_annotations)
    services = {
        "8188": _service_evaluation(
            name="8188 review-agent",
            summary_path=args.service_8188_summary,
            attempts_path=args.service_8188_attempts,
            gold=gold,
        ),
        "8189": _service_evaluation(
            name="8189 review-agent",
            summary_path=args.service_8189_summary,
            attempts_path=args.service_8189_attempts,
            gold=gold,
        ),
    }
    shadow_metrics = read_json(args.shadow_metrics) or {}
    qwen_train = read_json(args.qwen_train_summary) or {}
    formal_doc = args.yoloe_report if args.yoloe_report.is_file() else None
    prompt_lineage = _prompt_lineage(args.prompt_snapshot_dir, args.grok_catalog_dir)
    prompt_eval = _prompt_eval_report(args.prompt_eval_report)
    catalog_status = (prompt_lineage.get("grok_catalog") or {}).get("approval_status")
    locked_test_note = (
        "Locked-test 仍未在本报告中执行；catalog 已 approved，但仍需单独的一次性 locked_test 授权。"
        if catalog_status == "approved"
        else "Locked-test 未执行；Grok prompt catalog 尚未处于 approved 状态。"
    )

    # Re-read predictions once for the per-class table; service evaluation keeps
    # its public shape compact while this table carries class denominators.
    service_predictions = {
        service_id: _predictions(Path(service["source"]["source"]))[0]
        for service_id, service in services.items()
        if service["source"].get("source")
    }
    per_class: dict[str, Any] = {
        class_name: {}
        for class_name in sorted({str(record["class_name"]) for record in gold.values()})
    }
    for class_name in per_class:
        class_gold = {
            candidate_id: record
            for candidate_id, record in gold.items()
            if str(record["class_name"]) == class_name
        }
        for service_id, predictions in service_predictions.items():
            per_class[class_name][service_id] = _metrics_for(class_gold, predictions)

    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "scope": "candidate_box",
        "scope_note": "Candidate-box evaluation; it is not whole-image recall or mAP.",
        "repo": {"path": str(repo_root), "commit": git_head(repo_root), "dirty": git_dirty(repo_root)},
        "gold": gold_summary,
        "services": services,
        "per_class": per_class,
        "qwen_shadow": {
            "metrics_path": str(args.shadow_metrics.resolve()),
            "metrics_sha256": sha256_file(args.shadow_metrics),
            "model": shadow_metrics.get("model"),
            "status": shadow_metrics.get("status"),
            "input_count": shadow_metrics.get("input_count"),
            "status_counts": shadow_metrics.get("status_counts"),
            "protocol_valid_rate": ratio(
                int((shadow_metrics.get("status_counts") or {}).get("succeeded", 0)),
                int(shadow_metrics.get("input_count") or 0),
            ),
            "invalid_json": (shadow_metrics.get("errors") or {}).get("invalid_json", 0),
            "pseudo_reference": shadow_metrics.get("comparison"),
            "latency_ms": shadow_metrics.get("latency_ms"),
            "accuracy_status": "not_human_gold; pseudo-label agreement only",
        },
        "prompt_eval": prompt_eval,
        "training": {
            "qwen_qlora_smoke": {
                "summary_path": str(args.qwen_train_summary.resolve()),
                "summary_sha256": sha256_file(args.qwen_train_summary),
                "summary": qwen_train,
                "status": (
                    "smoke_only_pseudo_labels"
                    if qwen_train.get("pseudo_labels_only")
                    else "not_found" if not qwen_train else "see_summary"
                ),
            },
            "yoloe_gpu_smoke_report": {
                "path": str(formal_doc.resolve()) if formal_doc else None,
                "sha256": sha256_file(formal_doc) if formal_doc else None,
                "status": "documented_smoke_only" if formal_doc else "not_found",
                "production_deployed": False,
            },
        },
        "prompt_lineage": prompt_lineage,
        "caveats": [
            "8188/8189 service results are evaluated only where candidate_id matches human-labeled rows.",
            "UNCERTAIN is reported separately; it is not silently converted to false or counted as a safe accept.",
            "The Qwen shadow comparison reference is GPT/model pseudo-labels, not human gold.",
            "Qwen smoke training is not a deployable production weight and no online service was changed by this report.",
            locked_test_note,
        ],
    }
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    snapshot["snapshot_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    snapshot["generated_at"] = datetime.now(timezone.utc).isoformat()
    return snapshot


def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.2f}%"


def render_markdown(snapshot: dict[str, Any]) -> str:
    gold = snapshot["gold"]
    lines = [
        "# Qwen3-VL / YOLOE 复检评测闭环报告",
        "",
        f"> 生成时间：{snapshot['generated_at']}  | 证据快照：`{snapshot['snapshot_hash'][:16]}`",
        "> 评测范围：`candidate_box`，即已有候选框的真假/存疑判定，不是整图 recall/mAP。",
        "",
        "## 当前结论",
        "",
    ]
    for service_id, service in snapshot["services"].items():
        evaluation = service.get("evaluation") or {}
        lines.append(
            f"- **{service_id}**：匹配人工金标 {evaluation.get('matched_predictions', 0)}/"
            f"{evaluation.get('gold_total', 0)}；二态覆盖 {_pct(evaluation.get('resolved_coverage'))}；"
            f"已解析二态准确率 {_pct(evaluation.get('binary_accuracy_on_resolved'))}；"
            f"危险误杀 {evaluation.get('dangerous_miss', 0)}，存疑 {evaluation.get('uncertain_on_true', 0) + evaluation.get('uncertain_on_false', 0)}。"
        )
    shadow = snapshot["qwen_shadow"]
    lines.append(
        f"- **Qwen3-VL 影子运行**：{shadow.get('input_count', 0)} 条，协议合法率 "
        f"{_pct(shadow.get('protocol_valid_rate'))}，`invalid_json={shadow.get('invalid_json', 0)}`；"
        "与 GPT 伪标签的差异仅作诊断，不称为准确率。"
    )
    lines += [
        "",
        "## 人工金标范围",
        "",
        f"- 原始导出：{gold['raw_rows']} 条；人工金标：{gold['human_gold_rows']} 条；自动标注排除：{gold['auto_rows_excluded']} 条。",
        f"- 人工标签：`true={gold['labels'].get('true', 0)}`，`false={gold['labels'].get('false', 0)}`。",
        f"- 金标文件：`{gold['source']}`；SHA-256：`{gold['source_sha256']}`。",
        "",
        "## 服务指标",
        "",
        "| 服务 | 运行摘要 | 金标匹配 | 覆盖率 | 二态准确率 | 危险误杀 | 误接受 | 存疑 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for service_id, service in snapshot["services"].items():
        summary = service.get("summary") or {}
        evaluation = service.get("evaluation") or {}
        lines.append(
            f"| {service_id} | images={summary.get('images_succeeded', '-')}, candidates={summary.get('candidates_applied', '-')} "
            f"(true/false/uncertain={summary.get('model_true', '-')}/{summary.get('model_false', '-')}/{summary.get('model_uncertain', '-')}) | "
            f"{evaluation.get('matched_predictions', 0)}/{evaluation.get('gold_total', 0)} | {_pct(evaluation.get('resolved_coverage'))} | "
            f"{_pct(evaluation.get('binary_accuracy_on_resolved'))} | {evaluation.get('dangerous_miss', 0)} | {evaluation.get('false_accept', 0)} | "
            f"{evaluation.get('uncertain_on_true', 0) + evaluation.get('uncertain_on_false', 0)} |"
        )
    lines += ["", "## 逐类人工金标指标（以 8188 为主）", "", "| 类别 | gold true | gold false | 覆盖率 | 二态准确率 | 危险误杀 | 误接受 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for class_name, values in snapshot["per_class"].items():
        counts = gold["by_class"].get(class_name, {})
        evaluation = values.get("8188") or {}
        lines.append(
            f"| {class_name} | {counts.get('true', 0)} | {counts.get('false', 0)} | {_pct(evaluation.get('resolved_coverage'))} | "
            f"{_pct(evaluation.get('binary_accuracy_on_resolved'))} | {evaluation.get('dangerous_miss', 0)} | {evaluation.get('false_accept', 0)} |"
        )
    prompt_eval = snapshot.get("prompt_eval")
    if prompt_eval:
        lines += [
            "",
            "## Prompt Eval 真实 GPU 运行",
            "",
            f"- run：`{prompt_eval.get('run_id')}`；mode：`{prompt_eval.get('mode')}`；status：`{prompt_eval.get('status')}`；最大实测并发：`{prompt_eval.get('observed_max_concurrency')}`。",
            f"- selection gate：`{json.dumps(prompt_eval.get('selection_gate'), ensure_ascii=False, sort_keys=True)}`。",
        ]
        for prompt_key, values in prompt_eval.get("prompts", {}).items():
            lines.append(
                f"- `{prompt_key}`：合法 JSON {values.get('valid_json_count', '-')}/{values.get('total_cases', '-')}；"
                f"评分 {values.get('scored', '-')}；准确率 {_pct(values.get('accuracy'))}。"
            )
    lines += [
        "",
        "## 训练与 Prompt 状态",
        "",
        f"- Qwen QLoRA：`{snapshot['training']['qwen_qlora_smoke']['status']}`；详见 `{snapshot['training']['qwen_qlora_smoke']['summary_path']}`。",
        f"- YOLOE：`{snapshot['training']['yoloe_gpu_smoke_report']['status']}`；该证据不等于线上权重已发布。",
        f"- Prompt lineage：`{json.dumps(snapshot['prompt_lineage'], ensure_ascii=False, sort_keys=True)}`。",
        "",
        "## 解释边界",
        "",
    ]
    lines.extend(f"- {item}" for item in snapshot["caveats"])
    lines += ["", "## 下一轮", "", "1. 扩充人工金标，优先补齐各类别的正例、负例和难例。", "2. 修复 Qwen 输出协议并用同一冻结金标重跑 canary/selection。", "3. 只有 challenger 满足危险误杀硬门禁且不劣于 baseline，才进入后续 locked-test；审批状态未改变前不执行。", ""]
    return "\n".join(lines)


def render_html(snapshot: dict[str, Any]) -> str:
    title = "Qwen3-VL / YOLOE 复检评测闭环报告"
    rows = []
    for service_id, service in snapshot["services"].items():
        evaluation = service.get("evaluation") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(service_id)}</td>"
            f"<td>{evaluation.get('matched_predictions', 0)}/{evaluation.get('gold_total', 0)}</td>"
            f"<td>{html.escape(_pct(evaluation.get('resolved_coverage')))}</td>"
            f"<td>{html.escape(_pct(evaluation.get('binary_accuracy_on_resolved')))}</td>"
            f"<td>{evaluation.get('dangerous_miss', 0)}</td>"
            f"<td>{evaluation.get('false_accept', 0)}</td>"
            f"<td>{evaluation.get('uncertain_on_true', 0) + evaluation.get('uncertain_on_false', 0)}</td>"
            "</tr>"
        )
    bars = []
    for class_name, values in snapshot["per_class"].items():
        evaluation = values.get("8188") or {}
        width = round(float(evaluation.get("binary_accuracy_on_resolved") or 0) * 100, 2)
        bars.append(f"<div class='bar-row'><span>{html.escape(class_name)}</span><div class='bar'><i style='width:{width}%'></i></div><b>{width:.2f}%</b></div>")
    caveats = "".join(f"<li>{html.escape(item)}</li>" for item in snapshot["caveats"])
    prompt_eval = snapshot.get("prompt_eval")
    prompt_eval_html = ""
    if prompt_eval:
        prompt_rows = []
        for prompt_key, values in prompt_eval.get("prompts", {}).items():
            prompt_rows.append(
                "<tr>"
                f"<td>{html.escape(str(prompt_key))}</td>"
                f"<td>{values.get('valid_json_count', '-')}/{values.get('total_cases', '-')}</td>"
                f"<td>{values.get('scored', '-')}</td>"
                f"<td>{html.escape(_pct(values.get('accuracy')))}</td>"
                "</tr>"
            )
        prompt_eval_html = (
            f"<h2>Prompt Eval 真实 GPU 运行</h2>"
            f"<p>run：<code>{html.escape(str(prompt_eval.get('run_id')))}</code>；"
            f"status：{html.escape(str(prompt_eval.get('status')))}；"
            f"最大实测并发：{html.escape(str(prompt_eval.get('observed_max_concurrency')))}。</p>"
            f"<p>selection gate：<code>{html.escape(json.dumps(prompt_eval.get('selection_gate'), ensure_ascii=False, sort_keys=True))}</code></p>"
            "<table><thead><tr><th>Prompt</th><th>合法 JSON</th><th>评分数</th><th>准确率</th></tr></thead>"
            f"<tbody>{''.join(prompt_rows)}</tbody></table>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{color-scheme:light;--ink:#18212b;--muted:#65717c;--line:#d9e0e6;--accent:#176b87;--warn:#b64b35;--bg:#f6f8fa}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:36px 24px 60px}} h1{{font-size:28px;margin:0 0 8px}} h2{{font-size:19px;margin:34px 0 12px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.meta,.note{{color:var(--muted);font-size:13px}} .callout{{background:#fff;border-left:4px solid var(--accent);padding:14px 16px;margin:18px 0}} table{{width:100%;border-collapse:collapse;background:#fff}} th,td{{text-align:left;border-bottom:1px solid var(--line);padding:9px 10px;white-space:nowrap}} th{{font-size:13px;color:var(--muted);font-weight:600}} .bar-row{{display:grid;grid-template-columns:110px 1fr 80px;gap:12px;align-items:center;margin:8px 0}} .bar{{height:13px;background:#e2e8ed;border-radius:2px;overflow:hidden}} .bar i{{display:block;height:100%;background:var(--accent)}} .bar-row b{{font-size:13px;text-align:right}} li{{margin:5px 0}} code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}}
@media (max-width:700px){{main{{padding:24px 14px}} h1{{font-size:23px}} table{{display:block;overflow:auto}} .bar-row{{grid-template-columns:84px 1fr 64px;font-size:13px}}}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="meta">生成时间：{html.escape(snapshot['generated_at'])} · 证据快照：<code>{html.escape(snapshot['snapshot_hash'])}</code></p>
<div class="callout">评测范围：<code>candidate_box</code>。这是已有候选框的判定评测，不是整图 recall/mAP；存疑结果单列，不折算为假目标。</div>
<h2>服务指标</h2><table><thead><tr><th>服务</th><th>匹配人工金标</th><th>二态覆盖率</th><th>二态准确率</th><th>危险误杀</th><th>误接受</th><th>存疑</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>8188 各类别二态准确率</h2>{''.join(bars)}
<h2>Qwen 影子运行</h2><p>模型：<code>{html.escape(str(snapshot['qwen_shadow'].get('model')))}</code>；输入：{snapshot['qwen_shadow'].get('input_count', 0)}；协议合法率：{html.escape(_pct(snapshot['qwen_shadow'].get('protocol_valid_rate')))}；invalid_json：{snapshot['qwen_shadow'].get('invalid_json', 0)}。</p><p class="note">影子运行与 GPT 伪标签的比较只用于诊断，不是人工金标准准确率。</p>
{prompt_eval_html}
<h2>边界与后续</h2><ul>{caveats}</ul><p class="meta">完整结构化数据见同目录 <code>latest.json</code>，历史快照见 <code>history.jsonl</code>。</p>
</main></body></html>"""


def update_outputs(snapshot: dict[str, Any], output_dir: Path, markdown: str, page: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "latest.json"
    history = output_dir / "history.jsonl"
    previous: list[dict[str, Any]] = []
    if history.is_file():
        previous, _ = read_jsonl(history)
    previous = [row for row in previous if row.get("snapshot_hash") != snapshot["snapshot_hash"]]
    previous.append(snapshot)
    history.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in previous), encoding="utf-8")
    latest.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "报告.md").write_text(markdown, encoding="utf-8")
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Desktop" / "训练报告")
    parser.add_argument("--gold-annotations", type=Path, default=repo_default / "data/gold-export-live-20260903/annotations.jsonl")
    parser.add_argument("--service-8188-summary", type=Path, default=repo_default / "data/ktd-8188-kimi-prefill-20260903/summary.json")
    parser.add_argument("--service-8188-attempts", type=Path, default=repo_default / "data/ktd-8188-kimi-prefill-20260903/attempts.jsonl")
    parser.add_argument("--service-8189-summary", type=Path, default=repo_default / "data/ktd-8189-prefill-20260903/summary.json")
    parser.add_argument("--service-8189-attempts", type=Path, default=repo_default / "data/ktd-8189-prefill-20260903/attempts.jsonl")
    parser.add_argument("--shadow-metrics", type=Path, default=repo_default / "data/agent-runs/ktd-qwen-shadow-20260903/metrics.json")
    parser.add_argument("--qwen-train-summary", type=Path, default=repo_default / "data/agent-runs/ktd-qwen-qlora-smoke-20260903/adapter/smoke_summary.json")
    parser.add_argument("--yoloe-report", type=Path, default=repo_default / "docs/KTD_GPU_SMOKE_TRAINING_20260903.md")
    parser.add_argument("--prompt-eval-report", type=Path)
    parser.add_argument("--prompt-snapshot-dir", type=Path)
    parser.add_argument("--grok-catalog-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = build_snapshot(args)
    update_outputs(snapshot, args.output_dir.expanduser().resolve(), render_markdown(snapshot), render_html(snapshot))
    print(json.dumps({
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "snapshot_hash": snapshot["snapshot_hash"],
        "gold_rows": snapshot["gold"]["human_gold_rows"],
        "service_8188": snapshot["services"]["8188"]["evaluation"],
        "qwen_protocol_valid_rate": snapshot["qwen_shadow"]["protocol_valid_rate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
