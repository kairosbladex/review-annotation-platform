#!/usr/bin/env python3
"""从计分结果、冻结清单与调用台账生成中文报告（不调用模型）。

用法：
  python3 scripts/luna_live_baseline/build_report.py \
      --metrics data/luna-live-baseline/private/results/formal/metrics.json \
      --manifest data/luna-live-baseline/manifest.json \
      --ledger data/luna-live-baseline/private/results/formal/logs/formal_ledger.jsonl \
      --prior data/luna-live-baseline/private/prior_calls.jsonl \
      --out data/luna-live-baseline/REPORT.md
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger_util  # noqa: E402


def uh(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:16]


def pc(v):
    return "—" if v is None else f"{v:.2%}"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def ledger_summary(records: list[dict]) -> dict:
    return ledger_util.summarize(records, hash_units=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--prior", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    m = json.loads(args.metrics.read_text(encoding="utf-8"))
    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    led = read_jsonl(args.ledger) if args.ledger else []
    prior = read_jsonl(args.prior) if args.prior else []
    s = ledger_summary(led)
    ps = ledger_summary(prior)
    rounds = m["rounds"]
    comp = m["round_comparison"]["all"]
    sf = man["server_facts"]
    names = man["formal_scope"]["splits"]
    name = "+".join(names)

    r1 = rounds["1"]["splits"]["all"]
    r2 = rounds["2"]["splits"]["all"]

    L = []
    L.append("# Luna 当前版本基线评测报告（tests/luna-live-baseline）")
    L.append("")
    L.append(f"- 生成时间：{m.get('generated_at', man.get('generated_at', '—'))}")
    L.append(f"- 评测对象：ai-lingxi `codex-luna-review-worker`，提交 `{sf['effective_code_commit'][:12]}`，"
             f"提示词 `{sf['prompt_version']}`，模型 `{sf['model']}`")
    L.append(f"- 正式范围：{name}，{man['formal_scope']['units']} 图 / "
             f"{man['formal_scope']['labeled_boxes']} 有标签框 / {man['formal_scope']['candidates_sent']} 候选，"
             f"{man['formal_scope']['rounds']} 轮")
    L.append("")
    L.append("## 0. 结论摘要")
    L.append("")
    L.append(f"- 第 1 轮 {name} 总体准确率 **{pc(r1['accuracy'])}**（有效 {r1['n']}/{r1['n'] + r1['failed']} 框，"
             f"覆盖 {pc(r1['coverage'])}）；第 2 轮 **{pc(r2['accuracy'])}**（有效 {r2['n']} 框，"
             f"覆盖 {pc(r2['coverage'])}）。")
    L.append(f"- 两轮同一候选三值翻转 {comp['raw_flips']}/{comp['compared']} = {pc(comp['raw_flip_rate'])}；"
             f"真假翻转 {comp['decision_flips']} = {pc(comp['decision_flip_rate'])}。")
    if r1["accuracy"] is None or r2["accuracy"] is None:
        L.append("- 两轮准确率：至少一轮尚未完成，**待完成**，不计算轮次差、不生成虚构差异。")
    else:
        L.append(f"- 两轮准确率差 {abs(r1['accuracy'] - r2['accuracy']):.2%}；"
                 f"按图聚类 95% 区间见下表。该对比是同版本随机波动，不是版本改进。")
    L.append("- 本轮只测**当前版本基线**，不比较候选提示词、不声称泛化改善。")
    L.append("")
    L.append("## 1. 评测对象与只读核实")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    for k in ("worker_unit", "effective_execstart", "effective_code_commit", "effective_code_branch",
              "prompt_env_file", "prompt_version", "model", "model_provider_base_url",
              "model_reasoning_effort", "cli_version", "cli_sha256", "worker_source_sha256",
              "shared_worker_limits", "isolated_worker_limits", "formal_client_limits"):
        L.append(f"| {k} | `{sf[k]}` |")
    L.append(f"| worker 并发上限 | {sf['worker_concurrency_limit']} |")
    L.append(f"| 共享 worker | {sf['worker_bind']}；本任务未操作，MainPID 未变 |")
    L.append("")
    L.append("> 10-10 的 `25a612f` / `p000-pure-class-cards` 仅为线索，已按只读核实确认为**当前实际运行版本**"
             "（与发布 `SHA256SUMS` 逐字节一致）。")
    L.append("")
    if args.prior:
        L.append("## 2. 冻结数据与清单")
    else:
        L.append("## 2. 冻结数据与清单")
    L.append("")
    labels = man["labels"]
    L.append(f"- 冻结标签：`labels.json` sha256 `{labels['sha256'][:16]}…`，共 {labels['total_boxes']} 框，"
             f"划分 {labels['by_split']}。")
    L.append(f"- 正式评测集：{man['formal_scope']['units']} 图 / {man['formal_scope']['labeled_boxes']} 有标签框 / "
             f"{man['formal_scope']['candidates_sent']} 候选。")
    L.append(f"- 冻结 items.json sha256 `{man['items']['sha256'][:16]}…`；文件哈希核对通过："
             f"{man['integrity_checks']['all_expected_sha256_ok']}。")
    L.append(f"- 官方计分复现：p000 holdout 0.7562、p000 dev 0.7299、p000b dev 0.7124、p001 dev 0.7608、"
             f"p000b-holdout 0.7675（见 `repro_history.py`）。")
    L.append("")
    L.append("## 3. 方法与隔离")
    L.append("")
    L.append("- 每张图**全部候选一次请求**，`target_index=0..n-1`；客户端按编号回映射，假 worker 逆序返回用例已通过。")
    L.append(f"- 本任务独立 worker（**不是**共享 worker）：{sf['isolated_worker_limits']}；"
             f"客户端：{sf['formal_client_limits']}。共享 worker 限额 {sf['shared_worker_limits']}，本任务未使用。")
    L.append("- 并发 <= 2；每失败请求最多重试 1 次；总尝试上限 4N；append-only 台账每尝试 fsync；"
             "STOP 文件存在即停；每 60s 检查整机可用内存与共享容器健康，可用 <15% 或共享不健康即写 STOP。")
    L.append("- 不操作共享 worker 18396；不改提示词；不写业务数据；不部署、不重启共享服务。")
    L.append("")
    L.append("## 4. 执行台账")
    L.append("")
    L.append(f"- 正式台账 attempt={s['attempts']}（{s['by_phase']}），attempt_result={s['attempt_results']}，"
             f"按尝试计的状态 {s['by_status']}。")
    L.append(f"- 计划 2N={man['formal_scope']['planned_requests']}，上限 4N={man['formal_scope']['budget_cap_4N']}；"
             f"实际尝试 {s['attempts']}，额外尝试（按 (round,unit) 超过 1 次计）{s['extra_attempts']} 次。")
    if s["in_flight"]:
        L.append(f"- 进程被杀时的在途请求 {len(s['in_flight'])} 个（已记为调用，结果丢失；unit 以哈希列出）：")
        for u in s["in_flight"][:20]:
            L.append(f"  - {u['phase']} unit_hash={u['unit_hash']} attempt={u['attempt']} "
                     f"uid={u.get('attempt_uid')}")
    if s["failed"]:
        L.append("- 失败明细（unit 哈希）：")
        for r in s["failed"][:40]:
            L.append(f"  - {r['phase']} unit_hash={r['unit_hash']} attempt={r['attempt']} status={r['status']}")
    if s["units_over_cap"]:
        L.append(f"- **每单元 >2 次的单元 {len(s['units_over_cap'])} 个**（如实列出，不隐藏）：")
        for u in s["units_over_cap"][:20]:
            L.append(f"  - {u['phase']} unit_hash={u['unit_hash']} attempts={u['attempts']}")
    else:
        L.append("- 据实际台账核验：**没有任何单元超过每单元 2 次上限**。")
    L.append("- **临时进程违规记录（不隐藏）**：2026-10-11 01:46 前后，本任务曾用 `nohup python3` 启动"
             "**未限资源**的客户端（且当时范围仍含 dev）；01:50 被协调方叫停。随后改回 systemd-run 限资源"
             "（客户端 MemoryMax=1G / CPUQuota=100%，worker 独立限额）。该次全范围部分运行与 smoke 的"
             "全部调用已记入 prior 台账（见 `results/prior_summary.json`），未从记录中移除。")
    L.append(f"- 运行版本 vs 交付版本：本次正式运行用 evaluate.py sha256 "
             f"`{man['deliverable_code']['server_executed_evaluate_py_sha256'][:16]}…`；交付修复版 "
             f"`{man['deliverable_code']['evaluate_py_sha256'][:16]}…`（新增按 (round,unit) 持久额度与 "
             f"attempt_uid，resume 不突破每单元 2 次）。运行中客户端**未替换**；据实际台账核验，"
             f"超过每单元 2 次上限的单元数 = {len(s['units_over_cap'])}，额外尝试 "
             f"{s['extra_attempts']} 次（预期 274 唯一成功 + 2 中断多尝试 = 276）。")
    if prior:
        L.append(f"- 正式范围外的历史调用台账（prior）：attempt={ps['attempts']}，"
                 f"按尝试计状态 {ps['by_status']}；含 smoke 修复前 2 次 KeyError 丢失、修复后 2 次成功、"
                 f"更正前全范围部分运行 8 次（其中 2 次在途被杀）。")
    L.append("")
    L.append("## 5. 两轮结果")
    L.append("")
    for rid in ("1", "2"):
        d = rounds[rid]["splits"]["all"]
        L.append(f"### 第 {rid} 轮")
        L.append("")
        L.append("| 冻结框 | 有效框 | 覆盖 | 金标是 | 金标不是 | 准确率 | 真目标召回 | 非目标误放率 | 真目标驳回 | 非目标放过 | uncertain |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        L.append(f"| {d['n'] + d['failed']} | {d['n']} | {pc(d['coverage'])} | {d['gold_true']} | {d['gold_false']} | "
                 f"{pc(d['accuracy'])} | {pc(d['true_recall'])} | {pc(d['false_release_rate'])} | {d['true_rejected']} | "
                 f"{d['false_released']} | {d['uncertain']} |")
        L.append("")
        L.append("| 类别 | 框数 | 金标是 | 金标不是 | 准确率 | 真目标召回 | 非目标误放率 | uncertain |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for cls, c in d["by_class"].items():
            L.append(f"| {cls} | {c['n']} | {c['gold_true']} | {c['gold_false']} | {pc(c['accuracy'])} | "
                     f"{pc(c['true_recall'])} | {pc(c['false_release_rate'])} | {c['uncertain']} |")
        L.append("")
        cross = d["cross"]
        L.append("| 金标 | 判 true | 判 false | 判 uncertain |")
        L.append("|---|---:|---:|---:|")
        for gold in ("gold_true", "gold_false"):
            L.append(f"| {gold} | {cross[gold + ':true']} | {cross[gold + ':false']} | {cross[gold + ':uncertain']} |")
        L.append("")
        b = d["bootstrap"]
        ci = b["accuracy_ci95"]
        L.append(f"按图（unit）聚类 95% 区间：中位 {pc(b['accuracy_median'])}，"
                 f"区间 {'—' if ci is None else f'[{ci[0]:.2%}, {ci[1]:.2%}]'}（{b['n_clusters']} 图 / {b['n_boxes']} 框）。"
                 "**未解决同航次相关，不能据此证明泛化提升。**")
        gb = d.get("bootstrap_group_known")
        if gb is not None:
            gci = gb["accuracy_ci95"]
            cov = gb["coverage"]
            L.append(f"已核实航次子集聚类 95% 区间：中位 {pc(gb['accuracy_median'])}，"
                     f"区间 {'—' if gci is None else f'[{gci[0]:.2%}, {gci[1]:.2%}]'}；"
                     f"覆盖 图 {cov['units_covered']}/{cov['units_total']}、框 {cov['boxes_covered']}/{cov['boxes_total']}、"
                     f"分组 {cov['groups']}（未知航次的图被排除，未静默当作独立）。")
        L.append("")
    L.append("## 6. 两轮一致性与翻转")
    L.append("")
    L.append("| 同候选 | 三值翻转 | 三值翻转率 | 真假翻转 | 真假翻转率 |")
    L.append("|---:|---:|---:|---:|---:|")
    L.append(f"| {comp['compared']} | {comp['raw_flips']} | {pc(comp['raw_flip_rate'])} | "
             f"{comp['decision_flips']} | {pc(comp['decision_flip_rate'])} |")
    cc = comp.get("common_coverage")
    if cc:
        L.append("")
        L.append(f"- 两轮共同覆盖子集口径：共同框 {cc['n_common']}/{cc['n_denominator']}，"
                 f"全覆盖={cc['common_is_full']}；共同子集第 1 轮 {pc(cc['r1_accuracy_common'])}、"
                 f"第 2 轮 {pc(cc['r2_accuracy_common'])}。"
                 + ("两轮均 100% 覆盖，全量与共同子集一致。" if cc["common_is_full"] else
                    "存在覆盖差，轮次比较以共同子集为准。"))
    L.append("")
    L.append("## 7. 失败与有效覆盖")
    L.append("")
    for rid in ("1", "2"):
        rr = rounds[rid]
        L.append(f"- 第 {rid} 轮：图 {rr['units']}，最终失败图 {rr['units_failed']}；"
                 f"最终结果文件保留的尝试 {rr['attempts']}，完整台账尝试 {s['by_phase'].get('round' + rid, 0)}。"
                 "中断前的尝试以完整台账为准。")
    L.append("")
    L.append("## 8. 局限与未知")
    L.append("")
    L.append(f"- **航次不独立**：{man['external_verification']['source']} 发现："
             f"{man['external_verification']['finding']} {man['external_verification']['implication']}")
    vm = man.get("voyage_mapping")
    if vm:
        L.append(f"- 航次可映射覆盖率（holdout）：图 {vm['holdout_units_mapped']}/{vm['holdout_units_total']}"
                 f"（未知 {vm['holdout_units_unknown']} 图）；有标签框 映射 "
                 f"{vm['holdout_labeled_boxes_mapped']}/"
                 f"{vm['holdout_labeled_boxes_mapped'] + vm['holdout_labeled_boxes_unknown']}，"
                 f"已核实分组 {vm['distinct_groups_mapped']} 个。未映射部分不纳入航次聚类。")
    L.append("- 冻结 items.json 的 group==unit，聚类区间只能按图片（原片/帧）；"
             "**不能把 event/group 当成航次**。已知航次子集 CI 只覆盖可映射部分并标注覆盖率，"
             "未映射的 61 图（168 框）不纳入航次聚类，不能把已知航次子集的区间推广到整个集合。")
    L.append("- 137 张执行图中 6 张没有二元金标；计分及图片聚类只覆盖有金标的 131 张。"
             "已知分组的 76 张执行图中 71 张有金标，因此计分子集为 71/131 图、275/443 框。")
    L.append("- holdout 443 框不含道路裂纹，dev 也无道路裂纹；纯类道路裂纹卡本轮不命中。")
    L.append("- 只评测已有候选框的真假分类，不代表全图召回率或 mAP。")
    L.append("- 同一提示词两次运行存在随机波动（见翻转率），两轮差异不应直接归因于版本。")
    L.append("- 模型账号/上游渠道为既有 Luna 渠道；未购买额度、未扩权。")
    L.append("")
    L.append("## 9. 证据与复现")
    L.append("")
    L.append("- 指令：见 `scripts/luna_live_baseline/README.md`。")
    L.append("- 冻结来源与哈希：`data/luna-live-baseline/provenance.json`、`data/luna-live-baseline/manifest.json`。")
    L.append("- 可入库去标识结果：`data/luna-live-baseline/results/deidentified/round*.json`（稳定哈希键）、"
             "`ledger_summary.json`、`metrics_aggregate.json`、`report_tables.md`。")
    L.append("- 私有（gitignore，不入库）：`data/luna-live-baseline/private/`，含线上只读核实 `server_probe.txt`、"
             "图像预处理哈希 `preprocess_hash.json`、原始 items/labels、含业务标识的逐框结果与原始台账。")
    L.append("- 原始模型自由文本只保留在服务器 `raw/`，不进入本地或模型上下文。")
    L.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L).rstrip() + "\n", encoding="utf-8")
    print(f"写入 {args.out}（{len(L)} 行）")


def _overlap(r1, r2):
    a = r1["bootstrap"]["accuracy_ci95"]
    b = r2["bootstrap"]["accuracy_ci95"]
    if not a or not b:
        return False
    return not (a[1] < b[0] or b[1] < a[0])


if __name__ == "__main__":
    main()
