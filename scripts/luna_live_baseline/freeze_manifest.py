#!/usr/bin/env python3
"""冻结 Luna 当前版本基线的清单与哈希。

只读输入：`data/luna-live-baseline/frozen/` 下由 yoloe `feature/luna-class-prompt-release`
分支 `data/luna-prompt-autotune-final-20261009/` 取出的冻结资料，以及
`feature/luna-worker-failure-log` 提交 25a612f 的线上 worker 源码与提示词。

产出：`data/luna-live-baseline/frozen/baseline_manifest.json`，记录：
  - 逐文件 sha256（并与发布自带的 MANIFEST.sha256.json 交叉核对）
  - 标签框数、dev/holdout 划分、逐类真假数
  - 评测单元数、候选框数、路由（纯类别 / 混合）
  - 线上 worker / CLI / 模型 / 提示词版本等只读核实事实

本脚本不下模型、不读凭据、不写业务数据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

# 只读核实的线上事实（2026-10-11 01:41 CST，ai-lingxi user 会话）。
# 值来自 systemctl --user cat / sha256sum / codex --version 的只读输出，
# 不含任何环境密钥或账号材料。
SERVER_FACTS = {
    "host": "ai-lingxi (192.168.130.238)",
    "observed_at_local": "2026-10-11T01:41:00+08:00",
    "worker_unit": "codex-luna-review-worker.service",
    "worker_active_since": "2026-10-10 23:16:32 CST",
    "worker_main_pid": 1203610,
    "effective_execstart": (
        "/usr/bin/python3 /data/workspace/jzheng/codex-luna-review/releases/"
        "25a612f28403bed971db96430ab638508589b5f6/scripts/run_codex_luna_worker.py"
    ),
    "effective_code_commit": "25a612f28403bed971db96430ab638508589b5f6",
    "effective_code_commit_subject": "fix: Luna 执行服务失败时记录脱敏的 CLI 错误摘要",
    "effective_code_branch": "feature/luna-worker-failure-log",
    "prompt_env_file": (
        "/data/workspace/jzheng/codex-luna-review/releases/"
        "47ca276bf77f4901627b12e81b49b5363555757b/prompt.env"
    ),
    "prompt_version": "p000-pure-class-cards",
    "prompt_suffix": "不运行任何工具，只返回schema中的JSON；target_index必须使用下面原编号，解释使用中文。候选：",
    "model": "gpt-5.6-luna",
    "model_provider_base_url": "https://sub.butterfly2sea.us",
    "model_reasoning_effort": "none",
    "model_wire_api": "responses",
    "request_max_retries": 0,
    "cli_binary": "bin/codex-0.162.0-alpha.17.2/codex",
    "cli_version": "codex-cli 0.162.0-alpha.17.2",
    "cli_sha256": "73b73b048546993298234a295de01ef2424e7a8b29def9d0ecb1ca7cb77500d4",
    "worker_concurrency_limit": 20,
    "worker_bind": "192.168.130.238:18396 (共享，本任务不使用)",
    "shared_worker_limits": "MemoryMax=8G CPUQuota=400% MemorySwapMax=0（线上/共享 worker，本任务未操作）",
    "isolated_worker_limits": "MemoryMax=4G CPUQuota=200% MemorySwapMax=0 @127.0.0.1:18398（本任务独立实例，实测 CPUQuotaPerSecUSec=2s）",
    "worker_resource_guard": "MemAvailable/MemTotal < 0.15 时终止自身请求并拒绝新请求",
    "auth_material": "/home/user/.codex/auth.json (仅存在性核对，未读取内容)",
    "worker_source_sha256": "b909136b16f0f357e58f9d3e221482e122195b34e7cec124bf0760aca59495e1",
    "server_resources_snapshot": "251G RAM / 209G available / 64 cores / /data 11T avail / 6 GPU (L20×2,T4×4)",
    "formal_client_limits": "MemoryMax=1G CPUQuota=100% MemorySwapMax=0",
    "shared_worker_untouched": "codex-luna-review-worker MainPID=1203610 未变（2026-10-10 23:16:32 起）",
    "resource_guard": "客户端每 60s 检查 MemAvailable/MemTotal>=0.15 与共享容器 healthy，否则写 STOP",
    "cleanup": "任务结束停止本任务独立 worker 与客户端；保留工作树供验收；共享 worker 与测试服务未动",
}

# 线上发布 SHA256SUMS 中的期望哈希（25a612f 与 47ca276 两版提示词逐字节相同）。
EXPECTED_SHA256 = {
    "worker_25a612f/run_codex_luna_worker.py": "b909136b16f0f357e58f9d3e221482e122195b34e7cec124bf0760aca59495e1",
    "worker_25a612f/luna_prompts/p000.json": "0f265c686c73c751c545d440e3e3e5a1de7cc8206ca86c73e6710412f70e67e9",
    "worker_25a612f/luna_prompts/p002.json": "8da46d8cf25ce51c92291e586561e0cedfa7c268dd3d03bb824f22e7366ffa45",
    "worker_25a612f/luna_prompts/card-road-crack-20261010.json": "4ca5e02a16b99071eaa87b8846bd2d3039518ea6c71587e241b1a0dcbf31539a",
}

# 评测划分与规模（任务给定：最终历史标签 dev 685 / holdout 443 框）。
EVAL_SPLITS = ("dev", "holdout")

# 协调方 2026-10-11 更正后的正式范围：只跑历史 holdout。
FORMAL_SCOPE = {
    "splits": ["holdout"],
    "units": 137,
    "labeled_boxes": 443,
    "candidates_sent": 452,
    "rounds": 2,
    "planned_requests": 274,
    "max_retry_per_request": 1,
    "budget_cap_4N": 548,
    "concurrency": 2,
    "note": "dev 289 图不跑；无标签池不跑。",
    "server_executed_evaluate_sha256": "61d41f91e2bd462d91d634c3195263a0e343c8efcf6cde03066d4ecfabc5d6b3",
}

# 外部只读核验（协调方 Codex 会话，2026-10-11）：本报告采信并在正文标注。
EXTERNAL_VERIFICATION = {
    "source": "协调方 Codex 会话只读核验",
    "finding": "历史 dev 中至少 67 个目标与 holdout 同航次（含人员 64 个）；旧 holdout 不是按航次独立的新盲测。",
    "implication": (
        "dev/holdout 的独立性被削弱；冻结 items.json 中 group==unit，只有图片级键，"
        "没有可直接复算的航次键，故本任务不独立重算该数字，而在报告中明确标注该局限。"
    ),
}



def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def class_counts(labels: dict, split: str) -> dict:
    out: dict[str, Counter] = {}
    for v in labels.values():
        if v["split"] != split:
            continue
        c = out.setdefault(v["class_name"], Counter())
        c["n"] += 1
        c["gold_true" if v["label"] == "true" else "gold_false"] += 1
    return {k: dict(v) for k, v in sorted(out.items(), key=lambda x: -x[1]["n"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=Path, default=Path("data/luna-live-baseline/private/frozen"))
    ap.add_argument("--out", type=Path, default=Path("data/luna-live-baseline/manifest.json"))
    ap.add_argument("--release-manifest", type=Path, default=None,
                    help="发布自带 MANIFEST.sha256.json（默认 frozen/MANIFEST.sha256.json）")
    ap.add_argument("--grouping", type=Path, default=None,
                    help="私有 unit->航次哈希映射；提供时在清单记录可映射覆盖率（只存计数）")
    args = ap.parse_args()

    frozen = args.frozen
    release_manifest_path = args.release_manifest or (frozen / "MANIFEST.sha256.json")
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))

    labels = json.loads((frozen / "labels.json").read_text(encoding="utf-8"))
    items = json.loads((frozen / "items.json").read_text(encoding="utf-8"))
    build = json.loads((frozen / "build.json").read_text(encoding="utf-8"))

    # ---- 文件哈希 + 发布清单核对 ----
    files = {}
    for name, expected in EXPECTED_SHA256.items():
        p = frozen / name
        got = sha256_file(p)
        files[name] = {"sha256": got, "expected": expected, "ok": got == expected}
    # 本仓库实际提交的 worker 源码副本（文件名不同，内容与线上一致）
    files["worker_25a612f/run_codex_luna_worker.py"]["note"] = "与线上 ExecStart 指向的 25a612f 脚本逐字节相同"

    manifest_check = {}
    for rel, exp in release_manifest.items():
        p = frozen / rel
        if p.exists():
            got = sha256_file(p)
            manifest_check[rel] = {"sha256": got, "expected": exp, "ok": got == exp}

    # ---- 规模统计 ----
    label_splits = Counter(v["split"] for v in labels.values())
    unit_splits = Counter(u["split"] for u in items)
    eval_units = [u for u in items if u["split"] in EVAL_SPLITS]
    cand_by_split = Counter()
    for u in items:
        cand_by_split[u["split"]] += len(u["candidates"])
    routed_pure = Counter()
    routed_mixed = 0
    for u in eval_units:
        names = {c["class_name"] for c in u["candidates"]}
        if len(names) == 1:
            routed_pure[(u["split"], next(iter(names)))] += 1
        else:
            routed_mixed += 1

    all_label_keys = set(labels)
    all_cand_keys = {c["key"] for u in items for c in u["candidates"]}
    eval_cand_keys = {c["key"] for u in eval_units for c in u["candidates"]}

    voyage_mapping = None
    if args.grouping and args.grouping.exists():
        g = json.loads(args.grouping.read_text(encoding="utf-8"))
        hold_units = [u for u in items if u["split"] == "holdout"]
        mapped = [u for u in hold_units if g.get(u["unit"])]
        unknown = [u for u in hold_units if not g.get(u["unit"])]

        def nlab(us):
            return sum(1 for u in us for c in u["candidates"] if c["key"] in labels)

        voyage_mapping = {
            "source": "协调方私有映射 unit->已核实 task/航次哈希或 null（不入库）",
            "holdout_units_total": len(hold_units),
            "holdout_units_mapped": len(mapped),
            "holdout_units_unknown": len(unknown),
            "holdout_labeled_boxes_mapped": nlab(mapped),
            "holdout_labeled_boxes_unknown": nlab(unknown),
            "distinct_groups_mapped": len({g[u["unit"]] for u in mapped}),
            "note": "未映射图不纳入航次聚类，不被静默当作独立。",
        }

    manifest = {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "Luna 当前版本基线评测（baseline）",
        "frozen_source": {
            "repo": "yoloe-algorithm-service",
            "branch": "feature/luna-class-prompt-release",
            "path": "data/luna-prompt-autotune-final-20261009/",
            "note": "最终冻结资料；dev 685 / holdout 443 为更正后最终历史标签",
        },
        "labels": {
            "file": "labels.json",
            "sha256": sha256_file(frozen / "labels.json"),
            "total_boxes": len(labels),
            "by_split": dict(label_splits),
            "by_split_class": {s: class_counts(labels, s) for s in EVAL_SPLITS},
        },
        "items": {
            "file": "items.json",
            "sha256": sha256_file(frozen / "items.json"),
            "total_units": len(items),
            "by_split": dict(unit_splits),
            "candidates_by_split": dict(cand_by_split),
        },
        "build_json": build,
        "evaluation_set": {
            "splits": list(EVAL_SPLITS),
            "units": len(eval_units),
            "candidates_sent": sum(len(u["candidates"]) for u in eval_units),
            "labeled_boxes": sum(1 for k in eval_cand_keys if k in all_label_keys),
            "labeled_units": sum(1 for u in eval_units if any(c["key"] in all_label_keys for c in u["candidates"])),
            "candidates_without_label_in_evaluated_units": sum(
                1 for u in eval_units for c in u["candidates"] if c["key"] not in all_label_keys
            ),
            "max_candidates_per_unit": max(len(u["candidates"]) for u in eval_units),
            "routing_pure_single_class_units": {f"{s}/{c}": n for (s, c), n in sorted(routed_pure.items())},
            "routing_mixed_units": routed_mixed,
            "pure_building_units": sum(n for (s, c), n in routed_pure.items() if c == "违建"),
            "pure_crack_units": sum(n for (s, c), n in routed_pure.items() if c == "道路裂纹"),
        },
        "unlabeled_pool": {
            "units": unit_splits.get("unlabeled", 0),
            "candidates": cand_by_split.get("unlabeled", 0),
            "note": "本轮 baseline 按任务口径只在有标签的 dev+holdout 上评测；无标签池未调用，未计分。",
        },
        "integrity_checks": {
            "label_keys_subset_of_candidate_keys": all_label_keys <= all_cand_keys,
            "duplicate_candidate_keys": len(all_cand_keys) - sum(len(u["candidates"]) for u in items),
            "files": files,
            "release_manifest": manifest_check,
            "all_expected_sha256_ok": all(v["ok"] for v in files.values()),
        },
        "server_facts": SERVER_FACTS,
        "deliverable_code": {
            "dir": "scripts/luna_live_baseline",
            "evaluate_py_sha256": sha256_file(Path(__file__).resolve().parent / "evaluate.py"),
            "server_executed_evaluate_py_sha256": FORMAL_SCOPE["server_executed_evaluate_sha256"],
            "note": "本次正式运行使用 server_executed 版本；交付版修复了跨 resume 的每单元额度。",
        },
        "formal_scope": FORMAL_SCOPE,
        "external_verification": EXTERNAL_VERIFICATION,
        "voyage_mapping": voyage_mapping,
        "cluster_key_note": (
            "冻结 items.json 的 group 与 unit 相同，聚类区间按**图片（原片/帧）**做有放回重采样；"
            "没有航次键时不能用 event/group 冒充航次，报告已明确该口径。"
        ),
        "call_budget": {
            "N_units": len(eval_units),
            "planned_2N": 2 * len(eval_units),
            "max_retry_per_request": 1,
            "cap_4N": 4 * len(eval_units),
            "concurrency": 2,
        },
        "notes": [
            "holdout 443 框不含道路裂纹，且 dev 也无道路裂纹；纯类道路裂纹卡在本轮 baseline 中不会命中。",
            "冻结标签含 3 条人工更正（全部 false→true，全部落在 dev），见 data/luna-relabel-rescore-20261010/。",
            "线上当前提示词版本为 p000-pure-class-cards：仅当整图候选全为同一类且该类有卡（违建/道路裂纹）时插入卡。",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {args.out}")
    print(f"  labels {len(labels)} 框 {dict(label_splits)}")
    print(f"  评测集 {len(eval_units)} 图 / {manifest['evaluation_set']['candidates_sent']} 候选框")
    print(f"  纯违建图 {manifest['evaluation_set']['pure_building_units']}，纯裂纹图 "
          f"{manifest['evaluation_set']['pure_crack_units']}，混合图 {routed_mixed}")
    print(f"  期望哈希全部一致: {manifest['integrity_checks']['all_expected_sha256_ok']}")
    if not manifest["integrity_checks"]["all_expected_sha256_ok"]:
        raise SystemExit("哈希核对失败")


if __name__ == "__main__":
    main()
