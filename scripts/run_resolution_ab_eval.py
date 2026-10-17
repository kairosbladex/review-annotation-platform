#!/usr/bin/env python3
"""在隔离容器内对比复检四视图判定在低清/高清干净帧输入下的输出。

只运行 v3 同版判定程序的四视图二分类（基础模型 + 微调模型），不读取人工标签。
两组输入只差干净帧：系统截图与归一化框相同。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

SERVICE_GPU = "GPU-REDACTED"
EXPECTED_VERIFIER_SHA256 = "057af21f30a6e7ebf9d9a8061eb1672fe21f5d2b8eb3f0cc546920e1a9de72af"
BASE_SHA256 = "df34ae794600cbb39478018b79d68734752429ce55ad42d339287aa25a0e132c"
ADAPTER_SHA256 = "edd5999b0bb89381c0034713f3709562cacb35daeea1986fc6b8e091b7b49560"
MIN_FREE_MIB = 16 * 1024


def nvidia() -> tuple[int, int, str, str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,uuid,name",
                          "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True).stdout
    used, free, uuid, name = [part.strip() for part in out.splitlines()[0].split(",")]
    return int(used), int(free), uuid, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", default=ADAPTER_SHA256, help="adapter_model.safetensors 的 SHA-256")
    parser.add_argument("--models", default="base,adapter", help="逗号分隔，只跑其中几路（比较新微调模型时可只跑 adapter）")
    parser.add_argument("--expect-gpu", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--allow-service-gpu", default="",
                        help="用户明确授权占用共享服务 GPU 时填写授权说明；为空则禁止")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    used, free, uuid, name = nvidia()
    pre = {"gpu_uuid": uuid, "gpu_name": name, "memory_used_mib": used, "memory_free_mib": free,
           "expect_gpu": args.expect_gpu, "service_gpu_forbidden": SERVICE_GPU,
           "service_gpu_authorization": args.allow_service_gpu or None}
    service_blocked = uuid == SERVICE_GPU and not args.allow_service_gpu
    if service_blocked or uuid != args.expect_gpu or "L20" not in name.upper() or free < MIN_FREE_MIB:
        pre["stopped"] = "gpu_preflight_failed"
    (args.out / "preflight.json").write_text(json.dumps(pre, ensure_ascii=False, indent=2) + "\n")
    if pre.get("stopped"):
        raise SystemExit(pre["stopped"])

    digest = hashlib.sha256(args.verifier.read_bytes()).hexdigest()
    if digest != EXPECTED_VERIFIER_SHA256:
        raise SystemExit(f"verifier_sha256_mismatch:{digest}")
    spec = importlib.util.spec_from_file_location("ab_verifier", args.verifier)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    engine = module.CleanSourceQwenVerifier(base_model_path=args.base_model, adapter_path=args.adapter,
                                            device="cuda:0", base_model_sha256=BASE_SHA256,
                                            adapter_sha256=args.adapter_sha256)
    engine._load()
    pre.update(adapter=str(args.adapter), adapter_sha256=args.adapter_sha256, models=args.models)
    (args.out / "preflight.json").write_text(json.dumps(pre, ensure_ascii=False, indent=2) + "\n")

    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows
            if int(hashlib.sha256(r["event_id"].encode()).hexdigest(), 16) % args.shard_count == args.shard_index]
    output = args.out / "predictions.jsonl"
    done = set()
    if output.exists():
        done = {(json.loads(l)["candidate_id"], json.loads(l)["arm"]) for l in output.read_text().splitlines() if l.strip()}

    for index, row in enumerate(rows, start=1):
        event = row["event_id"]
        with Image.open(args.images / "annotated" / f"{event}.png") as handle:
            annotated = handle.convert("RGB")
        for arm, clean_path in (("low", args.images / "low_clean" / f"{event}.png"),
                                ("high", args.images / "high_clean" / f"{event}.jpg")):
            if (row["candidate_id"], arm) in done:
                continue
            with Image.open(clean_path) as handle:
                clean = handle.convert("RGB")
            views = module.build_four_views(annotated, clean, row["bbox"], row["bbox"])
            record = {"candidate_id": row["candidate_id"], "event_id": event, "arm": arm,
                      "clean_size": list(clean.size)}
            try:
                for name, adapter in (("base", False), ("adapter", True)):
                    if name not in args.models.split(","):
                        continue
                    started = time.perf_counter()
                    raw, error, decision = "", None, None
                    try:
                        # 与筛选脚本一致：每路调用各自准备输入
                        inputs = engine._prepare_inputs(views, module.build_prompt(row["class_name"]))
                        raw = engine._generate(inputs, adapter_enabled=adapter, max_new_tokens=128)
                        decision = module.parse_strict_binary_json(raw)
                    except Exception as exc:  # 记录后继续，结果中按解析失败统计
                        error = type(exc).__name__ + ":" + str(exc).split("\n", 1)[0][:200]
                    record[f"{name}_raw"] = raw
                    record[f"{name}_decision"] = decision
                    record[f"{name}_error"] = error
                    record[f"{name}_seconds"] = round(time.perf_counter() - started, 3)
            finally:
                for view in views:
                    view.close()
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"{index}/{len(rows)} {row['candidate_id']}", flush=True)
    used, free, uuid, _ = nvidia()
    (args.out / "done.json").write_text(json.dumps({"rows": len(rows), "gpu_uuid": uuid, "memory_used_mib": used}) + "\n")


if __name__ == "__main__":
    main()
