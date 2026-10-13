#!/usr/bin/env python3
"""在隔离容器内给复检四视图二分类打连续分数（方案 1 离线评测）。

与 run_resolution_ab_eval.py 同一判定程序、同一四视图和提示词，不做生成，
而是用教师强制分别计算两个合法答案的对数似然：
  {"decision":"CONFIRMED"} 与 {"decision":"REJECTED"}
p_confirm = softmax(ll_confirmed, ll_rejected)。基础模型、微调模型各算一次。
不读取人工标签。
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

SERVICE_GPU = "GPU-9f2ac4d0-be45-fe5b-d16a-86d1f3fd10f7"
EXPECTED_VERIFIER_SHA256 = "057af21f30a6e7ebf9d9a8061eb1672fe21f5d2b8eb3f0cc546920e1a9de72af"
BASE_SHA256 = "df34ae794600cbb39478018b79d68734752429ce55ad42d339287aa25a0e132c"
ADAPTER_SHA256 = "edd5999b0bb89381c0034713f3709562cacb35daeea1986fc6b8e091b7b49560"
MIN_FREE_MIB = 16 * 1024
ANSWERS = {"CONFIRMED": '{"decision":"CONFIRMED"}', "REJECTED": '{"decision":"REJECTED"}'}


def nvidia() -> tuple[int, int, str, str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,uuid,name",
                          "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True).stdout
    used, free, uuid, name = [part.strip() for part in out.splitlines()[0].split(",")]
    return int(used), int(free), uuid, name


def answer_loglik(engine, inputs, answer_ids, end_id, *, adapter: bool) -> dict:
    """教师强制：把答案 token 接在生成提示后，求答案（含结束符）逐 token 对数概率之和。"""
    import contextlib
    import torch

    ids = torch.tensor(answer_ids + [end_id], device=inputs["input_ids"].device).unsqueeze(0)
    prompt_len = inputs["input_ids"].shape[1]
    feed = dict(inputs)
    # 逐 token 的字段（input_ids、attention_mask、mm_token_type_ids 等）都要接上答案长度；
    # 答案是纯文本：attention 取 1，其余类型字段取 0。
    for key, value in inputs.items():
        if hasattr(value, "dim") and value.dim() == 2 and value.shape[1] == prompt_len:
            fill = ids if key == "input_ids" else (
                torch.ones_like(ids) if key == "attention_mask" else torch.zeros_like(ids))
            feed[key] = torch.cat([value, fill.to(value.dtype)], dim=1)
    context = contextlib.nullcontext() if adapter else engine._model.disable_adapter()
    with context, torch.inference_mode():
        logits = engine._model(**feed).logits[0, prompt_len - 1:-1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(1, ids[0].unsqueeze(1)).squeeze(1)
    return {"sum": float(logp.sum()), "tokens": [float(v) for v in logp]}


def use_linear_patch_embed(model) -> int:
    """视觉 patch_embed 的 Conv3d 核等于步长，等价于一次矩阵乘；
    该镜像里 Conv3d 单次约 21 秒，改成等价的 F.linear。返回替换的模块数。"""
    import torch.nn.functional as F
    from transformers.models.qwen3_vl import modeling_qwen3_vl as mq

    count = 0
    for module in model.modules():
        if isinstance(module, mq.Qwen3VLVisionPatchEmbed):
            proj = module.proj
            assert tuple(proj.kernel_size) == tuple(proj.stride) and proj.padding in ((0, 0, 0), 0)

            def forward(hidden_states, _proj=proj, _dim=module.embed_dim):
                weight = _proj.weight
                x = hidden_states.to(weight.dtype).reshape(-1, weight[0].numel())
                return F.linear(x, weight.reshape(weight.shape[0], -1), _proj.bias).view(-1, _dim)

            module.forward = forward
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-patch-embed", action="store_true")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expect-gpu", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    used, free, uuid, name = nvidia()
    pre = {"gpu_uuid": uuid, "gpu_name": name, "memory_used_mib": used, "memory_free_mib": free,
           "expect_gpu": args.expect_gpu, "service_gpu_forbidden": SERVICE_GPU}
    if uuid == SERVICE_GPU or uuid != args.expect_gpu or "L20" not in name.upper() or free < MIN_FREE_MIB:
        pre["stopped"] = "gpu_preflight_failed"
    (args.out / "preflight.json").write_text(json.dumps(pre, ensure_ascii=False, indent=2) + "\n")
    if pre.get("stopped"):
        raise SystemExit(pre["stopped"])

    digest = hashlib.sha256(args.verifier.read_bytes()).hexdigest()
    if digest != EXPECTED_VERIFIER_SHA256:
        raise SystemExit(f"verifier_sha256_mismatch:{digest}")
    spec = importlib.util.spec_from_file_location("score_verifier", args.verifier)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    engine = module.CleanSourceQwenVerifier(base_model_path=args.base_model, adapter_path=args.adapter,
                                            device="cuda:0", base_model_sha256=BASE_SHA256,
                                            adapter_sha256=ADAPTER_SHA256)
    engine._load()
    pre["linear_patch_embed_modules"] = use_linear_patch_embed(engine._model) if args.linear_patch_embed else 0
    (args.out / "preflight.json").write_text(json.dumps(pre, ensure_ascii=False, indent=2) + "\n")
    tokenizer = engine._processor.tokenizer
    answer_ids = {k: tokenizer(v, add_special_tokens=False)["input_ids"] for k, v in ANSWERS.items()}
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    (args.out / "tokenization.json").write_text(json.dumps(
        {"answer_ids": answer_ids, "end_id": end_id,
         "answer_tokens": {k: tokenizer.convert_ids_to_tokens(v) for k, v in answer_ids.items()}},
        ensure_ascii=False, indent=2) + "\n")

    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows
            if int(hashlib.sha256(r["event_id"].encode()).hexdigest(), 16) % args.shard_count == args.shard_index]
    output = args.out / "scores.jsonl"
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
            started = time.perf_counter()
            try:
                inputs = engine._prepare_inputs(views, module.build_prompt(row["class_name"]))
                record["prepare_seconds"] = round(time.perf_counter() - started, 3)
                if index == 1 and arm == "low":
                    record["input_keys"] = {k: list(getattr(v, "shape", [])) for k, v in inputs.items()}
                for name, adapter in (("base", False), ("adapter", True)):
                    ll = {k: answer_loglik(engine, inputs, ids, end_id, adapter=adapter)
                          for k, ids in answer_ids.items()}
                    record[f"{name}_ll"] = {k: v["sum"] for k, v in ll.items()}
                    record[f"{name}_token_logp"] = {k: v["tokens"] for k, v in ll.items()}
                record["error"] = None
            except Exception as exc:  # 记录后继续，分析时按缺失处理
                record["error"] = type(exc).__name__ + ":" + str(exc).split("\n", 1)[0][:200]
            finally:
                for view in views:
                    view.close()
            record["seconds"] = round(time.perf_counter() - started, 3)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"{index}/{len(rows)} {row['candidate_id']}", flush=True)
    used, free, uuid, _ = nvidia()
    (args.out / "done.json").write_text(json.dumps({"rows": len(rows), "gpu_uuid": uuid, "memory_used_mib": used}) + "\n")


if __name__ == "__main__":
    main()
