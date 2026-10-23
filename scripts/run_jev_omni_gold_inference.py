#!/usr/bin/env python3
"""在 GPU 宿主机上对 Jev-Omni 双视角合图逐条推理，输出真实 ``p_true`` / ``p_false``。

输入是 ``scripts/build_jev_omni_gold_bundle.py`` 产出的 ``manifest.jsonl``：
每行已包含合图路径与 ``(state, question, options)``，因此本脚本**不依赖后端代码**，
只需要 ``torch`` + ``transformers`` + ``jev_omni``（模型仓库自带的 ``jev_omni.py``）。

输出 JSONL 每行：

    {"candidate_id": ..., "p_true": 0.93, "p_false": 0.07,
     "latency_ms": 812.4, "prediction": "Yes", "confidence": 0.93,
     "model_id": "akhilaaa3/Jev-Omni", "view_source": ..., "error": null}

单条失败只记 ``error`` 并继续，绝不中断整批；任何一条 ``error`` 非空时进程返回码为 2。

用法（GPU 宿主机）：
    python3 scripts/run_jev_omni_gold_inference.py \
        --manifest /work/bundle/manifest.jsonl \
        --bundle-root /work/bundle \
        --output /work/out/jev_omni_probs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_probabilities(result: Any, options: list[str]) -> tuple[float, float]:
    """把 ``jev_omni`` 的返回值读成 ``(p_true, p_false)``，并强制互补。"""
    mapping: dict[str, float] | None = None
    if isinstance(result, dict):
        for key in ("probabilities", "probs", "probability"):
            value = result.get(key)
            if isinstance(value, dict):
                mapping = {str(k): float(v) for k, v in value.items()}
                break
    if mapping is None:
        confidence = float(result["confidence"])  # type: ignore[index]
        prediction = str(result["prediction"])  # type: ignore[index]
        if prediction == options[0]:
            p_true = confidence
        elif prediction == options[1]:
            p_true = 1.0 - confidence
        else:
            raise ValueError(f"prediction={prediction!r} 不在 options 内")
        return p_true, 1.0 - p_true
    if options[0] in mapping:
        p_true = float(mapping[options[0]])
    else:
        raise ValueError(f"返回概率缺少选项 {options[0]!r}：{sorted(mapping)}")
    p_true = min(1.0, max(0.0, p_true))
    return p_true, 1.0 - p_true


def _load_from_local_dir(model_dir: Path, device: str) -> Any:
    """从已下载好的本地目录装配 ``JevOmni``（等价于 ``load_jev_omni``，但跳过 HF 缓存）。"""
    import json

    import torch
    import transformers
    from jev_omni import JevOmni, _find_backbone, _Head256  # noqa: PLC0415
    from transformers import AutoProcessor

    config = transformers.AutoConfig.from_pretrained(model_dir)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device
    ).eval()
    decision = json.loads((model_dir / "decision_config.json").read_text(encoding="utf-8"))
    head = _Head256(decision["hidden_size"]).to(device).eval()
    head.load_state_dict(
        torch.load(model_dir / "head.pt", map_location=device, weights_only=True)
    )
    _, decoder = _find_backbone(model)
    processor = AutoProcessor.from_pretrained(model_dir)
    return JevOmni(model, head, processor, decoder, device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="akhilaaa3/Jev-Omni")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="已下载的本地权重目录（优先于 --model-id 走 HF 缓存）",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="仅跑前 N 条（0=全部），用于冒烟")
    parser.add_argument("--warmup", type=int, default=3, help="前 N 条不计入统计（预热）")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    import torch

    if args.model_dir is not None:
        sys.path.insert(0, str(args.model_dir.resolve()))
    from jev_omni import load_jev_omni  # noqa: PLC0415 - 依赖只在 GPU 侧存在

    rows = _read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[load] model={args.model_id} device={args.device} samples={len(rows)}", flush=True)
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)}", flush=True)

    started = time.perf_counter()
    if args.model_dir is not None and (args.model_dir / "config.json").exists():
        print(f"[load] local dir {args.model_dir}", flush=True)
        classifier = _load_from_local_dir(args.model_dir, args.device)
    else:
        classifier = load_jev_omni(args.model_id, device=args.device)
    print(f"[load] ready in {time.perf_counter() - started:.1f}s", flush=True)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[load] vram_allocated={allocated:.2f} GiB", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    latencies: list[float] = []
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            media = (args.bundle_root / row["composite"]).resolve()
            options = list(row["options"])
            record: dict[str, Any] = {
                "candidate_id": row["candidate_id"],
                "legacy_candidate_id": row.get("legacy_candidate_id"),
                "class_name": row.get("class_name"),
                "human_label": row.get("human_label"),
                "detector_confidence": row.get("detector_confidence"),
                "view_source": row.get("view_source"),
                "model_id": args.model_id,
                "p_true": None,
                "p_false": None,
                "latency_ms": None,
                "prediction": None,
                "confidence": None,
                "warmup": index < args.warmup,
                "error": None,
            }
            try:
                if not media.exists():
                    raise FileNotFoundError(f"合图缺失：{media}")
                tick = time.perf_counter()
                result = classifier.predict(
                    state=row["state"],
                    question=row["question"],
                    options=options,
                    media=str(media),
                    modality="image",
                )
                elapsed_ms = (time.perf_counter() - tick) * 1000.0
                p_true, p_false = _resolve_probabilities(result, options)
                record.update(
                    {
                        "p_true": round(p_true, 6),
                        "p_false": round(p_false, 6),
                        "latency_ms": round(elapsed_ms, 2),
                        "prediction": result.get("prediction") if isinstance(result, dict) else None,
                        "confidence": round(float(result["confidence"]), 6)
                        if isinstance(result, dict) and "confidence" in result
                        else None,
                    }
                )
                if index >= args.warmup:
                    latencies.append(elapsed_ms)
            except Exception as exc:  # noqa: BLE001 - 单条失败不中断整批
                errors += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[error] {row['candidate_id']}: {record['error']}", flush=True)
                traceback.print_exc()
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % args.log_every == 0 or index + 1 == len(rows):
                mean = sum(latencies) / len(latencies) if latencies else 0.0
                print(
                    f"[progress] {index + 1}/{len(rows)} errors={errors} mean_latency={mean:.1f}ms",
                    flush=True,
                )

    if latencies:
        ordered = sorted(latencies)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        print(
            f"[done] n={len(latencies)} mean={sum(latencies) / len(latencies):.1f}ms "
            f"p50={p50:.1f}ms p95={p95:.1f}ms errors={errors}",
            flush=True,
        )
    print(f"[done] wrote {args.output}", flush=True)
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
