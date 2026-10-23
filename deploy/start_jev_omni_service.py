#!/usr/bin/env python3
"""Jev-Omni 推理微服务（轻量 FastAPI/Uvicorn 包装器）。

主后端通过 ``POST /v1/decide`` 把「双视角合图 + 决策 query」发到本服务，由本服务
调用 ``akhilaaa3/Jev-Omni`` 返回校准后的 ``P("Yes")``。这样 12B 权重只占 GPU 宿主机，
API 进程保持轻量，且可以按卡型分别部署（L20 BF16 / T4 4-bit）。

接口：

* ``GET  /healthz``     —— 存活与权重加载状态（不触发加载）
* ``POST /v1/decide``   —— 单图 Yes/No 校准概率
* ``POST /v1/warmup``   —— 显式预加载权重（可选，避免首个请求超时）

环境变量：

* ``JEV_OMNI_MODEL_ID``      默认 ``akhilaaa3/Jev-Omni``
* ``JEV_OMNI_MODEL_PATH``    本地权重目录（优先于 snapshot_download）
* ``JEV_OMNI_DEVICE``        ``cuda`` / ``cuda:1`` / ``cpu``
* ``JEV_OMNI_DTYPE``         ``bfloat16`` / ``float16``
* ``JEV_OMNI_QUANTIZATION``  ``4bit`` / ``8bit`` / 空（L20 BF16 留空，T4 用 4bit）
* ``JEV_OMNI_PRELOAD``       ``true`` 时进程启动即加载权重
* ``JEV_OMNI_HOST``/``JEV_OMNI_PORT`` 默认 ``0.0.0.0`` / ``8010``

本文件在无 GPU、未安装 ``jev_omni`` 的机器上依然可以正常导入（延迟加载）。
"""

from __future__ import annotations

import base64
import binascii
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.adapters.jev_omni_classifier import normalise_binary_probabilities  # noqa: E402

DEFAULT_MODEL_ID = os.getenv("JEV_OMNI_MODEL_ID", "akhilaaa3/Jev-Omni")
DEFAULT_OPTIONS = ("Yes", "No")
MAX_IMAGE_BYTES = int(os.getenv("JEV_OMNI_MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))

app = FastAPI(title="Jev-Omni Tier 2.5 Fast Gate", version="1.0.0")
_inference_lock = threading.Lock()

_state: dict[str, Any] = {
    "predictor": None,
    "loaded_at": None,
    "load_error": None,
    "model_id": DEFAULT_MODEL_ID,
    "device": os.getenv("JEV_OMNI_DEVICE", ""),
    "dtype": os.getenv("JEV_OMNI_DTYPE", ""),
    "quantization": os.getenv("JEV_OMNI_QUANTIZATION", ""),
}


class DecideRequest(BaseModel):
    state: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: list[str] = Field(default_factory=lambda: list(DEFAULT_OPTIONS))
    image_base64: str = Field(min_length=1)
    image_media_type: str = "image/jpeg"
    modality: str = "image"
    model_id: str | None = None


def _loader_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    device = _state["device"]
    dtype = _state["dtype"]
    quantization = str(_state["quantization"] or "").strip().lower()
    if device:
        kwargs["device"] = device
    if dtype:
        kwargs["dtype"] = dtype
    if quantization in {"4bit", "4-bit", "int4"}:
        kwargs["load_in_4bit"] = True
    elif quantization in {"8bit", "8-bit", "int8"}:
        kwargs["load_in_8bit"] = True
    return kwargs


def _load_from_local_dir(model_dir: Path, *, device: str, dtype: str) -> Any:
    """Assemble the public JevOmni classifier from a local checkpoint directory.

    The upstream ``load_jev_omni`` helper only accepts a HuggingFace repo id and
    would call ``snapshot_download`` even when a verified local checkpoint is
    already present.  Keep the same assembly steps here so the service can run
    fully offline from ``JEV_OMNI_MODEL_PATH``.
    """
    import json

    import torch
    import transformers
    from transformers import AutoProcessor

    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from jev_omni import JevOmni, _find_backbone, _Head256  # type: ignore[import-not-found]

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    config = transformers.AutoConfig.from_pretrained(model_dir)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_dir, dtype=torch_dtype, device_map=device
    ).eval()
    decision = json.loads(
        (model_dir / "decision_config.json").read_text(encoding="utf-8")
    )
    head = _Head256(decision["hidden_size"]).to(device).eval()
    head.load_state_dict(
        torch.load(model_dir / "head.pt", map_location=device, weights_only=True)
    )
    _, decoder = _find_backbone(model)
    processor = AutoProcessor.from_pretrained(model_dir)
    return JevOmni(model, head, processor, decoder, device)


def load_predictor() -> Any:
    """加载 12B 权重（幂等）。失败时把原因留在 ``_state['load_error']``。"""
    if _state["predictor"] is not None:
        return _state["predictor"]
    model_path = os.getenv("JEV_OMNI_MODEL_PATH", "").strip()
    if not model_path:
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

            model_path = snapshot_download(repo_id=_state["model_id"])
        except Exception as exc:  # noqa: BLE001
            _state["load_error"] = f"snapshot_download failed: {exc}"
            raise RuntimeError(_state["load_error"]) from exc

    # The checkpoint ships its own ``jev_omni.py``; make it importable before
    # resolving the loader, otherwise a local-only deployment fails at import.
    if Path(model_path).is_dir() and str(Path(model_path)) not in sys.path:
        sys.path.insert(0, str(Path(model_path)))
    try:
        import jev_omni  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        _state["load_error"] = f"import jev_omni failed: {exc}"
        raise RuntimeError(_state["load_error"]) from exc

    try:
        local_path = Path(model_path)
        if local_path.is_dir() and (local_path / "config.json").is_file():
            predictor = _load_from_local_dir(
                local_path,
                device=_state["device"] or "cuda",
                dtype=_state["dtype"] or "bfloat16",
            )
        else:
            loader = getattr(jev_omni, "load_jev_omni", None)
            if loader is None:
                _state["load_error"] = "jev_omni.load_jev_omni missing"
                raise RuntimeError(_state["load_error"])
            try:
                predictor = loader(model_path, **_loader_kwargs())
            except TypeError:
                # 兼容不接受 device/dtype/quantization 关键字的旧版加载器。
                predictor = loader(model_path)
    except Exception as exc:  # noqa: BLE001
        _state["load_error"] = str(exc)
        raise
    _state["predictor"] = predictor
    _state["loaded_at"] = time.time()
    _state["load_error"] = None
    return predictor


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_id": _state["model_id"],
        "loaded": _state["predictor"] is not None,
        "loaded_at": _state["loaded_at"],
        "load_error": _state["load_error"],
        "device": _state["device"],
        "dtype": _state["dtype"],
        "quantization": _state["quantization"],
    }


@app.post("/v1/warmup")
def warmup() -> dict[str, Any]:
    try:
        load_predictor()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Jev-Omni 权重加载失败：{exc}") from exc
    return {"status": "ready", "model_id": _state["model_id"]}


@app.post("/v1/decide")
def decide(payload: DecideRequest) -> dict[str, Any]:
    """单图 Yes/No 决策：返回校准后的 ``p_true`` / ``p_false``。"""
    options = tuple(payload.options or DEFAULT_OPTIONS)
    if len(options) != 2:
        raise HTTPException(status_code=400, detail="options 必须恰好两个（Yes/No）")

    try:
        image_bytes = base64.b64decode(payload.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_base64 不是合法 base64") from exc
    if not image_bytes:
        raise HTTPException(status_code=400, detail="图像为空")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图像超过大小上限")

    try:
        predictor = load_predictor()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Jev-Omni 权重加载失败：{exc}") from exc

    suffix = ".png" if "png" in payload.image_media_type else ".jpg"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = Path(handle.name)
    started = time.perf_counter()
    try:
        handle.write(image_bytes)
        handle.close()
        # JevOmni keeps a per-instance capture buffer; concurrent calls would
        # overwrite it.  Serialize GPU inference while keeping HTTP intake
        # concurrent so the service reports honest queueing latency.
        with _inference_lock:
            output = predictor.predict(
                state=payload.state,
                question=payload.question,
                options=list(options),
                media=str(path),
                modality=payload.modality or "image",
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Jev-Omni 推理失败：{exc}") from exc
    finally:
        try:
            handle.close()
        finally:
            path.unlink(missing_ok=True)

    try:
        p_true, p_false, raw_probs = normalise_binary_probabilities(output, options)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Jev-Omni 输出不可解析：{exc}") from exc

    return {
        "p_true": p_true,
        "p_false": p_false,
        "raw_probs": raw_probs,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "model_id": payload.model_id or _state["model_id"],
    }


def main() -> int:
    import uvicorn

    if str(os.getenv("JEV_OMNI_PRELOAD", "false")).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            load_predictor()
            print(f"[jev-omni] preloaded {_state['model_id']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            # 预加载失败不退出进程：/healthz 会暴露 load_error，便于运维排查。
            print(f"[jev-omni] preload failed: {exc}", flush=True)

    uvicorn.run(
        app,
        host=os.getenv("JEV_OMNI_HOST", "0.0.0.0"),
        port=int(os.getenv("JEV_OMNI_PORT", "8010")),
        log_level=os.getenv("JEV_OMNI_LOG_LEVEL", "info"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
