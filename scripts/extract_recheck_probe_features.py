#!/usr/bin/env python3
"""复检方案 5：用冻结的 DINOv2 / SigLIP2 图像编码器给候选框裁剪图提取特征。

只读取干净帧和归一化框，不读取人工标签，不调用 Qwen。
每个候选取两种以框中心为中心的正方形裁剪（框最长边的 2 倍、4 倍），
原图与水平翻转两份特征取平均，输出 npz 供本机离线训练探针。
清单每行：{"uid", "image", "bbox":[x1,y1,x2,y2] 归一化, "downscale_width": 可选}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

SERVICE_GPU = "GPU-REDACTED"
FACTORS = (2.0, 4.0)
# 诊断用裁剪：tight=只含目标本身（框 1.25 倍）；ctxmask=框 4 倍但框内涂成中灰，只剩周边环境
VARIANTS = {"tight": (1.25, False), "ctxmask_f4": (4.0, True)}
MIN_SIDE_AT_960 = 24  # 与低清 960 宽画面对应的最小裁剪边长，按画面宽度等比放大


def square_crop(image: Image.Image, bbox, factor: float, mask_box: bool = False) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    side = max(MIN_SIDE_AT_960 * width / 960.0, max(x2 - x1, y2 - y1) * factor)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    side = max(8, int(round(side)))
    array = np.asarray(image)
    pad = side  # 反射填充，保证越界部分不是黑边
    padded = np.pad(array, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    crop = padded[top + pad:top + pad + side, left + pad:left + pad + side].copy()
    if mask_box:
        bx1, by1 = int(np.floor(x1 - left)), int(np.floor(y1 - top))
        bx2, by2 = int(np.ceil(x2 - left)), int(np.ceil(y2 - top))
        crop[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)] = 128
    return Image.fromarray(crop)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dinov2", type=Path, required=True)
    parser.add_argument("--siglip2", type=Path, required=True)
    parser.add_argument("--expect-gpu", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", action="store_true", help="改为只提取诊断用裁剪 tight / ctxmask_f4")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel

    uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], check=True,
                          text=True, capture_output=True).stdout.split()[0]
    if uuid == SERVICE_GPU or uuid != args.expect_gpu:
        raise SystemExit("gpu_preflight_failed")

    device = "cuda:0"
    dino = AutoModel.from_pretrained(args.dinov2, torch_dtype=torch.float16).to(device).eval()
    siglip = AutoModel.from_pretrained(args.siglip2, torch_dtype=torch.float16).to(device).eval()
    # 按模型自带配置取归一化参数，输入尺寸固定：DINOv2 224，SigLIP2 384
    dino_cfg = json.loads((args.dinov2 / "preprocessor_config.json").read_text())
    sig_cfg = json.loads((args.siglip2 / "preprocessor_config.json").read_text())

    def to_tensor(images, size, mean, std):
        arrays = [np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC),
                             dtype=np.float32) / 255.0 for im in images]
        x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
        x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
        return x.to(device, torch.float16)

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    feats: dict[str, list[np.ndarray]] = {}
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        with Image.open(args.image_root / row["image"]) as handle:
            image = handle.convert("RGB")
        if row.get("downscale_width") and image.width > row["downscale_width"]:
            w = int(row["downscale_width"])
            image = image.resize((w, round(image.height * w / image.width)), Image.Resampling.LANCZOS)
        crops = ({tag: square_crop(image, row["bbox"], f, m) for tag, (f, m) in VARIANTS.items()} if args.variants
                 else {f"f{int(f)}": square_crop(image, row["bbox"], f) for f in FACTORS})
        for tag, crop in crops.items():
            pair = [crop, crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]
            with torch.inference_mode():
                d = dino(pixel_values=to_tensor(pair, 224, dino_cfg["image_mean"], dino_cfg["image_std"]))
                hidden = d.last_hidden_state.float()
                cls = hidden[:, 0].mean(0)
                patch = hidden[:, 1:].mean(1).mean(0)
                s = siglip.get_image_features(
                    pixel_values=to_tensor(pair, 384, sig_cfg["image_mean"], sig_cfg["image_std"]))
                s = (s.pooler_output if hasattr(s, "pooler_output") else s).float().mean(0)
            feats.setdefault(f"dino_cls_{tag}", []).append(cls.cpu().numpy())
            feats.setdefault(f"dino_patch_{tag}", []).append(patch.cpu().numpy())
            feats.setdefault(f"siglip_{tag}", []).append(s.cpu().numpy())
        if index % 50 == 0:
            print(f"{index}/{len(rows)} {time.perf_counter() - started:.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, uid=np.array([r["uid"] for r in rows]),
                        **{k: np.stack(v) for k, v in feats.items()})
    meta = {"rows": len(rows), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "factors": VARIANTS if args.variants else FACTORS, "gpu_uuid": uuid, "seconds": round(time.perf_counter() - started, 1),
            "keys": sorted(feats)}
    args.out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
