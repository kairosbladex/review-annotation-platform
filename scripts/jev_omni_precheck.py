#!/usr/bin/env python3
"""Call the isolated Jev-Omni service with the same double-view payload as the backend."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import httpx

from app.adapters.jev_omni_classifier import (
    build_jev_omni_composite_image,
    build_jev_omni_decision_query,
    build_jev_omni_review_views,
    encode_jpeg,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--base-url", default="http://127.0.0.1:18010")
    args = parser.parse_args()

    context, roi = build_jev_omni_review_views(args.image.read_bytes(), args.bbox)
    composite = encode_jpeg(build_jev_omni_composite_image(context, roi), quality=92)
    query = build_jev_omni_decision_query(
        args.class_name,
        {
            "target_index": args.target_index,
            "class_name": args.class_name,
            "detector_confidence": args.confidence,
        },
    )
    payload = {
        **query,
        "image_base64": base64.b64encode(composite).decode("ascii"),
        "image_media_type": "image/jpeg",
        "modality": "image",
    }
    response = httpx.post(
        f"{args.base_url.rstrip('/')}/v1/decide",
        json=payload,
        timeout=120.0,
        trust_env=False,
    )
    response.raise_for_status()
    body = response.json()
    print(json.dumps(body, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
