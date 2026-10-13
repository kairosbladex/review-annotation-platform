"""Run a real detection-then-review contract check without printing secrets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="image file used for the non-destructive contract check")
    parser.add_argument("--env-file", type=Path, help="dotenv file to load before connecting")
    parser.add_argument("--rule-code", default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if args.env_file:
        _load_env_file(args.env_file)

    base_url = os.getenv("MODEL_SERVICE_BASE_URL", "").rstrip("/")
    api_key = os.getenv("MODEL_SERVICE_API_KEY", "")
    detect_endpoint = os.getenv("MODEL_SERVICE_DETECT_ENDPOINT", "/api/v1/review-agent/yolo-detect")
    review_endpoint = os.getenv("MODEL_SERVICE_REVIEW_ENDPOINT", "/api/v1/review-agent/review")
    if not base_url or not api_key:
        print("model service config invalid: MODEL_SERVICE_BASE_URL and MODEL_SERVICE_API_KEY are required")
        return 1
    if not args.image.is_file():
        print(f"image does not exist: {args.image}")
        return 1

    verify_tls = os.getenv("MODEL_SERVICE_VERIFY_TLS", "true").lower() in {"1", "true", "yes", "on"}
    headers = {"X-API-Key": api_key}
    image_bytes = args.image.read_bytes()
    files = {"image": (args.image.name, image_bytes, "application/octet-stream")}
    with httpx.Client(timeout=120, follow_redirects=True, trust_env=False, verify=verify_tls) as client:
        detection = client.post(f"{base_url}{detect_endpoint}", files=files, params={"conf": args.confidence}, headers=headers)
        detection.raise_for_status()
        detection_body = detection.json()
        if detection_body.get("success") is False:
            print("model detection failed")
            return 1
        detections = detection_body.get("detections") or detection_body.get("predictions") or []
        review_data = {
            "detections": json.dumps(detections, ensure_ascii=False),
            "yolo_conf_threshold": str(args.confidence),
        }
        if args.rule_code:
            review_data["rule_code"] = args.rule_code
        review = client.post(f"{base_url}{review_endpoint}", files=files, data=review_data, headers=headers)
        review.raise_for_status()
        review_body = review.json()
        if review_body.get("success") is False:
            print("model review failed")
            return 1

    results = review_body.get("results") or []
    uncertain = sum(1 for item in results if item.get("rawDecision") == "uncertain" or item.get("decision") == "uncertain")
    print(json.dumps({"detection_count": len(detections), "review_count": len(results), "uncertain_count": uncertain, "preview_present": bool(review_body.get("image_url") or review_body.get("preview_url"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
