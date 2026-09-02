from __future__ import annotations

import base64
from typing import Any

import httpx

from .object_store import ObjectStore


class HttpModelAdapter:
    """Adapter for the existing review service and compatible HTTP detectors."""

    def __init__(self, *, object_store: ObjectStore, default_base_url: str = "", default_api_key: str = ""):
        self.object_store = object_store
        self.default_base_url = default_base_url.rstrip("/")
        self.default_api_key = default_api_key

    def infer(self, image: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        profile = options.get("_model_profile", {})
        base_url = str(profile.get("base_url") or self.default_base_url).rstrip("/")
        endpoint = str(profile.get("endpoint") or "/api/v1/review-agent/yolo-detect")
        if not base_url:
            raise RuntimeError("model profile 未配置 base_url")
        image_bytes = self.object_store.get_bytes(image["bucket"], image["object_key"])
        headers = {}
        api_key = str(profile.get("api_key") or self.default_api_key)
        if api_key:
            headers["X-API-Key"] = api_key
        timeout = float(profile.get("timeout_seconds", 120))
        response = httpx.post(
            f"{base_url}{endpoint}",
            files={"image": (image["object_key"].rsplit("/", 1)[-1], image_bytes, "application/octet-stream")},
            params={"conf": options.get("confidence_threshold", 0.25)},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("success") is False:
            raise RuntimeError(str(body.get("error") or "model service returned failure"))
        predictions = body.get("predictions") or body.get("detections") or []
        normalized = []
        for index, prediction in enumerate(predictions):
            bbox = prediction.get("bbox") or prediction.get("box")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise RuntimeError(f"model response candidate {index} has invalid bbox")
            normalized.append({
                "target_index": index,
                "class_id": prediction.get("class_id"),
                "class_name": prediction.get("class_name") or prediction.get("class") or "unknown",
                "confidence": prediction.get("confidence", prediction.get("score")),
                "bbox": [float(value) for value in bbox],
                "raw_decision": prediction.get("raw_decision"),
                "yolo_decision": prediction.get("yolo_decision"),
                "llm_decision": prediction.get("llm_decision"),
                "reason_code": prediction.get("reason_code"),
                "error_code": prediction.get("error_code"),
                "reasoning": prediction.get("reasoning", ""),
            })
        return {"predictions": normalized, "artifact": body.get("preview_url") or body.get("result_image_url")}

    def fetch_artifact(self, artifact: Any) -> tuple[bytes, str] | None:
        """Resolve an optional adapter preview into bytes for object storage."""
        if isinstance(artifact, bytes):
            return artifact, "image/jpeg"
        if not isinstance(artifact, str) or not artifact:
            return None
        if artifact.startswith("data:") and "," in artifact:
            header, encoded = artifact.split(",", 1)
            media_type = header[5:].split(";", 1)[0] or "image/jpeg"
            return base64.b64decode(encoded), media_type
        if artifact.startswith(("http://", "https://")):
            response = httpx.get(artifact, timeout=120, follow_redirects=True)
            response.raise_for_status()
            return response.content, response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
        return None
