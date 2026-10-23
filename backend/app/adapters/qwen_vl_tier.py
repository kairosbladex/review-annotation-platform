"""Tier-3 adapter for the real Qwen3-VL review service.

The standalone vision workbench uses the legacy YOLOE review service for
candidate verification.  Auto-annotation previously always used the Codex
Responses adapter, which made the Tier-3 branch depend on a separate Luna
credential instead of the deployed Qwen3-VL service.  This adapter passes the
already-known candidates to the review endpoint so the same Qwen3-VL model is
used without re-running detection.
"""

from __future__ import annotations

import json
import mimetypes
from typing import Any

import httpx

from ..services.auto_annotation_prompts import ROLE_ORDER


class QwenVLTierAdapter:
    def __init__(
        self,
        *,
        object_store: Any,
        base_url: str,
        api_key: str = "",
        endpoint: str = "/api/v1/review-agent/review",
        timeout_seconds: float = 120.0,
        verify_tls: bool = True,
    ) -> None:
        self.object_store = object_store
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key or "")
        self.endpoint = endpoint
        self.timeout_seconds = float(timeout_seconds)
        self.verify_tls = bool(verify_tls)

    def review(
        self,
        image: dict[str, Any],
        candidates: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("MODEL_SERVICE_BASE_URL 未配置")
        options = options or {}
        image_bytes = self.object_store.get_bytes(image["bucket"], image["object_key"])
        filename = str(image.get("object_key") or "image").rsplit("/", 1)[-1]
        mime_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
        detections = [
            {
                "target_index": candidate.get("target_index", index),
                "class_id": candidate.get("class_id") if candidate.get("class_id") is not None else 0,
                "class_name": candidate.get("class_name", "unknown"),
                "confidence": candidate.get("confidence", 1.0),
                "bbox": candidate.get("bbox", [0.0, 0.0, 1.0, 1.0]),
            }
            for index, candidate in enumerate(candidates)
        ]
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        data = {
            "detections": json.dumps(detections, ensure_ascii=False),
            "yolo_conf_threshold": str(options.get("tier2_threshold", 0.12)),
        }
        if options.get("rule_code"):
            data["rule_code"] = str(options["rule_code"])
        with httpx.Client(timeout=self.timeout_seconds, verify=self.verify_tls) as client:
            response = client.post(
                f"{self.base_url}{self.endpoint}",
                files={
                    "image": (filename, image_bytes, mime_type),
                    # The clean-source gate is opt-in on the legacy service.
                    # Use the same source frame for the four-view Qwen3-VL
                    # verifier when no separately cleaned frame is available.
                    "clean_image": (f"clean-{filename}", image_bytes, mime_type),
                },
                data=data,
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()
        if body.get("success") is False:
            raise RuntimeError(str(body.get("error") or "Qwen3-VL review returned failure"))
        results = body.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Qwen3-VL review returned invalid results")

        decisions: list[dict[str, Any]] = []
        for index, result in enumerate(results):
            if not isinstance(result, dict):
                raise RuntimeError(f"Qwen3-VL result {index} is invalid")
            candidate = candidates[index] if index < len(candidates) else {}
            decision = str(result.get("decision") or "uncertain").strip().lower()
            label = {"confirmed": "true", "rejected": "false"}.get(decision, "uncertain")
            confidence = (
                result.get("modelProbability")
                or result.get("corrected_confidence")
                or result.get("original_confidence")
                or candidate.get("confidence")
                or 0.0
            )
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            reason = str(result.get("reasonCode") or result.get("reasoning") or decision)
            votes = [
                {"role": role, "label": label, "confidence": confidence, "reason": reason}
                for role in ROLE_ORDER
            ]
            decisions.append(
                {
                    "target_index": candidate.get("target_index", index),
                    "votes": votes,
                    "consensus": label,
                    "scene": None,
                    "evidence": {"qwen3_vl": result},
                }
            )
        return {"decisions": decisions, "qwen3_vl_session_id": body.get("session_id")}


def build_qwen_vl_tier_adapter(settings: Any, object_store: Any) -> QwenVLTierAdapter:
    return QwenVLTierAdapter(
        object_store=object_store,
        base_url=settings.model_service_base_url,
        api_key=settings.model_service_api_key,
        endpoint=settings.model_service_review_endpoint,
        timeout_seconds=settings.auto_annotation_timeout_seconds,
        verify_tls=settings.model_service_verify_tls,
    )


def build_auto_annotation_adapter(settings: Any, object_store: Any) -> Any:
    """Select Tier 3 consistently for the API and Celery worker."""
    if settings.model_adapter_type == "legacy_review_agent" and settings.model_service_base_url:
        return build_qwen_vl_tier_adapter(settings, object_store)
    from .codex_responses import build_codex_vision_adapter

    return build_codex_vision_adapter(settings, object_store)
