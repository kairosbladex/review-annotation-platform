from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import mimetypes
from typing import Any

import httpx

from .object_store import ObjectStore
from ..services.auto_annotation_prompts import (
    FAST_DECISION_SCHEMA,
    parse_fast_response,
    render_batch_prompt,
    render_fast_prompt,
)
from ..services.review_views import build_visual_views


DEFAULT_HTTP_LIMITS = httpx.Limits(
    max_connections=64,
    max_keepalive_connections=32,
)
_REAL_HTTPX_CLIENT = httpx.Client


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class HttpModelAdapter:
    """Adapter for the existing review service and compatible HTTP detectors."""

    def __init__(
        self,
        *,
        object_store: ObjectStore,
        default_base_url: str = "",
        default_api_key: str = "",
        default_verify_tls: bool = True,
    ):
        self.object_store = object_store
        self.default_base_url = default_base_url.rstrip("/")
        self.default_api_key = default_api_key
        self.default_verify_tls = default_verify_tls
        self._pooled_clients: dict[tuple[float, bool], httpx.Client] = {}
        self._pooled_async_clients: dict[tuple[float, bool], httpx.AsyncClient] = {}

    def get_pooled_client(self, *, timeout: float = 120.0, verify_tls: bool = True) -> httpx.Client:
        key = (float(timeout), bool(verify_tls))
        client = self._pooled_clients.get(key)
        if client is None or getattr(client, "is_closed", False):
            client = httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                trust_env=False,
                verify=verify_tls,
                limits=DEFAULT_HTTP_LIMITS,
            )
            self._pooled_clients[key] = client
        return client

    def get_pooled_async_client(self, *, timeout: float = 120.0, verify_tls: bool = True) -> httpx.AsyncClient:
        key = (float(timeout), bool(verify_tls))
        client = self._pooled_async_clients.get(key)
        if client is None or getattr(client, "is_closed", False):
            client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                trust_env=False,
                verify=verify_tls,
                limits=DEFAULT_HTTP_LIMITS,
            )
            self._pooled_async_clients[key] = client
        return client

    @contextmanager
    def _client_scope(self, *, timeout: float = 120.0, verify_tls: bool = True):
        if httpx.Client is _REAL_HTTPX_CLIENT:
            yield self.get_pooled_client(timeout=timeout, verify_tls=verify_tls)
        else:
            with httpx.Client(timeout=timeout, trust_env=False, verify=verify_tls) as client:
                yield client

    def close(self) -> None:
        for client in list(self._pooled_clients.values()):
            try:
                client.close()
            except Exception:
                pass
        self._pooled_clients.clear()

    async def aclose(self) -> None:
        for client in list(self._pooled_async_clients.values()):
            try:
                await client.aclose()
            except Exception:
                pass
        self._pooled_async_clients.clear()

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
        verify_tls = _as_bool(profile.get("verify_tls"), self.default_verify_tls)
        with self._client_scope(timeout=timeout, verify_tls=verify_tls) as client:
            response = client.post(
                f"{base_url}{endpoint}",
                files={"image": (image["object_key"].rsplit("/", 1)[-1], image_bytes, "application/octet-stream")},
                params={"conf": options.get("confidence_threshold", 0.25)},
                headers=headers,
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
        artifact = body.get("preview_url") or body.get("result_image_url")
        if isinstance(artifact, str) and artifact.startswith("/"):
            artifact = f"{base_url}{artifact}"
        return {"predictions": normalized, "artifact": artifact}

    def fetch_artifact(
        self,
        artifact: Any,
        options: dict[str, Any] | None = None,
    ) -> tuple[bytes, str] | None:
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
            profile = (options or {}).get("_model_profile", {})
            api_key = str(profile.get("api_key") or self.default_api_key)
            headers = {"X-API-Key": api_key} if api_key else {}
            verify_tls = _as_bool(profile.get("verify_tls"), self.default_verify_tls)
            response = httpx.get(
                artifact,
                timeout=120,
                follow_redirects=True,
                trust_env=False,
                verify=verify_tls,
                headers=headers,
            )
            response.raise_for_status()
            return response.content, response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
        return None


class LegacyReviewAgentAdapter(HttpModelAdapter):
    """Adapter for the existing YOLOE review service's two-stage HTTP API.

    The legacy service exposes detection and review as separate endpoints. A
    full review therefore first obtains candidate boxes from ``yolo-detect``
    and then submits those boxes to ``review``. The response's ``rawDecision``
    is preferred over the product-facing two-state ``decision`` so an
    ``uncertain`` result is never silently converted into a false candidate.
    """

    def __init__(
        self,
        *,
        object_store: ObjectStore,
        default_base_url: str = "",
        default_api_key: str = "",
        default_detect_endpoint: str = "/api/v1/review-agent/yolo-detect",
        default_review_endpoint: str = "/api/v1/review-agent/review",
        default_verify_tls: bool = True,
    ):
        super().__init__(
            object_store=object_store,
            default_base_url=default_base_url,
            default_api_key=default_api_key,
            default_verify_tls=default_verify_tls,
        )
        self.default_detect_endpoint = default_detect_endpoint
        self.default_review_endpoint = default_review_endpoint

    def infer(self, image: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        profile = options.get("_model_profile", {})
        base_url = str(profile.get("base_url") or self.default_base_url).rstrip("/")
        if not base_url:
            raise RuntimeError("model profile 未配置 base_url")

        detect_endpoint = str(
            profile.get("detect_endpoint") or self.default_detect_endpoint
        )
        review_endpoint = str(
            profile.get("review_endpoint") or self.default_review_endpoint
        )
        api_key = str(profile.get("api_key") or self.default_api_key)
        headers = {"X-API-Key": api_key} if api_key else {}
        timeout = float(profile.get("timeout_seconds", 120))
        verify_tls = _as_bool(profile.get("verify_tls"), self.default_verify_tls)
        confidence_threshold = options.get("confidence_threshold", 0.25)
        image_bytes = self.object_store.get_bytes(image["bucket"], image["object_key"])
        filename = image["object_key"].rsplit("/", 1)[-1]
        files = {"image": (filename, image_bytes, "application/octet-stream")}

        with self._client_scope(timeout=timeout, verify_tls=verify_tls) as client:
            detect_response = client.post(
                f"{base_url}{detect_endpoint}",
                files=files,
                params={"conf": confidence_threshold},
                headers=headers,
            )
            detect_response.raise_for_status()
            detect_body = detect_response.json()
            if detect_body.get("success") is False:
                raise RuntimeError(str(detect_body.get("error") or "detection service returned failure"))
            detections = detect_body.get("detections") or detect_body.get("predictions") or []
            if not isinstance(detections, list):
                raise RuntimeError("detection service returned invalid detections")

            review_data = {
                "detections": json.dumps(detections, ensure_ascii=False),
                "yolo_conf_threshold": str(confidence_threshold),
            }
            rule_code = options.get("rule_code") or profile.get("rule_code")
            if rule_code:
                review_data["rule_code"] = str(rule_code)
            rule = options.get("rule") or profile.get("rule")
            if rule:
                review_data["rule"] = json.dumps(rule, ensure_ascii=False) if isinstance(rule, dict) else str(rule)

            review_response = client.post(
                f"{base_url}{review_endpoint}",
                files=files,
                data=review_data,
                headers=headers,
            )
            review_response.raise_for_status()
            review_body = review_response.json()
            if review_body.get("success") is False:
                raise RuntimeError(str(review_body.get("error") or "review service returned failure"))
            results = review_body.get("results") or review_body.get("detections") or []
            if not isinstance(results, list):
                raise RuntimeError("review service returned invalid results")

        normalized = []
        for index, result in enumerate(results):
            if not isinstance(result, dict):
                raise RuntimeError(f"review response candidate {index} is invalid")
            bbox = result.get("original_bbox") or result.get("bbox") or result.get("box")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise RuntimeError(f"review response candidate {index} has invalid bbox")
            source_detection = detections[index] if index < len(detections) and isinstance(detections[index], dict) else {}
            corrected_confidence = result.get("corrected_confidence")
            confidence = corrected_confidence if corrected_confidence is not None else result.get(
                "original_confidence", result.get("confidence", result.get("score"))
            )
            raw_decision = result.get("rawDecision", result.get("raw_decision"))
            if raw_decision is None:
                raw_decision = result.get("decision")
            normalized.append({
                "target_index": index,
                "class_id": result.get("class_id", source_detection.get("class_id")),
                "class_name": result.get("corrected_class") or result.get("corrected_class_name") or result.get(
                    "original_class", result.get("class_name") or source_detection.get("class_name") or "unknown"
                ),
                "confidence": confidence,
                "bbox": [float(value) for value in bbox],
                "raw_decision": raw_decision,
                "yolo_decision": result.get("yolo_decision"),
                "llm_decision": result.get("llm_decision"),
                "reason_code": result.get("reasonCode", result.get("reason_code")),
                "error_code": result.get("errorCode", result.get("error_code")),
                "reasoning": result.get("reasoning", "") or "",
            })

        artifact = review_body.get("image_url") or review_body.get("preview_url") or review_body.get("result_image_url")
        if isinstance(artifact, str) and artifact.startswith("/"):
            artifact = f"{base_url}{artifact}"
        return {"predictions": normalized, "artifact": artifact}


class OpenAiCompatibleModelAdapter(HttpModelAdapter):
    """Adapter for vLLM / OpenAI-compatible multimodal endpoints (e.g. L20 Qwen3-VL with guided_json)."""

    def __init__(
        self,
        *,
        object_store: ObjectStore,
        default_base_url: str = "",
        default_api_key: str = "",
        default_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        default_endpoint: str = "/v1/chat/completions",
        default_verify_tls: bool = True,
        default_mode: str = "deep",
    ):
        super().__init__(
            object_store=object_store,
            default_base_url=default_base_url,
            default_api_key=default_api_key,
            default_verify_tls=default_verify_tls,
        )
        self.default_model = default_model
        self.default_endpoint = default_endpoint
        self.default_mode = default_mode

    def _call_chat_completions(
        self,
        *,
        url: str,
        headers: dict[str, str],
        model: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        candidates: list[dict[str, Any]],
        guided_schema: dict[str, Any] | None,
        timeout: float,
        verify_tls: bool,
    ) -> dict[str, Any]:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
        ]
        for _view_name, view_bytes in build_visual_views(image_bytes, candidates).items():
            encoded_view = base64.b64encode(view_bytes).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_view}"},
                }
            )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
        }
        if guided_schema is not None:
            payload["extra_body"] = {"guided_json": guided_schema}
        with self._client_scope(timeout=timeout, verify_tls=verify_tls) as client:
            response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""
            return json.loads(text) if isinstance(text, str) and text.strip() else body
        return body

    def infer(self, image: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        profile = options.get("_model_profile", {})
        base_url = str(profile.get("base_url") or self.default_base_url).rstrip("/")
        if not base_url:
            raise RuntimeError("model profile 未配置 base_url")
        endpoint = str(profile.get("endpoint") or self.default_endpoint)
        url = endpoint if endpoint.startswith(("http://", "https://")) else f"{base_url}/{endpoint.lstrip('/')}"
        model = str(profile.get("model") or options.get("model") or self.default_model)
        mode = str(options.get("mode") or profile.get("mode") or self.default_mode).strip().lower()
        api_key = str(profile.get("api_key") or self.default_api_key)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        timeout = float(profile.get("timeout_seconds", 120))
        verify_tls = _as_bool(profile.get("verify_tls"), self.default_verify_tls)
        image_bytes = self.object_store.get_bytes(image["bucket"], image["object_key"])
        mime_type = mimetypes.guess_type(str(image.get("object_key") or ""))[0] or "image/jpeg"
        candidates = list(options.get("candidates") or [])

        if mode in {"fast", "adaptive"}:
            fast_prompt = render_fast_prompt(candidates, options)
            raw_fast = self._call_chat_completions(
                url=url,
                headers=headers,
                model=model,
                prompt=fast_prompt,
                image_bytes=image_bytes,
                mime_type=mime_type,
                candidates=candidates,
                guided_schema=FAST_DECISION_SCHEMA,
                timeout=timeout,
                verify_tls=verify_tls,
            )
            fast_parsed = parse_fast_response(raw_fast)
            if mode == "fast":
                return fast_parsed
            needs_fallback = any(
                d.get("label") == "uncertain" or float(d.get("confidence", 0.0)) < 0.75
                for d in fast_parsed.get("decisions", [])
            )
            if not needs_fallback:
                return fast_parsed

        deep_prompt = render_batch_prompt(candidates, options)
        return self._call_chat_completions(
            url=url,
            headers=headers,
            model=model,
            prompt=deep_prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            candidates=candidates,
            guided_schema=options.get("guided_json"),
            timeout=timeout,
            verify_tls=verify_tls,
        )


class ModelAdapterRouter:
    """Select an adapter from the model profile without coupling the worker to it."""

    def __init__(
        self,
        *,
        object_store: ObjectStore,
        default_base_url: str = "",
        default_api_key: str = "",
        default_adapter_type: str = "http",
        default_detect_endpoint: str = "/api/v1/review-agent/yolo-detect",
        default_review_endpoint: str = "/api/v1/review-agent/review",
        default_verify_tls: bool = True,
    ):
        self.default_adapter_type = default_adapter_type
        openai_compat = OpenAiCompatibleModelAdapter(
            object_store=object_store,
            default_base_url=default_base_url,
            default_api_key=default_api_key,
            default_verify_tls=default_verify_tls,
        )
        self.adapters = {
            "http": HttpModelAdapter(
                object_store=object_store,
                default_base_url=default_base_url,
                default_api_key=default_api_key,
                default_verify_tls=default_verify_tls,
            ),
            "legacy_review_agent": LegacyReviewAgentAdapter(
                object_store=object_store,
                default_base_url=default_base_url,
                default_api_key=default_api_key,
                default_detect_endpoint=default_detect_endpoint,
                default_review_endpoint=default_review_endpoint,
                default_verify_tls=default_verify_tls,
            ),
            "openai_compatible": openai_compat,
            "vllm": openai_compat,
        }

    def _adapter(self, options: dict[str, Any]):
        profile = options.get("_model_profile", {})
        adapter_type = str(profile.get("adapter_type") or self.default_adapter_type)
        adapter = self.adapters.get(adapter_type)
        if adapter is None:
            raise RuntimeError(f"不支持的 model adapter: {adapter_type}")
        return adapter

    def infer(self, image: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        return self._adapter(options).infer(image, options)

    def fetch_artifact(
        self,
        artifact: Any,
        options: dict[str, Any] | None = None,
    ) -> tuple[bytes, str] | None:
        adapter = self._adapter(options or {})
        return adapter.fetch_artifact(artifact, options)


def build_model_adapter(settings, object_store: ObjectStore) -> ModelAdapterRouter:
    return ModelAdapterRouter(
        object_store=object_store,
        default_base_url=settings.model_service_base_url,
        default_api_key=settings.model_service_api_key,
        default_adapter_type=settings.model_adapter_type,
        default_detect_endpoint=settings.model_service_detect_endpoint,
        default_review_endpoint=settings.model_service_review_endpoint,
        default_verify_tls=settings.model_service_verify_tls,
    )
