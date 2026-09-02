"""Small HTTP adapter for an OpenAI Responses-compatible vision endpoint.

The adapter deliberately accepts only a key supplied by the process
environment (via :class:`Settings`).  In particular, model-profile options
are not used for credentials; this keeps secrets out of the database and
audit payloads.
"""

from __future__ import annotations

import base64
import atexit
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw

from .object_store import ObjectStore
from ..services.auto_annotation_prompts import render_batch_prompt
from ..services.auto_annotation_prompts import ROLE_ORDER


class CodexResponsesError(RuntimeError):
    """Raised when the Responses endpoint returns an unusable result."""


ROLE_INSTRUCTIONS = (
    "You are reviewing existing object-detection candidate boxes in one image. "
    "For every candidate, independently apply these three roles: "
    "visual_inspector checks whether the pixels inside the box are the stated "
    "object; geometry_checker checks whether the box is materially aligned "
    "with that object and not an adjacent object; policy_reviewer checks the "
    "same evidence conservatively, including occlusion and ambiguity. "
    "Each role must return exactly one label: true, false, or uncertain. "
    "Use confidence from 0 to 1. Do not infer a label from another role."
)


def build_three_role_prompt(candidates: list[dict[str, Any]], options: dict[str, Any] | None = None) -> str:
    """Build the single per-sample prompt used by the vision request."""
    # Keep the public helper aligned with the production prompt.  This avoids
    # two subtly different role contracts when callers use the legacy name.
    return render_batch_prompt(candidates, options or {})


def _extract_output_text(body: Any) -> str:
    if not isinstance(body, dict):
        raise CodexResponsesError("Responses response must be a JSON object")
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    pieces: list[str] = []
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, str):
                pieces.append(item)
                continue
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                pieces.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        pieces.append(part)
                    elif isinstance(part, dict) and isinstance(part.get("text"), str):
                        pieces.append(part["text"])

    # A few Responses-compatible gateways still use the Chat Completions
    # envelope. Supporting it costs little and makes migration less brittle.
    choices = body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                pieces.append(message["content"])

    text = "\n".join(piece for piece in pieces if piece.strip()).strip()
    if not text:
        raise CodexResponsesError("Responses response has no output text")
    return text


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some gateways prepend a short sentence despite the JSON-only
        # instruction. Decode the first complete JSON object if present.
        decoder = json.JSONDecoder()
        value = None
        for offset, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[offset:])
                break
            except json.JSONDecodeError:
                continue
        if value is None:
            raise CodexResponsesError("Responses output is not valid JSON")
    if not isinstance(value, dict):
        raise CodexResponsesError("Responses output JSON must be an object")
    return value


def _normalise_decisions(data: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = data.get("decisions")
    if decisions is None:
        decisions = data.get("results") or data.get("items")
    if decisions is None and isinstance(data.get("votes"), list):
        # Also accept a single-candidate response shaped as {votes: [...]}.
        decisions = [{"target_index": 0, "votes": data["votes"], "consensus": data.get("consensus")}]
    if not isinstance(decisions, list) or not decisions:
        raise CodexResponsesError("Responses output has no decisions")

    normalized: list[dict[str, Any]] = []
    for position, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise CodexResponsesError(f"decision {position} is not an object")
        target_index = decision.get("target_index", decision.get("candidate_index", position))
        if isinstance(target_index, bool):
            raise CodexResponsesError(f"decision {position} has invalid target_index")
        try:
            target_index = int(target_index)
        except (TypeError, ValueError) as exc:
            raise CodexResponsesError(f"decision {position} has invalid target_index") from exc
        votes = decision.get("votes")
        if isinstance(votes, dict):
            if set(votes) != set(ROLE_ORDER):
                raise CodexResponsesError(f"decision {position} must have exactly the three role votes")
            vote_values = [votes[role] for role in ROLE_ORDER]
            vote_roles = list(ROLE_ORDER)
        elif isinstance(votes, list):
            vote_values = votes
            vote_roles = []
        else:
            raise CodexResponsesError(f"decision {position} has invalid votes")
        normalized_votes: list[dict[str, Any]] = []
        for vote_position, vote in enumerate(vote_values):
            if not isinstance(vote, dict):
                raise CodexResponsesError(f"decision {position} vote {vote_position} is invalid")
            if vote_roles and set(vote) != {"label", "confidence", "reason"}:
                raise CodexResponsesError(f"decision {position} role vote {vote_position} has invalid fields")
            if not vote_roles:
                supplied_role = vote.get("role")
                expected_role = ROLE_ORDER[vote_position]
                if supplied_role is not None and str(supplied_role) != expected_role:
                    raise CodexResponsesError(f"decision {position} vote {vote_position} has the wrong role")
            label = str(vote.get("label", "")).strip().lower()
            if label not in {"true", "false", "uncertain"}:
                raise CodexResponsesError(f"decision {position} vote {vote_position} has invalid label")
            try:
                confidence = float(vote.get("confidence"))
            except (TypeError, ValueError) as exc:
                raise CodexResponsesError(f"decision {position} vote {vote_position} has invalid confidence") from exc
            if not 0 <= confidence <= 1:
                raise CodexResponsesError(f"decision {position} vote {vote_position} has invalid confidence")
            normalized_votes.append({
                "role": str(vote.get("role") or (vote_roles[vote_position] if vote_roles else ROLE_ORDER[vote_position])),
                "label": label,
                "confidence": confidence,
                "reason": str(vote.get("reason") or ""),
            })
        if len(normalized_votes) != 3:
            raise CodexResponsesError(f"decision {position} must have exactly three votes")
        consensus = str(decision.get("consensus") or "uncertain").strip().lower()
        if consensus not in {"true", "false", "uncertain"}:
            consensus = "uncertain"
        normalized.append({"target_index": target_index, "votes": normalized_votes, "consensus": consensus})
    return normalized


class CodexResponsesVisionAdapter:
    """One image + all candidates in one Responses API request."""

    def __init__(
        self,
        *,
        object_store: ObjectStore,
        api_key: str = "",
        base_url: str = "https://sub.butterfly2sea.us",
        endpoint: str = "/responses",
        model: str = "gpt-5.6-luna",
        reasoning_effort: str = "low",
        timeout_seconds: float = 120,
        verify_tls: bool = True,
        use_codex_cli: bool = False,
        codex_binary: str = "codex",
        actor_authorization: str = "local-image-extension",
    ):
        self.object_store = object_store
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = float(timeout_seconds)
        self.verify_tls = verify_tls
        self.use_codex_cli = use_codex_cli
        self.codex_binary = codex_binary
        self.actor_authorization = actor_authorization
        self._cli_home: Path | None = None

    def _url(self) -> str:
        if self.endpoint.startswith(("http://", "https://")):
            return self.endpoint
        return f"{self.base_url}/{self.endpoint.lstrip('/')}"

    @staticmethod
    def _overlay_bytes(image_bytes: bytes, candidates: list[dict[str, Any]]) -> bytes:
        """Render numbered candidate boxes so the vision model can localize them."""
        try:
            from io import BytesIO

            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise CodexResponsesError("original image cannot be rendered for overlay") from exc
        draw = ImageDraw.Draw(image)
        width, height = image.size
        line_width = max(2, min(width, height) // 300)
        for position, candidate in enumerate(candidates, start=1):
            bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in bbox)
            except (TypeError, ValueError):
                continue
            left = max(0, min(width - 1, round(x1 * width)))
            top = max(0, min(height - 1, round(y1 * height)))
            right = max(left + 1, min(width - 1, round(x2 * width)))
            bottom = max(top + 1, min(height - 1, round(y2 * height)))
            draw.rectangle((left, top, right, bottom), outline=(238, 56, 68), width=line_width)
            label = str(candidate.get("target_index", position - 1))
            text_top = max(0, top - 14)
            draw.rectangle((left, text_top, left + max(16, 7 * len(label) + 6), top), fill=(238, 56, 68))
            draw.text((left + 3, text_top + 1), label, fill=(255, 255, 255))
        output = BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
        return output.getvalue()

    def _ensure_cli_home(self) -> Path:
        """Create an ephemeral Codex home and authenticate once per worker."""
        if self._cli_home is not None:
            return self._cli_home
        home = Path(tempfile.mkdtemp(prefix="review-auto-codex-"))
        atexit.register(shutil.rmtree, home, ignore_errors=True)
        # Codex appends /responses itself. Preserve a custom parent path such
        # as /v1 when callers configure /v1/responses.
        endpoint = self.endpoint
        suffix = "/responses"
        parent_path = endpoint[:-len(suffix)] if endpoint.endswith(suffix) else ""
        base_url = f"{self.base_url.rstrip('/')}{parent_path}"
        config = (
            'model_provider = "review_company"\n'
            f'model = {json.dumps(self.model)}\n'
            f'model_reasoning_effort = {json.dumps(self.reasoning_effort)}\n'
            "disable_response_storage = true\n"
            "[model_providers.review_company]\n"
            'name = "review_company"\n'
            f'base_url = {json.dumps(base_url)}\n'
            'wire_api = "responses"\n'
            "requires_openai_auth = true\n"
            f'http_headers = {{ "x-openai-actor-authorization" = {json.dumps(self.actor_authorization)} }}\n'
        )
        (home / "config.toml").write_text(config, encoding="utf-8")
        key = self.api_key
        try:
            login = subprocess.run(
                [self.codex_binary, "login", "--with-api-key"],
                input=f"{key}\n",
                text=True,
                capture_output=True,
                env={**os.environ, "CODEX_HOME": str(home)},
                timeout=min(30, max(10, self.timeout_seconds)),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            shutil.rmtree(home, ignore_errors=True)
            raise CodexResponsesError("Codex CLI authentication failed") from exc
        if login.returncode != 0:
            shutil.rmtree(home, ignore_errors=True)
            raise CodexResponsesError("Codex CLI authentication failed")
        self._cli_home = home
        return home

    @staticmethod
    def _schema_file(home: Path) -> Path:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["decisions"],
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["target_index", "votes", "consensus"],
                        "properties": {
                            "target_index": {"type": "integer", "minimum": 0},
                            "votes": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["role", "label", "confidence", "reason"],
                                    "properties": {
                                        "role": {"type": "string"},
                                        "label": {"type": "string", "enum": ["true", "false", "uncertain"]},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                        "reason": {"type": "string"},
                                    },
                                },
                            },
                            "consensus": {"type": "string", "enum": ["true", "false", "uncertain"]},
                        },
                    },
                },
            },
        }
        path = home / "auto-annotation.schema.json"
        path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def _parse_cli_output(stdout: str) -> dict[str, Any]:
        messages: list[str] = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                messages.append(item["text"])
        if not messages:
            raise CodexResponsesError("Codex CLI returned no final message")
        return _parse_json_text(messages[-1])

    def _infer_cli(
        self,
        image_bytes: bytes,
        candidates: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        home = self._ensure_cli_home()
        work = Path(tempfile.mkdtemp(prefix="request-", dir=home))
        try:
            original = work / "original.jpg"
            overlay = work / "overlay.jpg"
            original.write_bytes(image_bytes)
            overlay.write_bytes(self._overlay_bytes(image_bytes, candidates))
            schema = self._schema_file(home)
            prompt = render_batch_prompt(candidates, options)
            command = [
                self.codex_binary,
                "exec",
                "--ephemeral",
                "--json",
                "--skip-git-repo-check",
                "--ignore-rules",
                "-C",
                str(work),
                "-s",
                "read-only",
                "-m",
                self.model,
                "-i",
                str(original),
                "-i",
                str(overlay),
                "--output-schema",
                str(schema),
                "-",
            ]
            try:
                result = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    env={**os.environ, "CODEX_HOME": str(home)},
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise CodexResponsesError("Codex CLI vision request failed") from exc
            if result.returncode != 0:
                raise CodexResponsesError("Codex CLI vision request failed")
            return {"decisions": _normalise_decisions(self._parse_cli_output(result.stdout))}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def infer(self, image: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise CodexResponsesError("AUTO_ANNOTATION_API_KEY 未配置")
        image_bytes = self.object_store.get_bytes(image["bucket"], image["object_key"])
        candidates = options.get("candidates") or []
        if not isinstance(candidates, list):
            raise CodexResponsesError("auto annotation candidates must be a list")
        if self.use_codex_cli:
            return self._infer_cli(image_bytes, candidates, options)

        mime_type = mimetypes.guess_type(str(image.get("object_key") or ""))[0] or "image/jpeg"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = render_batch_prompt(candidates, options)
        content = [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:{mime_type};base64,{encoded}", "detail": "high"},
        ]
        try:
            overlay_encoded = base64.b64encode(self._overlay_bytes(image_bytes, candidates)).decode("ascii")
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{overlay_encoded}", "detail": "high"})
        except CodexResponsesError:
            # Keep the raw HTTP fallback usable for non-image test doubles or
            # gateways that receive a source format PIL cannot decode.
            pass
        payload = {
            "model": self.model,
            "input": [{
                "role": "user",
                "content": content,
            }],
            "reasoning": {"effort": self.reasoning_effort},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                trust_env=False,
                verify=self.verify_tls,
            ) as client:
                response = client.post(self._url(), json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise CodexResponsesError(f"Codex Responses request failed: {exc}") from exc
        except (ValueError, TypeError) as exc:
            raise CodexResponsesError("Codex Responses response is not JSON") from exc
        if isinstance(body, dict) and body.get("error"):
            error = body["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise CodexResponsesError(f"Codex Responses returned error: {message}")
        # A gateway may return decisions directly, without output text.
        data = body if isinstance(body, dict) and (body.get("decisions") or body.get("votes")) else _parse_json_text(_extract_output_text(body))
        return {"decisions": _normalise_decisions(data)}

    def review(self, image: dict[str, Any], candidates: list[dict[str, Any]], options: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(options or {})
        merged["candidates"] = candidates
        return self.infer(image, merged)


# Names kept intentionally discoverable for callers that use the shorter
# adapter terminology.
CodexResponsesAdapter = CodexResponsesVisionAdapter
CodexVisionAdapter = CodexResponsesVisionAdapter


def build_codex_vision_adapter(settings, object_store: ObjectStore) -> CodexResponsesVisionAdapter:
    return CodexResponsesVisionAdapter(
        object_store=object_store,
        api_key=settings.auto_annotation_api_key,
        base_url=settings.auto_annotation_base_url,
        endpoint=settings.auto_annotation_endpoint,
        model=settings.auto_annotation_model,
        reasoning_effort=settings.auto_annotation_reasoning_effort,
        timeout_seconds=settings.auto_annotation_timeout_seconds,
        verify_tls=settings.auto_annotation_verify_tls,
        use_codex_cli=settings.auto_annotation_use_codex_cli,
        codex_binary=settings.auto_annotation_codex_binary,
        actor_authorization=settings.auto_annotation_actor_authorization,
    )
