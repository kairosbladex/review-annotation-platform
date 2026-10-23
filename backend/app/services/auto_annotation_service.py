"""Run one-request-per-sample visual consensus annotation jobs."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Annotation, AnnotationTask, Asset, AuditEvent, AutoAnnotationItem, AutoAnnotationRun, Candidate, Job, Sample, new_id, utc_now
from .auto_annotation_prompts import ROLE_ORDER, merge_agent_results, parse_fast_response
from .ktd_recheck import (
    JEV_OMNI_CONFIRM_THRESHOLD,
    JEV_OMNI_REJECT_THRESHOLD,
    TIER1_AUTO_CONFIRM_THRESHOLD,
    TIER2_AUTO_REJECT_THRESHOLD,
    apply_pre_llm_geometric_gate,
    evaluate_cascade_tier,
    evaluate_jev_omni_tier,
)
from .review_policy import apply_review_policy, normalize_scene


AUTO_ANNOTATION_CONFIDENCE_FLOOR = 0.6
_SECRET_KEY_RE = re.compile(r"(?i)(bearer\s+)[^\s,;]+")


def _safe_error(error: Exception | str) -> str:
    """Keep dependency errors useful without accidentally storing a token."""
    return _SECRET_KEY_RE.sub(r"\1[redacted]", str(error))[:2000]


def sanitize_options(value: Any) -> Any:
    """Remove credential-shaped keys before options enter the database/audit."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(secret_word in key_text for secret_word in ("api_key", "apikey", "authorization", "access_token", "secret", "password")):
                continue
            sanitized[str(key)] = sanitize_options(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_options(item) for item in value]
    return value


def build_auto_annotation_options(
    options: dict[str, Any] | None, settings: Any
) -> dict[str, Any]:
    """Merge Settings-derived cascade defaults with caller-supplied options.

    `deploy/production.env.example` documents AUTO_ANNOTATION_MODE,
    TIER1_AUTO_CONFIRM_THRESHOLD, TIER2_AUTO_REJECT_THRESHOLD and
    VLLM_L20_BASE_URL. Before this merge those values were parsed into
    Settings but never threaded into the job payload, so the three-tier
    cascade could not actually be enabled by configuration. Explicit
    caller-supplied options always take precedence.
    """
    defaults: dict[str, Any] = {
        "mode": str(getattr(settings, "auto_annotation_mode", "deep") or "deep"),
        "tier1_threshold": float(
            getattr(settings, "tier1_auto_confirm_threshold", TIER1_AUTO_CONFIRM_THRESHOLD)
        ),
        "tier2_threshold": float(
            getattr(settings, "tier2_auto_reject_threshold", TIER2_AUTO_REJECT_THRESHOLD)
        ),
        "cascade_routing": bool(
            getattr(settings, "auto_annotation_cascade_routing", False)
        ),
        # Tier 2.5 (Jev-Omni Fast Gate).  Defaults are off so a deployment that never
        # sets JEV_OMNI_* keeps the exact 3-tier behaviour.
        "jev_omni_enabled": bool(getattr(settings, "jev_omni_enabled", False)),
        "jev_omni_mode": str(getattr(settings, "jev_omni_mode", "http") or "http").strip().lower(),
        "jev_omni_base_url": str(getattr(settings, "jev_omni_base_url", "") or "").rstrip("/"),
        "jev_omni_model_id": str(
            getattr(settings, "jev_omni_model_id", "akhilaaa3/Jev-Omni") or "akhilaaa3/Jev-Omni"
        ),
        "jev_omni_confirm_threshold": float(
            getattr(settings, "jev_omni_confirm_threshold", JEV_OMNI_CONFIRM_THRESHOLD)
        ),
        "jev_omni_reject_threshold": float(
            getattr(settings, "jev_omni_reject_threshold", JEV_OMNI_REJECT_THRESHOLD)
        ),
        "jev_omni_timeout_seconds": float(
            getattr(settings, "jev_omni_timeout_seconds", 30.0)
        ),
        "jev_omni_verify_tls": bool(getattr(settings, "jev_omni_verify_tls", True)),
    }
    vllm_base = str(getattr(settings, "vllm_l20_base_url", "") or "").strip()
    if vllm_base:
        defaults["vllm_l20_base_url"] = vllm_base
    explicit = {k: v for k, v in (options or {}).items() if v is not None}
    return {**defaults, **explicit}


def create_auto_annotation_run(
    session: Session,
    *,
    sample_ids: list[str],
    options: dict[str, Any],
    model_name: str,
    created_by: str | None = None,
) -> tuple[AutoAnnotationRun, Job]:
    """Persist a run, its sample work items, and the generic job tracker."""
    run = AutoAnnotationRun(
        model_name=model_name,
        status="pending",
        input_sample_count=len(sample_ids),
        options_json=json.dumps(sanitize_options(options), ensure_ascii=False),
        created_by=created_by,
    )
    session.add(run)
    session.flush()
    for sample_id in sample_ids:
        session.add(AutoAnnotationItem(run_id=run.id, sample_id=sample_id, status="pending"))
    job = Job(job_type="auto_annotation", reference_id=run.id, total=len(sample_ids))
    session.add(job)
    session.add(AuditEvent(
        action="auto_annotation.created",
        user_id=created_by,
        entity_type="auto_annotation_run",
        entity_id=run.id,
        payload_json=json.dumps({"sample_count": len(sample_ids), "model_name": model_name}, ensure_ascii=False),
    ))
    session.flush()
    return run, job


def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.id,
        "target_index": candidate.target_index,
        "class_id": candidate.class_id,
        "class_name": candidate.class_name,
        "confidence": candidate.confidence,
        "bbox": [candidate.x1, candidate.y1, candidate.x2, candidate.y2],
        "raw_decision": candidate.raw_decision,
        "yolo_decision": candidate.yolo_decision,
        "llm_decision": candidate.llm_decision,
        "reason_code": candidate.reason_code,
    }


def _latest_candidates(session: Session, sample: Sample) -> tuple[Asset | None, list[Candidate]]:
    asset = session.scalar(select(Asset).where(Asset.sample_id == sample.id, Asset.kind == "original"))
    task = session.scalar(select(AnnotationTask).where(AnnotationTask.sample_id == sample.id).order_by(AnnotationTask.created_at.desc()))
    if task is None:
        return asset, []
    candidates = session.scalars(
        select(Candidate).where(
            Candidate.sample_id == sample.id,
            Candidate.model_run_id == task.model_run_id,
        ).order_by(Candidate.target_index)
    ).all()
    return asset, candidates


def _normalise_decisions(result: Any, candidates: list[Candidate]) -> dict[int, dict[str, Any]]:
    if not isinstance(result, dict):
        raise ValueError("auto annotation adapter returned an invalid object")
    decisions = result.get("decisions")
    if decisions is None and isinstance(result.get("results"), list):
        decisions = result["results"]
    if decisions is None and isinstance(result.get("votes"), list):
        decisions = [{"target_index": 0, "votes": result["votes"]}]
    if not isinstance(decisions, list):
        raise ValueError("auto annotation response has no decisions")
    response_scene = normalize_scene(result.get("scene"))
    expected = {candidate.target_index for candidate in candidates}
    parsed: dict[int, dict[str, Any]] = {}
    for position, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"decision {position} is invalid")
        try:
            target_index = int(decision.get("target_index", decision.get("candidate_index", position)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"decision {position} has invalid target_index") from exc
        if target_index in parsed:
            raise ValueError(f"duplicate decision for target {target_index}")
        votes = decision.get("votes")
        if votes is None and "decision" in decision:
            fast_item = parse_fast_response({"decisions": [{**decision, "target_index": target_index}]})["decisions"][0]
            votes = fast_item["votes"]
        if isinstance(votes, dict):
            # Preferred contract from auto_annotation_prompts.py: role names
            # are keys, making it impossible to silently reorder a vote.
            if set(votes) != set(ROLE_ORDER):
                raise ValueError(f"target {target_index} must have exactly the three role votes")
            vote_values = [votes[role] for role in ROLE_ORDER]
            vote_roles = list(ROLE_ORDER)
        elif isinstance(votes, list):
            if len(votes) != 3:
                raise ValueError(f"target {target_index} must have exactly three votes")
            vote_values = votes
            vote_roles = []
        else:
            raise ValueError(f"target {target_index} has invalid votes")
        normalized_votes = []
        for vote_position, vote in enumerate(vote_values):
            if not isinstance(vote, dict):
                raise ValueError(f"target {target_index} vote {vote_position} is invalid")
            if vote_roles and set(vote) != {"label", "confidence", "reason"}:
                raise ValueError(f"target {target_index} role vote {vote_position} has invalid fields")
            if not vote_roles:
                supplied_role = vote.get("role")
                expected_role = ROLE_ORDER[vote_position]
                if supplied_role is not None and str(supplied_role) != expected_role:
                    raise ValueError(f"target {target_index} vote {vote_position} has the wrong role")
            label = str(vote.get("label", "")).strip().lower()
            if label not in {"true", "false", "uncertain"}:
                raise ValueError(f"target {target_index} vote {vote_position} has invalid label")
            try:
                confidence = float(vote.get("confidence"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"target {target_index} vote {vote_position} has invalid confidence") from exc
            if not 0 <= confidence <= 1:
                raise ValueError(f"target {target_index} vote {vote_position} has invalid confidence")
            normalized_votes.append({
                "role": str(vote.get("role") or (vote_roles[vote_position] if vote_roles else ROLE_ORDER[vote_position])),
                "label": label,
                "confidence": confidence,
                "reason": str(vote.get("reason") or "")[:1000],
            })
        parsed[target_index] = {
            "target_index": target_index,
            "votes": normalized_votes,
            "votes_by_role": dict(zip(ROLE_ORDER, normalized_votes, strict=True)) if vote_roles else None,
            "consensus": decision.get("consensus"),
            "scene": normalize_scene(decision.get("scene", response_scene)),
            "evidence": decision.get("evidence") if isinstance(decision.get("evidence"), dict) else {
                key: decision[key]
                for key in (
                    "red_overlay_occlusion",
                    "overlay_obscures",
                    "background_only",
                    "target_visible",
                    "occlusion_type",
                    "evidence_reason",
                )
                if key in decision
            },
        }
    if set(parsed) != expected:
        missing = sorted(expected - set(parsed))
        extra = sorted(set(parsed) - expected)
        raise ValueError(f"decisions do not match candidates (missing={missing}, extra={extra})")
    return parsed


def decision_from_votes(votes: list[dict[str, Any]], *, votes_by_role: dict[str, dict[str, Any]] | None = None, consensus: Any = None) -> str:
    """Apply local fail-open consensus: only 2 confident votes can decide."""
    if votes_by_role is not None:
        strict_votes = {
            role: {key: vote[key] for key in ("label", "confidence", "reason")}
            for role, vote in votes_by_role.items()
        }
        merged = merge_agent_results({
            "votes": strict_votes,
            "consensus": consensus if isinstance(consensus, dict) else {"label": None, "confidence": 0.0, "reason": "local recomputation"},
        })
        return merged["label"] if merged is not None else "uncertain"
    eligible = [vote["label"] for vote in votes if vote["label"] in {"true", "false"} and vote["confidence"] >= AUTO_ANNOTATION_CONFIDENCE_FLOOR]
    counts = Counter(eligible)
    if counts["true"] >= 2 and counts["true"] > counts["false"]:
        return "true"
    if counts["false"] >= 2 and counts["false"] > counts["true"]:
        return "false"
    return "uncertain"


def _model_confidence(
    votes: list[dict[str, Any]],
    *,
    votes_by_role: dict[str, dict[str, Any]] | None = None,
    consensus: Any = None,
) -> float | None:
    """Return the local merge confidence without trusting model consensus."""
    if votes_by_role is not None:
        strict_votes = {
            role: {key: vote[key] for key in ("label", "confidence", "reason")}
            for role, vote in votes_by_role.items()
        }
        merged = merge_agent_results({
            "votes": strict_votes,
            "consensus": consensus if isinstance(consensus, dict) else {
                "label": None,
                "confidence": 0.0,
                "reason": "local recomputation",
            },
        })
        return float(merged["confidence"]) if merged is not None else None
    label = decision_from_votes(votes)
    agreeing = [vote["confidence"] for vote in votes if vote.get("label") == label and vote.get("confidence", 0) >= AUTO_ANNOTATION_CONFIDENCE_FLOOR]
    return min(agreeing) if agreeing else None


def _audit(session: Session, *, action: str, sample_id: str, candidate_id: str | None, payload: dict[str, Any]) -> None:
    session.add(AuditEvent(
        action=action,
        user_id=None,
        entity_type="candidate" if candidate_id else "sample",
        entity_id=candidate_id or sample_id,
        payload_json=json.dumps(sanitize_options(payload), ensure_ascii=False),
    ))


def _call_adapter_once(adapter: Any, image: dict[str, Any], candidates: list[dict[str, Any]], options: dict[str, Any]) -> Any:
    if hasattr(adapter, "review"):
        return adapter.review(image, candidates, options)
    adapter_options = dict(options)
    adapter_options["candidates"] = candidates
    return adapter.infer(image, adapter_options)


def _invoke_adapter(adapter: Any, image: dict[str, Any], candidates: list[dict[str, Any]], options: dict[str, Any]) -> Any:
    mode = str(options.get("mode") or "").strip().lower()
    if mode == "adaptive" and not hasattr(adapter, "_post_responses_payload") and not hasattr(adapter, "_call_chat_completions"):
        fast_opts = {**options, "mode": "fast"}
        fast_res = _call_adapter_once(adapter, image, candidates, fast_opts)
        if isinstance(fast_res, dict):
            decisions = fast_res.get("decisions") or []
            needs_fallback = False
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                if "decision" in d and "votes" not in d:
                    norm_d = parse_fast_response({"decisions": [d]})["decisions"][0]
                    if norm_d["label"] == "uncertain" or norm_d["confidence"] < 0.75:
                        needs_fallback = True
                        break
                elif "votes" in d:
                    votes = d["votes"]
                    vote_list = list(votes.values()) if isinstance(votes, dict) else list(votes)
                    lbl = decision_from_votes(vote_list)
                    conf = _model_confidence(vote_list) or 0.0
                    if lbl == "uncertain" or conf < 0.75:
                        needs_fallback = True
                        break
            if not needs_fallback:
                return fast_res
        deep_opts = {**options, "mode": "deep"}
        return _call_adapter_once(adapter, image, candidates, deep_opts)
    return _call_adapter_once(adapter, image, candidates, options)


def _synthetic_cascade_decision(target_index: int, routing: dict[str, Any]) -> dict[str, Any]:
    auto_label = str(routing["auto_label"])
    conf = float(routing.get("confidence") or (0.95 if auto_label == "true" else 0.95))
    reason = str(routing.get("reason_code") or "cascade_auto")
    votes = [
        {"role": role, "label": auto_label, "confidence": conf, "reason": reason}
        for role in ROLE_ORDER
    ]
    return {
        "target_index": target_index,
        "votes": votes,
        "votes_by_role": None,
        "consensus": auto_label,
        "scene": normalize_scene(None),
        "evidence": {},
        "cascade_routing": routing,
    }


class JevOmniTierRunner:
    """Tier 2.5 执行器：双视角合图 → Jev-Omni 校准概率 → 级联裁决。

    安全契约：**任何异常都返回 ``None``**（依赖缺失、权重未下载、HTTP 超时、
    推理服务 5xx、图像不可解码、概率不可解析）。调用方拿到 ``None`` 时把候选
    原样送入 Tier 3，因此接入 Jev-Omni 不会引入新的失败路径，也不会写入
    任何猜测标签。
    """

    def __init__(
        self,
        adapter: Any,
        image_loader: Any,
        *,
        confirm_threshold: float = JEV_OMNI_CONFIRM_THRESHOLD,
        reject_threshold: float = JEV_OMNI_REJECT_THRESHOLD,
    ) -> None:
        if adapter is None or not callable(image_loader):
            raise ValueError("JevOmniTierRunner 需要 adapter 与 image_loader")
        self.adapter = adapter
        self.image_loader = image_loader
        self.confirm_threshold = float(confirm_threshold)
        self.reject_threshold = float(reject_threshold)
        self.last_error: str | None = None
        self._image_cache_key: tuple[str, str] | None = None
        self._image_cache: bytes | None = None

    def _load_image(self, image_ref: dict[str, Any]) -> bytes:
        key = (str(image_ref.get("bucket") or ""), str(image_ref.get("object_key") or ""))
        if self._image_cache is not None and self._image_cache_key == key:
            return self._image_cache
        payload = self.image_loader(image_ref)
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            raise ValueError("源图为空")
        self._image_cache_key = key
        self._image_cache = bytes(payload)
        return self._image_cache

    def evaluate(
        self,
        image_ref: dict[str, Any],
        candidate: dict[str, Any],
        *,
        class_name: str | None = None,
        candidate_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """返回 ``evaluate_jev_omni_tier`` 的裁决，或 ``None`` 表示安全回退。"""
        from ..adapters.jev_omni_classifier import build_jev_omni_review_views

        self.last_error = None
        try:
            image_bytes = self._load_image(image_ref)
            context_img, roi_img = build_jev_omni_review_views(image_bytes, candidate.get("bbox"))
            result = self.adapter.decide(
                context_img,
                roi_img,
                class_name if class_name is not None else candidate.get("class_name"),
                candidate_meta,
            )
            return evaluate_jev_omni_tier(
                candidate,
                p_true=result["p_true"],
                p_false=result["p_false"],
                confirm_threshold=self.confirm_threshold,
                reject_threshold=self.reject_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - 安全回退是显式契约
            self.last_error = _safe_error(exc)
            return None


def _process_item(
    session: Session,
    item: AutoAnnotationItem,
    *,
    adapter: Any,
    options: dict[str, Any],
    jev_omni_runner: JevOmniTierRunner | None = None,
) -> None:
    sample = session.get(Sample, item.sample_id)
    if sample is None:
        raise ValueError("sample not found")
    asset, candidates = _latest_candidates(session, sample)
    if asset is None or asset.status == "missing":
        raise ValueError("original asset not found")
    if not candidates:
        # There is no candidate to label. This is a successful no-op and does
        # not spend a network request on an image that has nothing to review.
        _audit(session, action="auto_annotation.no_candidates", sample_id=sample.id, candidate_id=None, payload={"status": "no_candidates"})
        return
    image = {"bucket": asset.bucket, "object_key": asset.object_key, "sha256": asset.sha256}
    candidate_payloads = apply_pre_llm_geometric_gate(
        [_candidate_payload(candidate) for candidate in candidates]
    )
    decisions: dict[int, dict[str, Any]] = {}
    if options.get("cascade_routing"):
        tier1_th = float(options.get("tier1_threshold", TIER1_AUTO_CONFIRM_THRESHOLD))
        tier2_th = float(options.get("tier2_threshold", TIER2_AUTO_REJECT_THRESHOLD))
        tier3_payloads: list[dict[str, Any]] = []
        tier3_candidates: list[Candidate] = []
        cand_by_idx = {c.target_index: c for c in candidates}
        for payload in candidate_payloads:
            routing = evaluate_cascade_tier(
                payload, tier1_threshold=tier1_th, tier2_threshold=tier2_th
            )
            t_idx = int(payload["target_index"])
            if routing["requires_vlm"] and jev_omni_runner is not None:
                tier25 = jev_omni_runner.evaluate(
                    image,
                    payload,
                    class_name=payload.get("class_name"),
                    candidate_meta={
                        "candidate_id": payload.get("candidate_id"),
                        "class_name": payload.get("class_name"),
                        "detector_confidence": payload.get("confidence"),
                        "geometric_warning": payload.get("geometric_warning"),
                    },
                )
                if tier25 is not None and not tier25["requires_vlm"]:
                    decisions[t_idx] = _synthetic_cascade_decision(
                        t_idx,
                        {
                            **tier25,
                            "confidence": (
                                tier25["p_true"] if tier25["auto_label"] == "true" else tier25["p_false"]
                            ),
                        },
                    )
                    continue
                if tier25 is not None:
                    # 灰区/场景依赖：把 Jev-Omni 校准概率作为先验传给下游 L20。
                    payload = {
                        **payload,
                        "jev_omni_p_true": tier25["p_true"],
                        "jev_omni_p_false": tier25["p_false"],
                        "jev_omni_reason_code": tier25["reason_code"],
                        "jev_omni_explanation": tier25["explanation"],
                    }
                    routing = {**routing, "jev_omni": tier25}
            if routing["requires_vlm"]:
                tier3_payloads.append(payload)
                tier3_candidates.append(cand_by_idx[t_idx])
            else:
                decisions[t_idx] = _synthetic_cascade_decision(t_idx, routing)
        if tier3_payloads:
            result = _invoke_adapter(adapter, image, tier3_payloads, options)
            decisions.update(_normalise_decisions(result, tier3_candidates))
    else:
        result = _invoke_adapter(adapter, image, candidate_payloads, options)
        decisions = _normalise_decisions(result, candidates)
    for candidate in candidates:
        decision = decisions[candidate.target_index]
        votes = decision["votes"]
        model_label = decision_from_votes(votes, votes_by_role=decision.get("votes_by_role"), consensus=decision.get("consensus"))
        model_confidence = _model_confidence(
            votes,
            votes_by_role=decision.get("votes_by_role"),
            consensus=decision.get("consensus"),
        )
        policy = apply_review_policy(
            class_name=candidate.class_name,
            model_label=model_label,
            model_confidence=model_confidence,
            scene=decision.get("scene"),
            evidence=decision.get("evidence"),
            rooftop_confidence_threshold=options.get("rooftop_scene_confidence_threshold", 0.8),
        )
        label = policy.label
        cascade_routing = decision.get("cascade_routing")
        audit_payload = {
            "target_index": candidate.target_index,
            "votes": votes,
            "model_consensus": decision.get("consensus"),
            "model_label": model_label,
            "model_confidence": model_confidence,
            "scene": policy.scene,
            "evidence": policy.evidence,
            "policy": policy.as_dict(),
            "local_decision": label,
            "confidence_floor": AUTO_ANNOTATION_CONFIDENCE_FLOOR,
        }
        if cascade_routing:
            audit_payload["cascade_routing"] = cascade_routing
        if label == "uncertain":
            _audit(session, action="auto_annotation.uncertain", sample_id=sample.id, candidate_id=candidate.id, payload=audit_payload)
            continue
        annotation = session.scalar(select(Annotation).where(Annotation.candidate_id == candidate.id))
        if annotation is not None and annotation.label is not None:
            _audit(session, action="auto_annotation.skipped_existing", sample_id=sample.id, candidate_id=candidate.id, payload={**audit_payload, "existing_label": annotation.label})
            continue
        if annotation is None:
            annotation = Annotation(candidate_id=candidate.id, revision=0)
            session.add(annotation)
        else:
            annotation.revision = (annotation.revision or 0) + 1
        annotation.label = label
        note = f"auto annotation {policy.reason_code}: " + "; ".join(
            f"{vote['role']}={vote['label']} ({vote['confidence']:.2f})" for vote in votes
        )
        explanation = (cascade_routing or {}).get("explanation")
        if explanation:
            note = f"{note}; {explanation}"
        annotation.note = note
        annotation.labeled_by = None
        annotation.labeled_at = utc_now()
        session.flush()
        _audit(session, action="auto_annotation.applied", sample_id=sample.id, candidate_id=candidate.id, payload=audit_payload)


def _build_jev_omni_runner(
    options: dict[str, Any],
    object_store: Any,
    jev_omni_adapter: Any | None,
) -> tuple[JevOmniTierRunner | None, Any | None]:
    """构造 Tier 2.5 执行器，返回 ``(runner, 需要关闭的适配器)``。

    ``JEV_OMNI_ENABLED=false``（默认）时立即返回 ``(None, None)``，级联行为与
    接入前完全一致。任何构造失败都退化为 ``None``，不阻断主任务。
    """
    if not options.get("jev_omni_enabled"):
        return None, None
    owned_adapter: Any | None = None
    adapter = jev_omni_adapter
    if adapter is None:
        try:
            from ..adapters.jev_omni_classifier import build_jev_omni_adapter

            adapter = build_jev_omni_adapter(options)
        except Exception:  # noqa: BLE001 - 配置错误不应让整个 job 失败
            return None, None
        owned_adapter = adapter
    if adapter is None or object_store is None or not hasattr(object_store, "get_bytes"):
        return None, owned_adapter
    runner = JevOmniTierRunner(
        adapter,
        lambda ref: object_store.get_bytes(ref["bucket"], ref["object_key"]),
        confirm_threshold=float(
            options.get("jev_omni_confirm_threshold", JEV_OMNI_CONFIRM_THRESHOLD)
        ),
        reject_threshold=float(
            options.get("jev_omni_reject_threshold", JEV_OMNI_REJECT_THRESHOLD)
        ),
    )
    return runner, owned_adapter


def execute_auto_annotation_job(
    *,
    session_factory,
    object_store,
    adapter,
    job_id: str,
    jev_omni_adapter: Any | None = None,
) -> None:
    """Execute a job, isolating each sample's network/parse failure.

    ``object_store`` is now also used by the Tier 2.5 (Jev-Omni) fast gate: the
    runner needs the original frame to build the double-view composite image.
    When ``JEV_OMNI_ENABLED`` is false (the default) the runner is never built and
    behaviour is identical to the previous three-tier cascade.
    """
    with session_factory() as session:
        job = session.get(Job, job_id)
        if job is None or job.job_type != "auto_annotation":
            return
        run = session.get(AutoAnnotationRun, job.reference_id)
        if run is None:
            job.status = "failed"
            job.error_summary = "auto annotation run not found"
            session.commit()
            return
        if job.status == "succeeded":
            return
        job.status = "running"
        job.started_at = utc_now()
        run.status = "running"
        session.commit()
        options = json.loads(run.options_json or "{}")
        runner, owned_adapter = _build_jev_omni_runner(options, object_store, jev_omni_adapter)
        items = session.scalars(select(AutoAnnotationItem).where(AutoAnnotationItem.run_id == run.id).order_by(AutoAnnotationItem.created_at)).all()
        try:
            for item in items:
                if item.status == "succeeded":
                    continue
                try:
                    _process_item(
                        session,
                        item,
                        adapter=adapter,
                        options=options,
                        jev_omni_runner=runner,
                    )
                    item.status = "succeeded"
                    job.completed += 1
                except Exception as exc:
                    # A database/serialization failure after an earlier candidate
                    # in this sample was written must not leave a half-labelled
                    # sample behind. Roll back the item transaction, then record
                    # the isolated failure in a fresh session state.
                    item_id = item.id
                    session.rollback()
                    job = session.get(Job, job_id)
                    item = session.get(AutoAnnotationItem, item_id)
                    error = _safe_error(exc)
                    item.status = "failed"
                    item.error_message = error
                    job.failed += 1
                    _audit(session, action="auto_annotation.failed", sample_id=item.sample_id, candidate_id=None, payload={"error": error})
                session.commit()
        finally:
            if owned_adapter is not None:
                try:
                    owned_adapter.close()
                except Exception:  # noqa: BLE001 - 关闭连接池失败不影响任务结果
                    pass

        run.status = "partial_failed" if job.failed and job.completed else ("failed" if job.failed else "succeeded")
        run.error_summary = "one or more samples failed" if job.failed else None
        run.finished_at = utc_now()
        job.status = run.status
        job.finished_at = utc_now()
        session.commit()
