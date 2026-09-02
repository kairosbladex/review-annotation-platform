from __future__ import annotations

import json

import pytest

from app.services.ktd_recheck import (
    PROMPT_VERSION,
    ProviderChain,
    RecheckStore,
    RecheckProviderError,
    AttemptOutcome,
    CommandProvider,
    GroupedCandidate,
    GroupedImage,
    ProviderResponse,
    _preflight_provider,
    decision_label,
    group_source_rows,
    request_fingerprint,
    validate_batch_response,
    write_outputs,
)


def _candidate(x1: float = 0.1, *, confidence: float = 0.8) -> dict:
    return {
        "class_name": "车",
        "confidence": confidence,
        "bbox": [x1, 0.1, x1 + 0.1, 0.2],
    }


def _vote(label: str, confidence: float = 0.8, reason: str = "证据") -> dict:
    return {"label": label, "confidence": confidence, "reason": reason}


def _decision(target_index: int, label: str = "true") -> dict:
    return {
        "target_index": target_index,
        "votes": [
            {"role": "target_presence_fit", **_vote(label)},
            {"role": "class_scene", **_vote(label)},
            {"role": "adversarial", **_vote("uncertain", 0.7)},
        ],
        "consensus": label,
    }


def test_group_source_rows_preserves_ids_deduplicates_exact_candidates_and_marks_conflicts():
    images = group_source_rows(
        [
            {"id": "row-a", "file_path": "/same.jpg", "targets": [_candidate(), _candidate(0.4)]},
            {"id": "row-b", "file_path": "/same.jpg", "targets": [_candidate(), _candidate(0.4)]},
            {"id": "row-c", "file_path": "/conflict.jpg", "targets": [_candidate()]},
            {"id": "row-d", "file_path": "/conflict.jpg", "targets": [_candidate(0.5)]},
        ]
    )

    assert [item.file_path for item in images] == ["same.jpg", "conflict.jpg"]
    assert images[0].source_row_ids == ["row-a", "row-b"]
    assert len(images[0].candidates) == 2
    assert images[0].conflict is False
    assert images[1].source_row_ids == ["row-c", "row-d"]
    assert len(images[1].candidates) == 2
    assert images[1].conflict is True
    assert all(candidate.needs_manual_review for candidate in images[1].candidates)


def test_validate_batch_response_requires_every_image_and_candidate_exactly_once():
    candidates = {"image-a": [{"target_index": 0}, {"target_index": 1}], "image-b": [{"target_index": 0}]}
    response = {
        "images": [
            {"image_id": "image-a", "decisions": [_decision(0), _decision(1)]},
            {"image_id": "image-b", "decisions": [_decision(0)]},
        ]
    }

    parsed = validate_batch_response(response, candidates)

    assert set(parsed) == {"image-a", "image-b"}
    assert parsed["image-a"][1]["votes"][0]["label"] == "true"

    with pytest.raises(RecheckProviderError):
        validate_batch_response(
            {"images": [{"image_id": "image-a", "decisions": [_decision(0)]}]},
            candidates,
        )


def test_uncertain_is_valid_and_does_not_produce_a_binary_label():
    votes = [_vote("uncertain"), _vote("true"), _vote("false")]

    assert decision_label(votes) == "uncertain"


def test_request_fingerprint_changes_with_image_candidate_prompt_or_model():
    base = request_fingerprint(
        image_hashes=["a"],
        candidates=[_candidate()],
        prompt_version=PROMPT_VERSION,
        model="gpt-test",
    )

    assert base == request_fingerprint(
        image_hashes=["a"],
        candidates=[_candidate()],
        prompt_version=PROMPT_VERSION,
        model="gpt-test",
    )
    assert base != request_fingerprint(image_hashes=["b"], candidates=[_candidate()], prompt_version=PROMPT_VERSION, model="gpt-test")
    assert base != request_fingerprint(image_hashes=["a"], candidates=[_candidate(0.2)], prompt_version=PROMPT_VERSION, model="gpt-test")
    assert base != request_fingerprint(image_hashes=["a"], candidates=[_candidate()], prompt_version="other", model="gpt-test")
    assert base != request_fingerprint(image_hashes=["a"], candidates=[_candidate()], prompt_version=PROMPT_VERSION, model="other")


class FakeProvider:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls = 0

    def review(self, batch, prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def test_provider_chain_stops_after_first_valid_provider_and_fails_over_in_order():
    valid = {"images": [{"image_id": "image-a", "decisions": [_decision(0)]}]}
    first = FakeProvider(error=RuntimeError("rate limited"))
    second = FakeProvider(result=valid)
    third = FakeProvider(result=valid)
    chain = ProviderChain(
        [("gpt", "gpt-model", first), ("grok", "grok-model", second), ("kimi", "kimi-model", third)]
    )

    result = chain.review(
        batch=[{"image_id": "image-a", "candidates": [{"target_index": 0}]}],
        prompt="prompt",
        expected={"image-a": [{"target_index": 0}]},
    )

    assert result.provider == "grok"
    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 0


def test_provider_chain_does_not_fail_over_for_valid_uncertain_result():
    valid_uncertain = {
        "images": [{
            "image_id": "image-a",
            "decisions": [{
                "target_index": 0,
                "votes": [
                    {"role": "target_presence_fit", **_vote("uncertain")},
                    {"role": "class_scene", **_vote("uncertain")},
                    {"role": "adversarial", **_vote("uncertain")},
                ],
                "consensus": "uncertain",
            }],
        }]
    }
    first = FakeProvider(result=valid_uncertain)
    second = FakeProvider(result=valid_uncertain)
    chain = ProviderChain([("gpt", "gpt-model", first), ("grok", "grok-model", second)])

    result = chain.review(
        batch=[{"image_id": "image-a", "candidates": [{"target_index": 0}]}],
        prompt="prompt",
        expected={"image-a": [{"target_index": 0}]},
    )

    assert result.provider == "gpt"
    assert second.calls == 0


def test_provider_chain_reports_all_failures_without_partial_success():
    first = FakeProvider(error=RuntimeError("bad json"))
    second = FakeProvider(error=RuntimeError("timeout"))
    chain = ProviderChain([("gpt", "gpt-model", first), ("grok", "grok-model", second)])

    with pytest.raises(RecheckProviderError, match="all providers failed"):
        chain.review(
            batch=[{"image_id": "image-a", "candidates": [{"target_index": 0}]}],
            prompt="prompt",
            expected={"image-a": [{"target_index": 0}]},
        )

    assert first.calls == second.calls == 1


def test_command_provider_uses_cli_flags_supported_by_current_build(monkeypatch, tmp_path):
    calls = []

    class Process:
        pid = 12345
        returncode = 0

        def communicate(self, **_kwargs):
            return json.dumps({"text": "{}"}), ""

    def fake_popen(command, **_kwargs):
        calls.append(command)
        return Process()

    monkeypatch.setattr("app.services.ktd_recheck.subprocess.Popen", fake_popen)

    CommandProvider(name="grok", binary="grok", model="grok-test")._review_grok(tmp_path, [], "prompt")
    CommandProvider(name="kimi", binary="kimi", model="kimi-test")._review_kimi(tmp_path, [], "prompt")

    assert "--max-turns" in calls[0]
    assert calls[0][calls[0].index("--max-turns") + 1] == "4"
    assert "-p" in calls[1]
    assert "-y" not in calls[1]
    assert "--auto" not in calls[1]


def test_gpt_command_provider_can_use_responses_http_without_cli_plugin_sync(monkeypatch):
    response_payload = {
        "output_text": json.dumps({"images": []}),
    }
    captured = {}

    class Response:
        text = json.dumps(response_payload)

        def raise_for_status(self):
            return None

        def json(self):
            return response_payload

    class Client:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, *, json, headers):
            captured.update(url=url, json=json, headers=headers)
            return Response()

    monkeypatch.setattr("app.services.ktd_recheck.httpx.Client", Client)
    provider = CommandProvider(
        name="gpt",
        binary="codex",
        model="gpt-test",
        api_key="secret",
        base_url="https://gateway.example",
        endpoint="/v1/responses",
        use_codex_cli=False,
    )

    result = provider.review([], "prompt")

    assert isinstance(result, ProviderResponse)
    assert result.payload == {"images": []}
    assert captured["url"] == "https://gateway.example/v1/responses"
    assert captured["headers"] == {"Authorization": "Bearer secret", "Content-Type": "application/json"}
    assert captured["json"]["model"] == "gpt-test"


def test_failed_preflight_is_retried_after_resume(tmp_path):
    image = GroupedImage(
        image_id="image-a",
        file_path="images/a.jpg",
        source_row_ids=["row-a"],
        candidates=[GroupedCandidate(candidate_key="key-a", payload=_candidate(), source_row_ids=["row-a"])],
        conflict=False,
    )
    store = RecheckStore(tmp_path / "recheck.sqlite3")
    store.create_or_validate_run(
        run_id="run-a",
        output_dir=tmp_path / "outputs",
        images=[image],
        provider_order=["gpt"],
        batch_size=10,
        concurrency=1,
        resume=False,
    )
    bundle = {
        "image_id": "image-a",
        "file_path": "images/a.jpg",
        "source_row_ids": ["row-a"],
        "candidates": [{**_candidate(), "target_index": 0}],
        "sha256": "sha",
    }
    valid = {"images": [{"image_id": "image-a", "decisions": [_decision(0)]}]}
    provider = FakeProvider(error=RuntimeError("temporary outage"))

    assert _preflight_provider(store=store, run_id="run-a", provider_name="gpt", model="gpt-test", provider=provider, bundle=bundle) is False
    provider.error = None
    provider.result = valid
    assert _preflight_provider(store=store, run_id="run-a", provider_name="gpt", model="gpt-test", provider=provider, bundle=bundle) is True
    assert provider.calls == 2


def test_store_and_outputs_keep_attempts_traceable_without_secrets(tmp_path):
    store = RecheckStore(tmp_path / "recheck.sqlite3")
    image = GroupedImage(
        image_id="image-a",
        file_path="images/a.jpg",
        source_row_ids=["row-a", "row-b"],
        candidates=[GroupedCandidate(candidate_key="key-a", payload={"class_name": "车", "bbox": [0.1, 0.1, 0.2, 0.2], "confidence": 0.8}, source_row_ids=["row-a", "row-b"])],
        conflict=False,
        source_rows=[{"source_row_id": "row-a", "source_payload": {"id": "row-a"}}, {"source_row_id": "row-b", "source_payload": {"id": "row-b"}}],
    )
    store.create_or_validate_run(
        run_id="run-a",
        output_dir=tmp_path / "outputs",
        images=[image],
        provider_order=["gpt"],
        batch_size=10,
        concurrency=2,
        resume=False,
    )
    decision = _decision(0)
    result = {"image-a": {0: decision}}
    failed_outcome = AttemptOutcome(
        provider="gpt",
        model="gpt-test",
        status="failed",
        error_message="rate limited",
    )
    outcome = AttemptOutcome(
        provider="grok",
        model="grok-test",
        status="succeeded",
        raw_response='{"images": []}',
        parsed_results=result,
        error_message="Bearer super-secret should never be persisted",
    )
    fingerprint = request_fingerprint(image_hashes=["sha"], candidates=[_candidate()], prompt_version=PROMPT_VERSION, model="gpt-test")
    grok_fingerprint = request_fingerprint(image_hashes=["sha"], candidates=[_candidate()], prompt_version=PROMPT_VERSION, model="grok-test")
    store.save_attempts(
        run_id="run-a",
        batch_id="batch-a",
        outcomes=[failed_outcome, outcome],
        image_ids=["image-a"],
        source_row_ids=["row-a", "row-b"],
        candidate_indices={"image-a": [0]},
        fingerprints={("gpt", "gpt-test"): fingerprint, ("grok", "grok-test"): grok_fingerprint},
    )
    store.apply_results(run_id="run-a", image_ids=["image-a"], results=result, provider="gpt", model="gpt-test", sha256_by_image={"image-a": "sha"})

    summary = write_outputs(store, "run-a", tmp_path / "outputs")
    attempts_text = (tmp_path / "outputs" / "attempts.jsonl").read_text(encoding="utf-8")
    assert "super-secret" not in attempts_text
    assert "row-a" in attempts_text and "row-b" in attempts_text
    assert summary["unique_images"] == 1
    assert summary["labels"]["true"] == 1
    assert summary["fallback_attempts"] == 1
    assert len((tmp_path / "outputs" / "pseudo_label_manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert (tmp_path / "outputs" / "training_manifest.jsonl").read_text(encoding="utf-8") == ""

    assert len(store.pending_images("run-a")) == 0
    store.create_or_validate_run(
        run_id="run-a",
        output_dir=tmp_path / "outputs",
        images=[image],
        provider_order=["gpt"],
        batch_size=10,
        concurrency=2,
        resume=True,
    )
    assert len(store.pending_images("run-a")) == 0
