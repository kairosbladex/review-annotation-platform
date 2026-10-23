from __future__ import annotations

import json
import subprocess

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
    _write_yolo_training_files,
    arbitrate_jev_omni_and_vlm,
    dataset_class_map,
    decision_label,
    group_source_rows,
    load_uncertain_images,
    request_fingerprint,
    render_multi_image_prompt,
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


def test_arbiter_resolves_uncertain_with_jev_omni_and_detector_signals():
    assert arbitrate_jev_omni_and_vlm(
        "uncertain", jev_omni_p_true=0.10, detector_confidence=0.70
    ) == {
        "decision": "false",
        "reason_code": "arbiter_uncertain_resolved_false",
        "changed": True,
    }
    assert arbitrate_jev_omni_and_vlm(
        "uncertain", jev_omni_p_true=0.70, detector_confidence=0.30
    ) == {
        "decision": "true",
        "reason_code": "arbiter_uncertain_resolved_true",
        "changed": True,
    }


def test_arbiter_vetoes_known_vlm_false_positive_patterns_and_preserves_other_labels():
    assert arbitrate_jev_omni_and_vlm(
        "true", jev_omni_p_true=0.01, detector_confidence=0.90
    ) == {
        "decision": "false",
        "reason_code": "arbiter_fp_vetoed_by_jev_omni",
        "changed": True,
    }
    assert arbitrate_jev_omni_and_vlm(
        "true", jev_omni_p_true=0.19, detector_confidence=0.20
    ) == {
        "decision": "false",
        "reason_code": "arbiter_fp_vetoed_by_jev_omni",
        "changed": True,
    }
    assert arbitrate_jev_omni_and_vlm(
        "false", jev_omni_p_true=0.90, detector_confidence=0.90
    ) == {
        "decision": "false",
        "reason_code": "arbiter_kept_vlm_decision",
        "changed": False,
    }


@pytest.mark.parametrize(
    ("class_name", "has_geometric_warning", "reason_code"),
    [
        ("blue_illegal_roof", False, "clean_match_quality_passed"),
        ("road_crack", False, "clean_match_quality_passed"),
        ("vehicle", True, "clean_match_quality_passed"),
        ("vehicle", False, "clean_match_quality_missing"),
    ],
)
def test_arbiter_never_promotes_unverified_or_guarded_tier3_to_true(
    class_name, has_geometric_warning, reason_code
):
    result = arbitrate_jev_omni_and_vlm(
        "uncertain",
        jev_omni_p_true=0.90,
        detector_confidence=0.70,
        class_name=class_name,
        has_geometric_warning=has_geometric_warning,
        vlm_reason_code=reason_code,
    )
    assert result["decision"] == "uncertain"
    assert result["changed"] is False
    assert result["reason_code"] in {
        "arbiter_kept_uncertain",
        "arbiter_kept_unverified_tier3",
        "arbiter_kept_guarded_tier3",
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


def test_single_image_prompt_distinguishes_attachment_pair_from_logical_images():
    prompt = render_multi_image_prompt([{
        "image_id": "image-a",
        "file_path": "source/image-a.jpg",
        "candidates": [{"target_index": 0}],
    }])

    assert "只有一个逻辑图片" in prompt
    assert "image-a" in prompt
    assert "images 数组只能包含一个对象" in prompt
    assert "禁止调用任何 MCP、Computer Use、浏览器或终端工具" in prompt


def test_uncertain_is_valid_and_does_not_produce_a_binary_label():
    votes = [_vote("uncertain"), _vote("true"), _vote("false")]

    assert decision_label(votes) == "uncertain"


def test_dataset_class_map_is_stable_and_uses_class_names():
    assert dataset_class_map(["车", "人员", "车", "两轮车"]) == {
        "两轮车": 0,
        "人员": 1,
        "车": 2,
    }


def test_yolo_label_writer_clamps_rounded_edge_boxes(tmp_path):
    _write_yolo_training_files(tmp_path, [{
        "image_id": "edge-image",
        "image_bucket": "minio-src",
        "image_object_key": "images/edge.jpg",
        "file_path": "images/edge.jpg",
        "candidate_index": 0,
        "class_id": 8,
        "bbox": [0.0, 0.0, 0.028125, 0.041666666666666664],
    }])

    values = [float(value) for value in (tmp_path / "yolo_labels" / "edge-image.txt").read_text(encoding="utf-8").split()]
    _, center_x, center_y, width, height = values
    assert center_x - width / 2 >= 0
    assert center_y - height / 2 >= 0


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


def test_command_provider_timeout_does_not_wait_on_inherited_pipes(monkeypatch, tmp_path):
    class Process:
        pid = 12345
        returncode = -15
        stdin = stdout = stderr = None

        def __init__(self):
            self.communicate_calls = 0
            self.wait_calls = []

        def communicate(self, **_kwargs):
            self.communicate_calls += 1
            raise subprocess.TimeoutExpired("fake", 1, output="partial", stderr="diagnostic")

        def wait(self, **kwargs):
            self.wait_calls.append(kwargs)
            return self.returncode

    process = Process()
    monkeypatch.setattr("app.services.ktd_recheck.subprocess.Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr("app.services.ktd_recheck.os.killpg", lambda *_args: None)

    with pytest.raises(RecheckProviderError, match="timed out"):
        CommandProvider(name="gpt", binary="codex", model="gpt-test", timeout_seconds=1)._run(
            ["codex"], prompt="prompt", cwd=tmp_path
        )

    assert process.communicate_calls == 1
    assert process.wait_calls == [{"timeout": 5}]


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


def test_load_uncertain_images_selects_only_uncertain_candidates_and_preserves_lineage(tmp_path):
    store = RecheckStore(tmp_path / "recheck.sqlite3")
    image = GroupedImage(
        image_id="image-a",
        file_path="images/a.jpg",
        source_row_ids=["row-a"],
        candidates=[
            GroupedCandidate(candidate_key="uncertain-key", payload=_candidate(), source_row_ids=["row-a"]),
            GroupedCandidate(candidate_key="true-key", payload=_candidate(0.4), source_row_ids=["row-a"]),
        ],
        conflict=False,
        source_rows=[{"source_row_id": "row-a", "source_payload": {"id": "row-a"}}],
    )
    store.create_or_validate_run(
        run_id="first-pass",
        output_dir=tmp_path / "first",
        images=[image],
        provider_order=["gpt"],
        batch_size=10,
        concurrency=1,
        resume=False,
    )
    store.apply_results(
        run_id="first-pass",
        image_ids=["image-a"],
        results={"image-a": {0: _decision(0, "uncertain"), 1: _decision(1, "true")}},
        provider="gpt",
        model="gpt-test",
        sha256_by_image={"image-a": "sha"},
    )

    selected = load_uncertain_images(store, "first-pass")

    assert len(selected) == 1
    assert selected[0].image_id == "image-a"
    assert selected[0].source_row_ids == ["row-a"]
    assert len(selected[0].candidates) == 1
    assert selected[0].candidates[0].source_candidate_index == 0
    assert selected[0].candidates[0].payload["bbox"] == _candidate()["bbox"]


def test_second_pass_outputs_training_views_with_first_pass_evidence(tmp_path):
    store = RecheckStore(tmp_path / "recheck.sqlite3")
    image = GroupedImage(
        image_id="image-a",
        file_path="images/a.jpg",
        source_row_ids=["row-a"],
        candidates=[
            GroupedCandidate(candidate_key="uncertain-key", payload=_candidate(), source_row_ids=["row-a"]),
            GroupedCandidate(candidate_key="uncertain-key-2", payload=_candidate(0.4), source_row_ids=["row-a"]),
        ],
        conflict=False,
        source_rows=[{"source_row_id": "row-a", "source_payload": {"id": "row-a"}}],
    )
    store.create_or_validate_run(
        run_id="first-pass",
        output_dir=tmp_path / "first",
        images=[image],
        provider_order=["gpt"],
        batch_size=10,
        concurrency=1,
        resume=False,
    )
    store.apply_results(
        run_id="first-pass",
        image_ids=["image-a"],
        results={"image-a": {0: _decision(0, "uncertain"), 1: _decision(1, "uncertain")}},
        provider="gpt",
        model="gpt-first",
        sha256_by_image={"image-a": "sha"},
    )
    selected = load_uncertain_images(store, "first-pass")
    store.create_or_validate_run(
        run_id="second-pass",
        output_dir=tmp_path / "second",
        images=selected,
        provider_order=["gpt"],
        batch_size=10,
        concurrency=1,
        resume=False,
        parent_run_id="first-pass",
        run_stage="uncertain_recheck",
        selection="uncertain",
    )
    store.apply_results(
        run_id="second-pass",
        image_ids=["image-a"],
        results={"image-a": {0: _decision(0, "true"), 1: _decision(1, "false")}},
        provider="gpt",
        model="gpt-second",
        sha256_by_image={"image-a": "sha"},
    )

    summary = write_outputs(store, "second-pass", tmp_path / "second")
    candidate_rows = [json.loads(line) for line in (tmp_path / "second" / "candidate_results.jsonl").read_text(encoding="utf-8").splitlines()]
    yoloe_rows = [json.loads(line) for line in (tmp_path / "second" / "yoloe_training_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    qwen_rows = [json.loads(line) for line in (tmp_path / "second" / "qwen_vl_training.jsonl").read_text(encoding="utf-8").splitlines()]
    training_summary = json.loads((tmp_path / "second" / "training_data_summary.json").read_text(encoding="utf-8"))

    assert candidate_rows[0]["first_pass_label"] == "uncertain"
    assert candidate_rows[0]["second_pass_label"] == "true"
    assert candidate_rows[0]["source_candidate_index"] == 0
    assert candidate_rows[0]["parent_run_id"] == "first-pass"
    assert len(yoloe_rows) == 1
    assert yoloe_rows[0]["training_label"] == "true"
    assert yoloe_rows[0]["class_id"] == 0
    assert len(qwen_rows) == 2
    assert {row["training_label"] for row in qwen_rows} == {"true", "false"}
    assert summary["training_views"]["yoloe_positive_rows"] == 1
    assert training_summary["uncertain_excluded_rows"] == 0
    assert json.loads((tmp_path / "second" / "class_map.json").read_text(encoding="utf-8"))["classes"] == {"车": 0}
    assert (tmp_path / "second" / "yolo_labels" / "image-a.txt").read_text(encoding="utf-8") == "0 0.15000000 0.15000000 0.10000000 0.10000000\n"
    image_manifest = [json.loads(line) for line in (tmp_path / "second" / "yolo_image_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert image_manifest[0]["label_rows"] == 1


def test_apply_results_does_not_overwrite_human_confirmed_candidate(tmp_path):
    store = RecheckStore(tmp_path / "recheck.sqlite3")
    image = GroupedImage(
        image_id="image-human",
        file_path="images/human.jpg",
        source_row_ids=["row-human"],
        candidates=[GroupedCandidate(
            candidate_key="human-key",
            payload=_candidate(),
            source_row_ids=["row-human"],
        )],
        conflict=False,
        source_rows=[{"source_row_id": "row-human", "source_payload": {"id": "row-human"}}],
    )
    store.create_or_validate_run(
        run_id="human-run",
        output_dir=tmp_path / "human",
        images=[image],
        provider_order=["gpt"],
        batch_size=1,
        concurrency=1,
        resume=False,
    )
    with store.connect() as connection:
        connection.execute(
            "UPDATE candidates SET human_confirmed=1, auto_label='true', label_source='human_gold', decision_json=? "
            "WHERE run_id=? AND image_id=? AND candidate_index=0",
            (json.dumps({"human_label": "true"}), "human-run", "image-human"),
        )

    store.apply_results(
        run_id="human-run",
        image_ids=["image-human"],
        results={"image-human": {0: _decision(0, "false")}},
        provider="gpt",
        model="gpt-test",
        sha256_by_image={"image-human": "human-sha"},
    )

    row = store.rows_for_output("human-run")["candidates"][0]
    assert row["human_confirmed"] == 1
    assert row["auto_label"] == "true"
    assert row["label_source"] == "human_gold"
    assert json.loads(row["decision_json"]) == {"human_label": "true"}


def test_write_outputs_keeps_policy_ticket_and_human_empty_gold_fields(tmp_path):
    store = RecheckStore(tmp_path / "recheck.sqlite3")
    image = GroupedImage(
        image_id="image-policy",
        file_path="images/policy.jpg",
        source_row_ids=["row-policy"],
        candidates=[GroupedCandidate(
            candidate_key="policy-key",
            payload=_candidate(),
            source_row_ids=["row-policy"],
        )],
        conflict=False,
        source_rows=[{"source_row_id": "row-policy", "source_payload": {"id": "row-policy"}}],
    )
    store.create_or_validate_run(
        run_id="policy-run",
        output_dir=tmp_path / "policy",
        images=[image],
        provider_order=["gpt"],
        batch_size=1,
        concurrency=1,
        resume=False,
    )
    decision = _decision(0, "uncertain")
    decision.update({
        "scene": {"label": "ground", "confidence": 0.95, "reason": "road"},
        "evidence": {
            "red_overlay_occlusion": True,
            "overlay_obscures": True,
            "background_only": False,
            "target_visible": "occluded",
            "occlusion_type": "red_overlay",
        },
    })
    store.apply_results(
        run_id="policy-run",
        image_ids=["image-policy"],
        results={"image-policy": {0: decision}},
        provider="gpt",
        model="gpt-test",
        sha256_by_image={"image-policy": "policy-sha"},
        view_sha256_by_image={"image-policy": {"masked": "masked-sha", "roi_0": "roi-sha"}},
    )

    summary = write_outputs(store, "policy-run", tmp_path / "policy")
    candidate = json.loads((tmp_path / "policy" / "candidate_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    gold = json.loads((tmp_path / "policy" / "gold_review_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert candidate["model_label"] == "uncertain"
    assert candidate["policy"]["reason_code"] == "red_overlay_occlusion_exempt"
    assert candidate["scene"]["label"] == "ground"
    assert candidate["evidence"]["red_overlay_occlusion"] is True
    assert candidate["view_sha256"] == {"masked": "masked-sha", "roi_0": "roi-sha"}
    assert gold["human_label"] is None
    assert gold["presence"] is None
    assert gold["class_match"] is None
    assert gold["box_fit"] is None
    assert gold["model_evidence"]["red_overlay_occlusion"] is True
    assert summary["gold_review"]["human_fields_prefilled"] == 0


def test_pre_llm_geometric_gate_detects_vehicle_tricycle_iou_and_containment():
    from app.services.ktd_recheck import apply_pre_llm_geometric_gate

    candidates = [
        {"target_index": 0, "class_name": "车辆", "confidence": 0.95, "bbox": [0.10, 0.10, 0.50, 0.50]},
        {"target_index": 1, "class_name": "三轮车", "confidence": 0.85, "bbox": [0.11, 0.11, 0.49, 0.49]},
        {"target_index": 2, "class_name": "人员", "confidence": 0.80, "bbox": [0.15, 0.15, 0.25, 0.25]},
    ]
    gated = apply_pre_llm_geometric_gate(candidates)
    assert "cross_class_iou_conflict:vehicle_vs_tricycle" in gated[0]["geometric_warning"]
    assert "cross_class_iou_conflict:vehicle_vs_tricycle" in gated[1]["geometric_warning"]
    assert "part_in_whole_containment_warning" in gated[1]["geometric_warning"]
    assert "part_in_whole_containment_warning" in gated[2]["geometric_warning"]


def test_three_tier_cascade_routing_partitions_candidates_correctly():
    from app.services.ktd_recheck import partition_candidates_by_cascade_tier

    candidates = [
        # High-confidence non-roof without conflict -> Tier 1 auto confirm
        {"target_index": 0, "class_name": "车辆", "confidence": 0.95, "bbox": [0.05, 0.05, 0.15, 0.15]},
        # High-confidence blue_illegal_roof -> must go to Tier 3 VLM for rooftop scene check
        {"target_index": 1, "class_name": "蓝色违建顶棚", "confidence": 0.96, "bbox": [0.30, 0.30, 0.45, 0.45]},
        # Ultra-low confidence -> Tier 2 auto reject
        {"target_index": 2, "class_name": "道路裂缝", "confidence": 0.08, "bbox": [0.60, 0.60, 0.70, 0.70]},
        # Ambiguous confidence -> Tier 3 VLM verify
        {"target_index": 3, "class_name": "人员", "confidence": 0.55, "bbox": [0.75, 0.75, 0.85, 0.85]},
    ]
    buckets = partition_candidates_by_cascade_tier(candidates)
    assert [c["target_index"] for c in buckets["tier1_auto_confirm"]] == [0]
    assert [c["target_index"] for c in buckets["tier2_auto_reject"]] == [2]
    assert [c["target_index"] for c in buckets["tier3_vlm_verify"]] == [1, 3]


def test_scene_dependent_roof_class_is_forced_into_tier3_at_any_confidence():
    """blue_illegal_roof 不得被任何阈值自动裁决，必须 100% 进入 Tier 3。"""
    from app.services.ktd_recheck import evaluate_cascade_tier

    for confidence in (0.0, 0.05, 0.12, 0.5, 0.92, 0.96, 1.0):
        routing = evaluate_cascade_tier(
            {"class_name": "蓝色违建顶棚", "confidence": confidence, "bbox": [0.1, 0.1, 0.5, 0.5]}
        )
        assert routing["tier"] == "tier3_vlm_verify", confidence
        assert routing["requires_vlm"] is True
        assert routing["auto_label"] is None
        assert routing["reason_code"] == "tier3_scene_dependent_forced_vlm"


def test_low_confidence_non_roof_class_still_auto_rejects():
    from app.services.ktd_recheck import evaluate_cascade_tier

    routing = evaluate_cascade_tier(
        {"class_name": "道路裂缝", "confidence": 0.12, "bbox": [0.1, 0.1, 0.5, 0.5]}
    )
    assert routing["tier"] == "tier2_auto_reject"
    assert routing["auto_label"] == "false"


def test_tier2_auto_reject_votes_use_rule_confidence_not_detector_score():
    """A 0.05 detector score must still persist a decisive auto-reject label."""
    from app.services.auto_annotation_service import _synthetic_cascade_decision, decision_from_votes
    from app.services.ktd_recheck import evaluate_cascade_tier

    routing = evaluate_cascade_tier(
        {"class_name": "车辆", "confidence": 0.05, "bbox": [0.1, 0.1, 0.5, 0.5]}
    )
    decision = _synthetic_cascade_decision(0, routing)

    assert decision_from_votes(decision["votes"]) == "false"
    assert all(vote["confidence"] >= 0.6 for vote in decision["votes"])


def test_tier1_default_threshold_is_calibrated_on_real_gold_with_zero_added_fp():
    """回归 F6：默认阈值须与真实检测器分布匹配，且只对刚体类生效。"""
    from app.config import Settings
    from app.services.ktd_recheck import TIER1_AUTO_CONFIRM_THRESHOLD, evaluate_cascade_tier

    assert TIER1_AUTO_CONFIRM_THRESHOLD == 0.85
    assert Settings().tier1_auto_confirm_threshold == 0.85
    # AUTO_ANNOTATION_CASCADE_ROUTING stays off by default; enabling it uses the new value.
    assert Settings().auto_annotation_cascade_routing is False

    for class_name in ("车辆", "人员", "三轮车", "vehicle", "person", "tricycle"):
        routing = evaluate_cascade_tier({"class_name": class_name, "confidence": 0.85})
        assert routing["tier"] == "tier1_auto_confirm", class_name
        assert routing["auto_label"] == "true"

    # Just below the calibrated boundary still needs the L20 VLM.
    routing = evaluate_cascade_tier({"class_name": "车辆", "confidence": 0.8499})
    assert routing["tier"] == "tier3_vlm_verify"


def test_road_crack_can_never_be_auto_confirmed():
    """道路裂缝属纹理歧义类：高分只能进 Tier3，极低分才允许 Tier2。"""
    from app.services.ktd_recheck import evaluate_cascade_tier

    for class_name in ("道路裂缝", "道路裂纹", "road_crack", "crack"):
        for confidence in (0.85, 0.90, 0.95, 0.99, 1.0):
            routing = evaluate_cascade_tier({"class_name": class_name, "confidence": confidence})
            assert routing["tier"] == "tier3_vlm_verify", (class_name, confidence)
            assert routing["auto_label"] is None
            assert routing["reason_code"] == "tier3_road_crack_never_auto_confirmed"

        routing = evaluate_cascade_tier({"class_name": class_name, "confidence": 0.10})
        assert routing["tier"] == "tier2_auto_reject"
        assert routing["auto_label"] == "false"


def test_classes_outside_the_tier1_allowlist_are_never_auto_confirmed():
    from app.services.ktd_recheck import evaluate_cascade_tier

    for class_name in ("光斑", "光伏板", "车牌", "没戴头盔"):
        routing = evaluate_cascade_tier({"class_name": class_name, "confidence": 0.99})
        assert routing["tier"] == "tier3_vlm_verify", class_name
        assert routing["reason_code"] == "tier3_class_outside_tier1_allowlist"


def test_geometric_conflict_downgrades_a_tier1_eligible_candidate():
    from app.services.ktd_recheck import evaluate_cascade_tier

    routing = evaluate_cascade_tier(
        {
            "class_name": "车辆",
            "confidence": 0.99,
            "geometric_warning": "cross_class_iou_conflict:vehicle_vs_tricycle(iou=0.81)",
        }
    )
    assert routing["tier"] == "tier3_vlm_verify"
    assert routing["reason_code"] == "tier3_geometric_conflict"
