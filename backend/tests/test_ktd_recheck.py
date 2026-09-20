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
