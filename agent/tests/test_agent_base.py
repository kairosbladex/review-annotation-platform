"""Base-contract tests for the agent foundation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent import nodes  # noqa: F401  (registers built-in nodes)
from agent.core import registry
from agent.core.pipeline import PipelineDef, load
from agent.core.state import RunState
from agent import runner


def _pipeline(names: list[str], params: dict | None = None) -> PipelineDef:
    return PipelineDef(pipeline_id="test", version="1", nodes=names, params=params or {})


def _state(**overrides) -> RunState:
    state: RunState = {
        "run_id": "test-run",
        "pipeline_id": "test",
        "pipeline_version": "1",
        "params": {},
        "artifacts": [],
        "metrics": {},
        "gates": {},
        "errors": [],
        "status": "running",
    }
    state.update(overrides)
    return state


def test_registry_duplicate_and_unknown():
    @registry.register("test_node_registry")
    def node(state: RunState) -> RunState:
        return state

    with pytest.raises(ValueError, match="duplicate"):
        registry.register("test_node_registry")(node)
    with pytest.raises(ValueError, match="unknown node"):
        registry.get("does-not-exist")


def test_pipeline_loader_validation(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text('pipeline_id: x\nversion: "1"\nnodes: []\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty nodes"):
        load(bad)


def test_runner_fail_fast(tmp_path: Path):
    marker = tmp_path / "marker.txt"

    @registry.register("test_fail_first")
    def first(state: RunState) -> RunState:
        raise RuntimeError("boom")

    @registry.register("test_fail_never")
    def never(state: RunState) -> RunState:
        marker.write_text("reached", encoding="utf-8")
        return state

    final = runner.run(
        _pipeline(["test_fail_first", "test_fail_never"]), checkpoint_db=tmp_path / "cp.sqlite3"
    )
    assert final["status"] == "failed"
    assert final["errors"][0]["node"] == "test_fail_first"
    assert not marker.exists()


def test_resume_after_node_failure_starts_fresh_attempt(tmp_path: Path):
    counter = tmp_path / "counter.txt"
    flag = tmp_path / "flag"
    db = tmp_path / "cp.sqlite3"

    @registry.register("test_fresh_counted")
    def counted(state: RunState) -> RunState:
        count = int(counter.read_text() or 0) if counter.exists() else 0
        counter.write_text(str(count + 1), encoding="utf-8")
        return state

    @registry.register("test_fresh_gate")
    def gated(state: RunState) -> RunState:
        if not flag.exists():
            raise RuntimeError("not ready")
        return state

    pipeline = _pipeline(["test_fresh_counted", "test_fresh_gate"])
    first = runner.run(pipeline, run_id="retry-me", checkpoint_db=db)
    assert first["status"] == "failed"

    flag.write_text("ok", encoding="utf-8")
    second = runner.resume(pipeline, run_id="retry-me", checkpoint_db=db)
    assert second["status"] == "completed"
    # fresh attempt: idempotent nodes re-run (remote-style nodes attach instead)
    assert counter.read_text() == "2"


def test_crash_resume_skips_completed_nodes(tmp_path: Path):
    counter = tmp_path / "counter.txt"
    flag = tmp_path / "flag"
    db = tmp_path / "cp.sqlite3"

    @registry.register("test_crash_counted")
    def counted(state: RunState) -> RunState:
        count = int(counter.read_text() or 0) if counter.exists() else 0
        counter.write_text(str(count + 1), encoding="utf-8")
        return state

    @registry.register("test_crash_node")
    def crashing(state: RunState) -> RunState:
        if not flag.exists():
            raise KeyboardInterrupt("simulated process kill")  # not caught by the guard
        return state

    pipeline = _pipeline(["test_crash_counted", "test_crash_node"])
    with pytest.raises(KeyboardInterrupt):
        runner.run(pipeline, run_id="crash-me", checkpoint_db=db)
    assert counter.read_text() == "1"

    flag.write_text("ok", encoding="utf-8")
    resumed = runner.resume(pipeline, run_id="crash-me", checkpoint_db=db)
    assert resumed["status"] == "completed"
    assert counter.read_text() == "1"  # completed node skipped from checkpoint


def test_approval_gate_interrupt_and_decisions(tmp_path: Path):
    db = tmp_path / "cp.sqlite3"
    marker = tmp_path / "after.txt"

    @registry.register("test_gate_after")
    def after(state: RunState) -> RunState:
        marker.write_text("reached", encoding="utf-8")
        return state

    pipeline = _pipeline(
        ["approval_gate", "test_gate_after"], {"gate_name": "ship", "gate_message": "ship it?"}
    )

    waiting = runner.run(pipeline, run_id="gate-run", checkpoint_db=db)
    assert waiting["status"] == "waiting_human"
    assert not marker.exists()

    approved = runner.resume(
        pipeline,
        run_id="gate-run",
        decision={"approved": True, "decided_by": "wxl", "note": "ok"},
        checkpoint_db=db,
    )
    assert approved["status"] == "completed"
    assert approved["gates"]["ship"]["passed"] is True
    assert approved["gates"]["ship"]["decided_by"] == "wxl"
    assert marker.exists()

    rejected_run = runner.run(pipeline, run_id="gate-run-2", checkpoint_db=db)
    assert rejected_run["status"] == "waiting_human"
    rejected = runner.resume(
        pipeline,
        run_id="gate-run-2",
        decision={"approved": False, "decided_by": "wxl", "note": "not good enough"},
        checkpoint_db=db,
    )
    assert rejected["status"] == "failed"
    assert "not good enough" in rejected["errors"][0]["error"]


def test_ingest_files_dedup_and_quarantine(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("hello", encoding="utf-8")
    (source / "b.txt").write_text("hello", encoding="utf-8")  # duplicate content
    (source / "c.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (source / "empty.txt").write_bytes(b"")  # quarantine

    state = _state(
        params={"source_dir": str(source), "output_dir": str(tmp_path / "out" / "{run_id}")}
    )
    final = registry.get("ingest_files")(state)
    metrics = final["metrics"]["ingest"]
    assert metrics == {"items": 2, "duplicates": 1, "quarantined": 1}
    manifest = Path(final["artifacts"][-1]["uri"][len("file://"):])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert {row["kind"] for row in rows} == {"text", "image"}


def test_persist_store_idempotent(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [{"item_id": "a", "kind": "text", "sha256": "1"}, {"item_id": "b", "kind": "text"}]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    db = tmp_path / "store.sqlite3"
    state = _state(
        artifacts=[{"kind": "ingest_manifest", "uri": f"file://{manifest}", "producer": "t"}],
        params={"store_db": str(db)},
    )
    registry.get("persist_store")(state)
    again = registry.get("persist_store")(state)
    assert again["metrics"]["persist"] == {"written": 2, "store_total": 2}
    with sqlite3.connect(str(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2


def test_llm_analyze_quarantines_failures(tmp_path: Path, monkeypatch):
    item_file = tmp_path / "note.txt"
    item_file.write_text("content", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    items = [
        {"item_id": "ok", "kind": "text", "uri": f"file://{item_file}"},
        {"item_id": "bad", "kind": "text", "uri": f"file://{item_file}"},
    ]
    manifest.write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")

    def fake_call(content, *, timeout):
        if '"item_id": "ok"' in str(content) or "ok" in str(content[0]["text"]):
            return {"label": True}
        raise RuntimeError("llm exploded")

    monkeypatch.setattr("agent.nodes.llm_analyze._call_llm", fake_call)
    state = _state(
        params={
            "manifest": str(manifest),
            "prompt_template": "judge {item_id}",
            "output_dir": str(tmp_path / "out" / "{run_id}"),
        }
    )
    final = registry.get("llm_analyze")(state)
    assert final["metrics"]["analyze"] == {"ok": 1, "review_queue": 1}
    review = (tmp_path / "out" / "test-run" / "review_queue.jsonl").read_text()
    assert "llm exploded" in review


def test_dataset_audit_finds_blockers(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "classes.txt").write_text("0\t车\n1\t人\n", encoding="utf-8")
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "labels" / "val" / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    payload = b"same-image-bytes"
    (root / "images" / "train" / "x.jpg").write_bytes(payload)
    (root / "images" / "val" / "y.jpg").write_bytes(payload)  # cross-split leakage

    state = _state(params={"dataset_dir": str(root), "confirm_task": "detect", "on_block": "continue"})
    final = registry.get("dataset_audit")(state)
    gate = final["gates"]["dataset_audit"]
    assert gate["passed"] is False
    assert any("zero-positive" in b for b in gate["blockers"])
    assert any("leakage" in b for b in gate["blockers"])

    # fix: label class 1, distinct val image
    (root / "labels" / "val" / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "images" / "val" / "y.jpg").write_bytes(b"different-bytes")
    fixed = registry.get("dataset_audit")(
        _state(params={"dataset_dir": str(root), "confirm_task": "detect", "on_block": "fail"})
    )
    assert fixed["gates"]["dataset_audit"]["passed"] is True


def test_dataset_audit_requires_task_confirmation(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    (root / "classes.txt").write_text("0\t车\n", encoding="utf-8")
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "images" / "train" / "x.jpg").write_bytes(b"img")

    state = _state(params={"dataset_dir": str(root), "on_block": "continue"})
    final = registry.get("dataset_audit")(state)
    assert final["gates"]["dataset_audit"]["passed"] is False
    assert any("confirm_task" in b for b in final["gates"]["dataset_audit"]["blockers"])


def test_remote_gpu_selection_prefers_idle_l20_in_4t4_2l20_cluster():
    from agent.nodes.remote_job import parse_nvidia_smi_csv, select_gpu_device

    smi_output = "\n".join(
        [
            "0, Tesla T4, 14800, 309, 15109",
            "1, Tesla T4, 14500, 609, 15109",
            "2, Tesla T4, 14200, 909, 15109",
            "3, Tesla T4, 14900, 209, 15109",
            "4, NVIDIA L20, 4120, 41948, 46068",
            "5, NVIDIA L20, 45200, 868, 46068",
        ]
    )
    gpus = parse_nvidia_smi_csv(smi_output)
    assert len(gpus) == 6
    selected_l20 = select_gpu_device(
        gpus, min_free_gpu_mib=40960, preferred_gpu_arch="L20"
    )
    assert selected_l20 is not None
    assert selected_l20["index"] == 5
    assert "L20" in selected_l20["name"]

    selected_t4 = select_gpu_device(
        gpus, min_free_gpu_mib=12288, preferred_gpu_arch="T4"
    )
    assert selected_t4 is not None
    assert selected_t4["index"] == 3


