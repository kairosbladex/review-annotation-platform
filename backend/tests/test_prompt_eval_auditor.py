import builtins
import hashlib
import json
from pathlib import Path

from tools.prompt_eval_data.auditor import audit


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


def _refresh_integrity(root: Path) -> Path:
    member_names = sorted(
        path.name
        for path in root.iterdir()
        if path.name not in {"SHA256SUMS", "snapshot_manifest.json"}
    )
    sums = "".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in member_names
    ).encode()
    (root / "SHA256SUMS").write_bytes(sums)
    snapshot_id = "pg2-" + hashlib.sha256(
        "".join(
            f"{name}:{hashlib.sha256((root / name).read_bytes()).hexdigest()}\n"
            for name in member_names
        ).encode()
    ).hexdigest()[:12]
    manifest_names = member_names + ["SHA256SUMS"]
    manifest = {
        "schema_version": "prompt-gold/2",
        "snapshot_id": snapshot_id,
        "readiness": {},
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                "bytes": (root / name).stat().st_size,
            }
            for name in manifest_names
        ],
    }
    (root / "snapshot_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    target = root.parent / snapshot_id
    if target != root:
        root.rename(target)
    return target


def _snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "staging"
    root.mkdir()
    selection_input = {
        "canonical_id": "sel-1",
        "class_id": 1,
        "class_name": "person",
        "image_sha256": "a" * 64,
        "split_group_key": "group-0",
    }
    locked_input = {
        "canonical_id": "lock-1",
        "class_id": 2,
        "class_name": "vehicle",
        "image_sha256": "b" * 64,
        "split_group_key": "group-5",
    }
    selection_gold = {
        "canonical_id": "sel-1",
        "class_id": 1,
        "class_name": "person",
        "label": "true",
    }
    locked_gold = {
        "canonical_id": "lock-1",
        "class_id": 2,
        "class_name": "vehicle",
        "label": "false",
    }
    members = {
        "authoring.jsonl": [],
        "inference_records.jsonl": [],
        "selection_inputs.jsonl": [selection_input],
        "locked_test_inputs.jsonl": [locked_input],
        "selection_gold.jsonl": [selection_gold],
        "locked_test_gold.jsonl": [locked_gold],
        "gold_records.jsonl": [
            selection_gold | {"split": "selection"},
            locked_gold | {"split": "locked_test"},
        ],
        "quarantine.jsonl": [],
    }
    for name, rows in members.items():
        (root / name).write_bytes(_jsonl(rows))
    (root / "source_readiness.json").write_text("{}", encoding="utf-8")
    return _refresh_integrity(root)


def _failed(root: Path, *, protected: bool = False) -> set[str]:
    return {
        name
        for name, ok, _ in audit(str(root), include_protected_gold=protected)
        if not ok
    }


def test_auditor_rejects_duplicate_selection_gold(tmp_path):
    root = _snapshot(tmp_path)
    row = json.loads((root / "selection_gold.jsonl").read_text().splitlines()[0])
    (root / "selection_gold.jsonl").write_bytes(_jsonl([row, row]))
    root = _refresh_integrity(root)

    assert "selection_gold_unique" in _failed(root)


def test_auditor_rejects_selection_class_mismatch_without_protected_read(tmp_path):
    root = _snapshot(tmp_path)
    row = json.loads((root / "selection_gold.jsonl").read_text().splitlines()[0])
    row["class_name"] = "vehicle"
    (root / "selection_gold.jsonl").write_bytes(_jsonl([row]))
    root = _refresh_integrity(root)

    assert "selection_gold_identity_matches_inputs" in _failed(root)


def test_auditor_rejects_input_in_wrong_split_file(tmp_path):
    root = _snapshot(tmp_path)
    row = json.loads((root / "selection_inputs.jsonl").read_text().splitlines()[0])
    row["split_group_key"] = "group-5"
    (root / "selection_inputs.jsonl").write_bytes(_jsonl([row]))
    root = _refresh_integrity(root)

    assert "split_role_matches_assignment" in _failed(root)


def test_protected_audit_rejects_label_drift_from_full_gold(tmp_path):
    root = _snapshot(tmp_path)
    row = json.loads((root / "selection_gold.jsonl").read_text().splitlines()[0])
    row["label"] = "false"
    (root / "selection_gold.jsonl").write_bytes(_jsonl([row]))
    root = _refresh_integrity(root)

    assert not _failed(root)
    assert "split_gold_matches_full_gold" in _failed(root, protected=True)


def test_protected_audit_rejects_duplicate_full_gold(tmp_path):
    root = _snapshot(tmp_path)
    rows = [
        json.loads(line)
        for line in (root / "gold_records.jsonl").read_text().splitlines()
    ]
    (root / "gold_records.jsonl").write_bytes(_jsonl(rows + [rows[0]]))
    root = _refresh_integrity(root)

    assert "full_gold_unique" in _failed(root, protected=True)


def test_protected_audit_rejects_locked_class_mismatch(tmp_path):
    root = _snapshot(tmp_path)
    row = json.loads((root / "locked_test_gold.jsonl").read_text().splitlines()[0])
    row["class_id"] = 99
    (root / "locked_test_gold.jsonl").write_bytes(_jsonl([row]))
    root = _refresh_integrity(root)

    failed = _failed(root, protected=True)
    assert "locked_gold_identity_matches_inputs" in failed
    assert "split_gold_matches_full_gold" in failed


def test_protected_audit_rejects_full_gold_split_drift(tmp_path):
    root = _snapshot(tmp_path)
    rows = [
        json.loads(line)
        for line in (root / "gold_records.jsonl").read_text().splitlines()
    ]
    rows[0]["split"] = "locked_test"
    rows[1]["split"] = "selection"
    (root / "gold_records.jsonl").write_bytes(_jsonl(rows))
    root = _refresh_integrity(root)

    assert "full_gold_split_matches_membership" in _failed(root, protected=True)


def test_default_audit_never_opens_protected_gold(tmp_path, monkeypatch):
    root = _snapshot(tmp_path)
    protected = {
        "authoring.jsonl",
        "gold_records.jsonl",
        "locked_test_gold.jsonl",
        "quarantine.jsonl",
    }
    original_path_open = Path.open
    original_builtin_open = builtins.open

    def guarded_path_open(path, *args, **kwargs):
        if path.name in protected:
            raise AssertionError(f"protected file opened: {path.name}")
        return original_path_open(path, *args, **kwargs)

    def guarded_builtin_open(file, *args, **kwargs):
        if Path(file).name in protected:
            raise AssertionError(f"protected file opened: {Path(file).name}")
        return original_builtin_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)

    results = audit(str(root))

    assert all(ok for _, ok, _ in results)
    assert (
        "protected_label_content_skipped",
        True,
        "not opened; recorded hashes compared only",
    ) in results


def test_auditor_rejects_empty_integrity_catalogs(tmp_path):
    root = tmp_path / "pg2-empty"
    root.mkdir()
    (root / "SHA256SUMS").write_text("", encoding="utf-8")
    (root / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "prompt-gold/2",
                "snapshot_id": root.name,
                "files": [],
            }
        ),
        encoding="utf-8",
    )

    failed = _failed(root)

    assert "manifest_members_complete" in failed
    assert "sha256sums_catalog" in failed
    assert "snapshot_id_content_addressed" in failed


def test_auditor_rejects_manifest_path_escape_without_opening_it(tmp_path):
    root = _snapshot(tmp_path)
    outside = root.parent / "outside.jsonl"
    outside.write_text("sensitive", encoding="utf-8")
    manifest_path = root / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../outside.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert "manifest_entries_valid" in _failed(root)


def test_auditor_rejects_canonical_id_reused_across_splits(tmp_path):
    root = _snapshot(tmp_path)
    locked_input = json.loads(
        (root / "locked_test_inputs.jsonl").read_text().splitlines()[0]
    )
    locked_input["canonical_id"] = "sel-1"
    locked_gold = json.loads(
        (root / "locked_test_gold.jsonl").read_text().splitlines()[0]
    )
    locked_gold["canonical_id"] = "sel-1"
    full_gold = [
        json.loads(line)
        for line in (root / "gold_records.jsonl").read_text().splitlines()
    ]
    full_gold = [full_gold[1] | {"canonical_id": "sel-1"}]
    (root / "locked_test_inputs.jsonl").write_bytes(_jsonl([locked_input]))
    (root / "locked_test_gold.jsonl").write_bytes(_jsonl([locked_gold]))
    (root / "gold_records.jsonl").write_bytes(_jsonl(full_gold))
    root = _refresh_integrity(root)

    assert "split_input_canonical_disjoint" in _failed(root)
    assert "split_gold_canonical_disjoint" in _failed(root, protected=True)
