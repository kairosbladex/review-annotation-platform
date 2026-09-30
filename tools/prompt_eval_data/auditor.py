"""Snapshot auditor — contract section 9.

Verifies: file hashes vs manifest/SHA256SUMS, split image-SHA disjointness,
input files carry no gold/note/reviewer/prior fields, and selection-gold alignment.
Protected label-bearing files are not opened unless explicitly requested;
default mode compares only their recorded manifest/SHA256SUMS digests.

Usage: python3 -m tools.prompt_eval_data.auditor <snapshot_dir>
       python3 -m tools.prompt_eval_data.auditor --include-protected-gold <snapshot_dir>
Exit code 0 = AUDIT PASS, 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

from .common import sha256_bytes
from .exporter import INPUT_ALLOWED_KEYS, SPLIT_GOLD_KEYS
from .splits import LOCKED_TEST, SELECTION, assign_split

GOLD_LABELS = {"true", "false"}
PROTECTED_LABEL_FILES = frozenset(
    {
        "authoring.jsonl",
        "gold_records.jsonl",
        "locked_test_gold.jsonl",
        "quarantine.jsonl",
    }
)
BASE_SNAPSHOT_FILES = frozenset(
    {
        "SHA256SUMS",
        "authoring.jsonl",
        "gold_records.jsonl",
        "inference_records.jsonl",
        "locked_test_gold.jsonl",
        "locked_test_inputs.jsonl",
        "quarantine.jsonl",
        "selection_gold.jsonl",
        "selection_inputs.jsonl",
    }
)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _duplicates(rows: list[dict], key: str) -> list[object]:
    seen: set[object] = set()
    duplicates: set[object] = set()
    for row in rows:
        value = row.get(key)
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates, key=str)


def _identity_map(rows: list[dict]) -> dict[object, tuple[object, object]]:
    return {
        row.get("canonical_id"): (row.get("class_id"), row.get("class_name"))
        for row in rows
    }


def _gold_map(rows: list[dict]) -> dict[object, tuple[object, object, object]]:
    return {
        row.get("canonical_id"): (
            row.get("class_id"),
            row.get("class_name"),
            row.get("label"),
        )
        for row in rows
    }


def _safe_member_path(root: Path, name: object) -> Path | None:
    if not isinstance(name, str) or not name:
        return None
    rel = PurePosixPath(name)
    if rel.is_absolute() or len(rel.parts) != 1 or rel.parts[0] in {".", ".."}:
        return None
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        return None
    return path


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def audit(
    snapshot_dir: str, *, include_protected_gold: bool = False
) -> list[tuple[str, bool, str]]:
    root = Path(snapshot_dir)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    manifest_path = root / "snapshot_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        check("manifest_exists", False, "snapshot_manifest.json missing or symlinked")
        return checks
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        check("manifest_object", False, "snapshot manifest must be an object")
        return checks
    check("manifest_exists", True, manifest.get("snapshot_id", ""))

    # 1. Validate both integrity catalogs before following any member path.
    schema_version = manifest.get("schema_version")
    check(
        "schema_version_supported",
        schema_version in {"prompt-gold/1", "prompt-gold/2"},
        str(schema_version),
    )
    entries: dict[str, dict] = {}
    entry_errors: list[str] = []
    manifest_entries = manifest.get("files")
    if not isinstance(manifest_entries, list):
        entry_errors.append("files is not a list")
        manifest_entries = []
    for entry in manifest_entries:
        if not isinstance(entry, dict):
            entry_errors.append("entry is not an object")
            continue
        name = entry.get("path")
        path = _safe_member_path(root, name)
        if path is None:
            entry_errors.append(f"unsafe path: {name!r}")
            continue
        if name in entries:
            entry_errors.append(f"duplicate path: {name}")
            continue
        if not _valid_sha256(entry.get("sha256")):
            entry_errors.append(f"invalid sha256: {name}")
        if (
            not isinstance(entry.get("bytes"), int)
            or isinstance(entry.get("bytes"), bool)
            or entry["bytes"] < 0
        ):
            entry_errors.append(f"invalid bytes: {name}")
        entries[name] = entry
    check("manifest_entries_valid", not entry_errors, "; ".join(entry_errors[:5]))

    required_files = set(BASE_SNAPSHOT_FILES)
    if schema_version == "prompt-gold/2":
        required_files.add("source_readiness.json")
    actual_files = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "snapshot_manifest.json"
    }
    missing_required = sorted(required_files - set(entries))
    catalog_difference = sorted(actual_files ^ set(entries))
    check(
        "manifest_members_complete",
        not missing_required and not catalog_difference,
        f"missing={missing_required} catalog_difference={catalog_difference}",
    )

    sums_path = root / "SHA256SUMS"
    sums: dict[str, str] = {}
    sums_errors: list[str] = []
    if sums_path.is_file() and not sums_path.is_symlink():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                sums_errors.append("malformed line")
                continue
            sha, name = parts
            if _safe_member_path(root, name) is None or name == "SHA256SUMS":
                sums_errors.append(f"unsafe path: {name!r}")
                continue
            if name in sums:
                sums_errors.append(f"duplicate path: {name}")
                continue
            if not _valid_sha256(sha):
                sums_errors.append(f"invalid sha256: {name}")
                continue
            sums[name] = sha
    else:
        sums_errors.append("SHA256SUMS missing or symlinked")
    expected_sum_names = set(entries) - {"SHA256SUMS"}
    if "SHA256SUMS" not in entries:
        sums_errors.append("SHA256SUMS missing from manifest")
    if set(sums) != expected_sum_names:
        sums_errors.append(
            f"path set mismatch: missing={sorted(expected_sum_names - set(sums))} "
            f"extra={sorted(set(sums) - expected_sum_names)}"
        )
    recorded_mismatches = sorted(
        name
        for name in set(sums) & set(entries)
        if sums[name] != entries[name].get("sha256")
    )
    if recorded_mismatches:
        sums_errors.append(f"manifest hash mismatch: {recorded_mismatches}")
    check("sha256sums_catalog", not sums_errors, "; ".join(sums_errors[:5]))

    prefix = {"prompt-gold/1": "pg1", "prompt-gold/2": "pg2"}.get(schema_version)
    expected_snapshot_id = None
    if prefix is not None and not sums_errors:
        digest_input = "".join(f"{name}:{sums[name]}\n" for name in sorted(sums))
        expected_snapshot_id = f"{prefix}-{sha256_bytes(digest_input.encode())[:12]}"
    snapshot_id = manifest.get("snapshot_id")
    check(
        "snapshot_id_content_addressed",
        expected_snapshot_id is not None
        and snapshot_id == expected_snapshot_id
        and root.name == expected_snapshot_id,
        f"manifest={snapshot_id} directory={root.name} expected={expected_snapshot_id}",
    )

    structural_checks = {
        "schema_version_supported",
        "manifest_entries_valid",
        "manifest_members_complete",
        "sha256sums_catalog",
        "snapshot_id_content_addressed",
    }
    if any(not ok for name, ok, _ in checks if name in structural_checks):
        return checks

    # 2. Verify unprotected bytes. Protected files are not opened by default;
    # their two recorded hashes must agree, and their byte sizes are stat-only.
    mismatches = []
    for name, entry in entries.items():
        path = root / name
        if not path.is_file():
            mismatches.append(f"{name}: missing")
            continue
        if name in PROTECTED_LABEL_FILES and not include_protected_gold:
            if path.stat().st_size != entry["bytes"]:
                mismatches.append(f"{name}: size mismatch")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != entry["sha256"] or len(data) != entry["bytes"]:
            mismatches.append(f"{name}: sha/size mismatch")
    check("manifest_file_hashes", not mismatches, "; ".join(mismatches))

    # 3. Split source-group and image-SHA sets are disjoint.
    selection_inputs = _read_jsonl(root / "selection_inputs.jsonl")
    locked_inputs = _read_jsonl(root / "locked_test_inputs.jsonl")
    input_duplicates = {
        "selection": _duplicates(selection_inputs, "canonical_id"),
        "locked_test": _duplicates(locked_inputs, "canonical_id"),
    }
    check(
        "split_inputs_unique",
        not any(input_duplicates.values()),
        json.dumps(input_duplicates, sort_keys=True),
    )
    selection_input_cids = {row.get("canonical_id") for row in selection_inputs}
    locked_input_cids = {row.get("canonical_id") for row in locked_inputs}
    canonical_overlap = selection_input_cids & locked_input_cids
    check(
        "split_input_canonical_disjoint",
        not canonical_overlap,
        f"overlap={sorted(canonical_overlap)[:5]}",
    )
    sel_shas = {row["image_sha256"] for row in selection_inputs}
    lock_shas = {row["image_sha256"] for row in locked_inputs}
    overlap = sel_shas & lock_shas
    check("split_image_sha_disjoint", not overlap, f"overlap={len(overlap)}")
    sel_groups = {row.get("split_group_key", row["image_sha256"]) for row in selection_inputs}
    lock_groups = {row.get("split_group_key", row["image_sha256"]) for row in locked_inputs}
    group_overlap = sel_groups & lock_groups
    check("split_source_group_disjoint", not group_overlap, f"overlap={len(group_overlap)}")
    role_mismatches = [
        f"selection:{row.get('canonical_id')}"
        for row in selection_inputs
        if assign_split(row.get("split_group_key") or row["image_sha256"]) != SELECTION
    ] + [
        f"locked_test:{row.get('canonical_id')}"
        for row in locked_inputs
        if assign_split(row.get("split_group_key") or row["image_sha256"]) != LOCKED_TEST
    ]
    check(
        "split_role_matches_assignment",
        not role_mismatches,
        f"mismatches={role_mismatches[:5]}",
    )

    # 4. Input files contain only the allowed model-input keys (no gold/note/reviewer/priors).
    allowed = set(INPUT_ALLOWED_KEYS)
    key_violations = []
    identity_violations = []
    for name, rows in (("selection_inputs.jsonl", selection_inputs), ("locked_test_inputs.jsonl", locked_inputs)):
        for row in rows:
            extra = set(row) - allowed
            if extra:
                key_violations.append(f"{name}:{sorted(extra)}")
            if any(row.get(key) in (None, "") for key in ("canonical_id", "class_id", "class_name")):
                identity_violations.append(str(row.get("canonical_id")))
    check("inputs_schema_clean", not key_violations, "; ".join(key_violations[:5]))
    check(
        "inputs_identity_complete",
        not identity_violations,
        f"violations={identity_violations[:5]}",
    )

    # 5. Default audit parses selection gold only. Full/locked gold remain hash-only
    # unless the caller explicitly enters the protected final-evaluation boundary.
    selection_gold = _read_jsonl(root / "selection_gold.jsonl")
    selection_gold_duplicates = _duplicates(selection_gold, "canonical_id")
    selection_gold_cids = {row.get("canonical_id") for row in selection_gold}
    check(
        "selection_gold_schema",
        all(
            set(row) == set(SPLIT_GOLD_KEYS) and row.get("label") in GOLD_LABELS
            for row in selection_gold
        ),
    )
    check(
        "selection_gold_unique",
        not selection_gold_duplicates,
        f"duplicates={selection_gold_duplicates}",
    )
    check(
        "selection_gold_matches_inputs",
        selection_gold_cids == selection_input_cids,
        f"gold={len(selection_gold_cids)} inputs={len(selection_input_cids)}",
    )
    selection_identity_mismatches = sorted(
        canonical_id
        for canonical_id in selection_gold_cids & selection_input_cids
        if _identity_map(selection_gold)[canonical_id]
        != _identity_map(selection_inputs)[canonical_id]
    )
    check(
        "selection_gold_identity_matches_inputs",
        not selection_identity_mismatches,
        f"mismatches={selection_identity_mismatches[:5]}",
    )
    if include_protected_gold:
        full_gold = _read_jsonl(root / "gold_records.jsonl")
        locked_gold = _read_jsonl(root / "locked_test_gold.jsonl")
        quarantine = _read_jsonl(root / "quarantine.jsonl")
        full_gold_duplicates = _duplicates(full_gold, "canonical_id")
        locked_gold_duplicates = _duplicates(locked_gold, "canonical_id")
        locked_gold_cids = {row.get("canonical_id") for row in locked_gold}
        check(
            "full_gold_unique",
            not full_gold_duplicates,
            f"duplicates={full_gold_duplicates}",
        )
        check(
            "full_gold_identity_and_label_valid",
            all(
                row.get("canonical_id") not in (None, "")
                and row.get("class_id") not in (None, "")
                and row.get("class_name") not in (None, "")
                and row.get("label") in GOLD_LABELS
                and row.get("split") in {SELECTION, LOCKED_TEST}
                for row in full_gold
            ),
        )
        check(
            "protected_gold_schema",
            all(
                set(row) == set(SPLIT_GOLD_KEYS) and row.get("label") in GOLD_LABELS
                for row in locked_gold
            ),
        )
        check(
            "locked_gold_unique",
            not locked_gold_duplicates,
            f"duplicates={locked_gold_duplicates}",
        )
        check(
            "locked_gold_matches_inputs",
            locked_gold_cids == locked_input_cids,
            f"gold={len(locked_gold_cids)} inputs={len(locked_input_cids)}",
        )
        locked_identity_mismatches = sorted(
            canonical_id
            for canonical_id in locked_gold_cids & locked_input_cids
            if _identity_map(locked_gold)[canonical_id]
            != _identity_map(locked_inputs)[canonical_id]
        )
        check(
            "locked_gold_identity_matches_inputs",
            not locked_identity_mismatches,
            f"mismatches={locked_identity_mismatches[:5]}",
        )
        split_gold = selection_gold + locked_gold
        split_gold_cids = selection_gold_cids | locked_gold_cids
        full_gold_cids = {row.get("canonical_id") for row in full_gold}
        split_gold_overlap = selection_gold_cids & locked_gold_cids
        check(
            "split_gold_canonical_disjoint",
            not split_gold_overlap,
            f"overlap={sorted(split_gold_overlap)[:5]}",
        )
        check(
            "split_gold_covers_full_gold",
            split_gold_cids == full_gold_cids,
            f"split={len(split_gold_cids)} full={len(full_gold_cids)}",
        )
        mismatched_gold = sorted(
            canonical_id
            for canonical_id in split_gold_cids & full_gold_cids
            if _gold_map(split_gold)[canonical_id]
            != _gold_map(full_gold)[canonical_id]
        )
        check(
            "split_gold_matches_full_gold",
            not mismatched_gold,
            f"mismatches={mismatched_gold[:5]}",
        )
        expected_split = {
            **{canonical_id: SELECTION for canonical_id in selection_gold_cids},
            **{canonical_id: LOCKED_TEST for canonical_id in locked_gold_cids},
        }
        split_metadata_mismatches = sorted(
            row.get("canonical_id")
            for row in full_gold
            if row.get("canonical_id") in expected_split
            and row.get("split") != expected_split[row.get("canonical_id")]
        )
        check(
            "full_gold_split_matches_membership",
            not split_metadata_mismatches,
            f"mismatches={split_metadata_mismatches[:5]}",
        )
        quarantine_cids = {
            row.get("canonical_id") for row in quarantine if isinstance(row, dict)
        }
        split_input_cids = selection_input_cids | locked_input_cids
        quarantine_leak = quarantine_cids & split_input_cids
        check(
            "quarantine_not_in_split_inputs",
            not quarantine_leak,
            f"leak={len(quarantine_leak)}",
        )
    else:
        check(
            "protected_label_content_skipped",
            True,
            "not opened; recorded hashes compared only",
        )

    # 6. Readiness is a hashed snapshot member and agrees with the manifest.
    readiness_path = root / "source_readiness.json"
    if readiness_path.exists():
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        check("source_readiness_matches_manifest", readiness == manifest.get("readiness"))
    else:
        check("source_readiness_legacy_snapshot", manifest.get("schema_version") == "prompt-gold/1")

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir")
    parser.add_argument(
        "--include-protected-gold",
        action="store_true",
        help="parse gold_records and locked_test_gold during authorized final evaluation",
    )
    args = parser.parse_args(argv)
    checks = audit(
        args.snapshot_dir,
        include_protected_gold=args.include_protected_gold,
    )
    failed = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    print("AUDIT PASS" if failed == 0 else f"AUDIT FAIL ({failed} checks)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
