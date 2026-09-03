"""Snapshot auditor — contract section 9.

Verifies: file hashes vs manifest/SHA256SUMS, split image-SHA disjointness,
input files carry no gold/note/reviewer/prior fields, split gold ⊆ gold_records.

Usage: python3 -m tools.prompt_eval_data.auditor <snapshot_dir>
Exit code 0 = AUDIT PASS, 1 = at least one check failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .common import sha256_bytes
from .exporter import INPUT_ALLOWED_KEYS, SPLIT_GOLD_KEYS


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def audit(snapshot_dir: str) -> list[tuple[str, bool, str]]:
    root = Path(snapshot_dir)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    manifest_path = root / "snapshot_manifest.json"
    if not manifest_path.exists():
        check("manifest_exists", False, "snapshot_manifest.json missing")
        return checks
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check("manifest_exists", True, manifest.get("snapshot_id", ""))

    # 1. Every manifest file entry matches bytes on disk.
    mismatches = []
    for entry in manifest.get("files", []):
        path = root / entry["path"]
        if not path.exists():
            mismatches.append(f"{entry['path']}: missing")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != entry["sha256"] or len(data) != entry["bytes"]:
            mismatches.append(f"{entry['path']}: sha/size mismatch")
    check("manifest_file_hashes", not mismatches, "; ".join(mismatches))

    # 2. SHA256SUMS agrees with disk.
    sums_path = root / "SHA256SUMS"
    bad_lines = []
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sha, rel = line.split("  ", 1)
            target = root / rel
            if not target.exists() or sha256_bytes(target.read_bytes()) != sha:
                bad_lines.append(rel)
    check("sha256sums", sums_path.exists() and not bad_lines, "; ".join(bad_lines))

    # 3. Split image-SHA sets are disjoint.
    selection_inputs = _read_jsonl(root / "selection_inputs.jsonl")
    locked_inputs = _read_jsonl(root / "locked_test_inputs.jsonl")
    sel_shas = {row["image_sha256"] for row in selection_inputs}
    lock_shas = {row["image_sha256"] for row in locked_inputs}
    overlap = sel_shas & lock_shas
    check("split_image_sha_disjoint", not overlap, f"overlap={len(overlap)}")

    # 4. Input files contain only the allowed model-input keys (no gold/note/reviewer/priors).
    allowed = set(INPUT_ALLOWED_KEYS)
    key_violations = []
    for name, rows in (("selection_inputs.jsonl", selection_inputs), ("locked_test_inputs.jsonl", locked_inputs)):
        for row in rows:
            extra = set(row) - allowed
            if extra:
                key_violations.append(f"{name}:{sorted(extra)}")
    check("inputs_schema_clean", not key_violations, "; ".join(key_violations[:5]))

    # 5. Split gold files are minimal and a subset of gold_records.
    gold_cids = {row["canonical_id"] for row in _read_jsonl(root / "gold_records.jsonl")}
    gold_key_ok = True
    subset_ok = True
    for name in ("selection_gold.jsonl", "locked_test_gold.jsonl"):
        for row in _read_jsonl(root / name):
            if set(row) != set(SPLIT_GOLD_KEYS):
                gold_key_ok = False
            if row["canonical_id"] not in gold_cids:
                subset_ok = False
    check("split_gold_schema", gold_key_ok)
    check("split_gold_subset_of_gold_records", subset_ok)

    # 6. Quarantined canonical ids never appear in gold or splits.
    quarantine_cids = {row["canonical_id"] for row in _read_jsonl(root / "quarantine.jsonl")}
    leak = quarantine_cids & gold_cids
    check("quarantine_not_in_gold", not leak, f"leak={len(leak)}")

    return checks


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python3 -m tools.prompt_eval_data.auditor <snapshot_dir>", file=sys.stderr)
        return 2
    checks = audit(args[0])
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
