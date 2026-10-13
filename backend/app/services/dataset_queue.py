from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


V2_DATASET_TAG = "v2"


@dataclass(frozen=True)
class DatasetQueueIndex:
    """Read-only mapping from platform candidates to the V2 queue.

    The platform-map contains duplicate aliases for provenance. Only its
    representative candidate is a queue item, so duplicate source images do
    not expand the V2 worklist in the UI.
    """

    v2_canonical_by_candidate: dict[str, str]
    path: Path | None = None

    @classmethod
    def from_jsonl(cls, path: str | Path | None) -> "DatasetQueueIndex":
        if not path:
            return cls({})
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            return cls({}, resolved)

        canonical_by_candidate: dict[str, str] = {}
        for line_number, raw_line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"V2 queue row {line_number} must be an object")
            canonical_id = str(row.get("canonical_id") or "").strip()
            representative = row.get("representative")
            candidate_id = str(representative.get("candidate_id") or "").strip() if isinstance(representative, dict) else ""
            if not canonical_id or not candidate_id:
                raise ValueError(f"V2 queue row {line_number} is missing canonical_id or representative.candidate_id")
            existing = canonical_by_candidate.get(candidate_id)
            if existing and existing != canonical_id:
                raise ValueError(f"candidate {candidate_id} belongs to multiple V2 canonical boxes")
            canonical_by_candidate[candidate_id] = canonical_id
        return cls(canonical_by_candidate, resolved)

    def canonical_id_for(self, candidate_id: str) -> str | None:
        return self.v2_canonical_by_candidate.get(candidate_id)

    def is_v2(self, candidate_id: str) -> bool:
        return candidate_id in self.v2_canonical_by_candidate
