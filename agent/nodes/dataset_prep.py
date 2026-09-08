"""Wrap scripts/prepare_ktd_gpu_dataset.prepare as a pipeline node.

Idempotent for a fixed run_id: the output dir is derived from run_id and
prepare() rewrites its manifests deterministically from the same inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.core.registry import register
from agent.core.state import RunState

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@register("dataset_prep")
def dataset_prep(state: RunState) -> RunState:
    from scripts.prepare_ktd_gpu_dataset import prepare

    params = state["params"]
    output_dir = Path(str(params["output_dir"]).format(run_id=state["run_id"]))
    summary = prepare(
        manifest_path=Path(params["manifest"]),
        class_map_path=Path(params["class_map"]),
        output_dir=output_dir,
        image_base_url=str(params.get("image_base_url", "")),
        download=bool(params.get("download", False)),
        val_ratio=float(params.get("val_ratio", 0.2)),
        seed=int(params.get("seed", 42)),
        timeout=float(params.get("timeout", 60.0)),
    )
    state["artifacts"].append(
        {
            "kind": "dataset",
            "uri": f"file://{output_dir.resolve()}",
            "producer": "dataset_prep",
            "meta": summary,
        }
    )
    state["metrics"]["dataset"] = summary
    return state
