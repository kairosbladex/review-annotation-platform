"""Publish stage: render a markdown run report from the state.

Reference implementation: one report per run, same shape as the manual
training reports under docs/. Publishing never deploys anything.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.core.registry import register
from agent.core.state import RunState


@register("render_report")
def render_report(state: RunState) -> RunState:
    params = state["params"]
    output_dir = Path(
        str(params.get("report_dir", "data/agent-runs/{run_id}")).format(run_id=state["run_id"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Run report: {state['run_id']}",
        "",
        f"> generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"- pipeline: `{state['pipeline_id']}` v{state['pipeline_version']}",
        f"- status at generation time: **{state['status']}** (rendered mid-run)",
        "",
        "## Metrics",
        "",
    ]
    for name, value in state.get("metrics", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```json")
        import json

        lines.append(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        lines.append("```")
        lines.append("")
    lines.append("## Gates")
    lines.append("")
    if state.get("gates"):
        for name, gate in state["gates"].items():
            lines.append(f"- `{name}`: {gate}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    if state.get("artifacts"):
        for artifact in state["artifacts"]:
            sha = artifact.get("sha256", "")
            lines.append(
                f"- `{artifact['kind']}` {artifact['uri']}"
                + (f" (sha256 `{sha[:12]}…`)" if sha else "")
            )
    else:
        lines.append("- none")
    lines.append("")
    if state.get("errors"):
        lines.append("## Errors")
        lines.append("")
        for error in state["errors"]:
            lines.append(f"- `{error.get('node')}`: {error.get('error')}")
        lines.append("")

    report = output_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    state["artifacts"].append(
        {
            "kind": "report",
            "uri": f"file://{report.resolve()}",
            "producer": "render_report",
        }
    )
    return state
