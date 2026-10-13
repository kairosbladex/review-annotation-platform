#!/usr/bin/env python3
"""Agent pipeline CLI.

Run from the repo root:

    set -a; source .env.agent.local; set +a
    .venv/bin/python agent/cli.py run agent/pipelines/ktd_dataset_prep.yaml
    .venv/bin/python agent/cli.py resume agent/pipelines/gpu_remote_smoke.yaml --run-id <run_id>

.env.agent.local carries LANGCHAIN_API_KEY (LangSmith tracing; absent means
untraced) and GPU_SSH_* (remote job nodes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import nodes  # noqa: F401  (import registers all nodes)
from agent.core import pipeline as pipeline_mod
from agent.core import tracing
from agent import runner


def _print(state: dict, traced: bool) -> None:
    print(
        json.dumps(
            {
                "run_id": state["run_id"],
                "status": state["status"],
                "tracing": "langsmith" if traced else "disabled",
                "metrics": state.get("metrics", {}),
                "remote_job": state.get("remote_job"),
                "artifacts": len(state.get("artifacts", [])),
                "errors": state.get("errors", []),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run a pipeline YAML")
    run_p.add_argument("pipeline", type=Path)
    run_p.add_argument(
        "--project",
        default="review-annotation-agent",
        help="LangSmith project name",
    )
    resume_p = sub.add_parser("resume", help="resume a crashed or waiting run")
    resume_p.add_argument("pipeline", type=Path)
    resume_p.add_argument("--run-id", required=True)
    resume_p.add_argument(
        "--decision",
        default=None,
        help='JSON resume payload for a human gate, e.g. \'{"approved": true}\'',
    )
    args = parser.parse_args()

    traced = tracing.setup(getattr(args, "project", "review-annotation-agent"))
    if args.command == "run":
        state = runner.run(pipeline_mod.load(args.pipeline))
    elif args.command == "resume":
        decision = json.loads(args.decision) if args.decision else None
        state = runner.resume(
            pipeline_mod.load(args.pipeline), run_id=args.run_id, decision=decision
        )
    else:
        return 2
    _print(state, traced)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
