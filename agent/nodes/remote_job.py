"""Generic remote-job nodes over SSH (e.g. the AI-lingxi GPU server).

Job layout on the remote host, under <remote_root>/<run_id>/:
  run.sh    rendered command wrapper (fixed template from pipeline params)
  job.log   stdout+stderr of the job
  JOB.pid   pid of the wrapper process
  JOB.exit  exit code, written by run.sh when the job finishes

Submit is idempotent per run_id: a live job is attached to, a finished job
is detected via JOB.exit, and nothing is ever resubmitted blindly. Monitor
blocks locally but survives local crashes: after a resume it re-reads the
remote markers instead of assuming anything about the job.
"""

from __future__ import annotations

import base64
import hashlib
import shlex
import time
from pathlib import Path

from agent.core.remote import RemoteExecutor
from agent.core.registry import register
from agent.core.state import RunState

_RUN_SH = """#!/bin/bash
set -u
cd "$(dirname "$0")"
( {command} ) > job.log 2>&1
echo $? > JOB.exit
"""


def _format(template: str, state: RunState, **extra: object) -> str:
    scalars = {
        key: value
        for key, value in state["params"].items()
        if isinstance(value, (str, int, float))
    }
    return template.format(run_id=state["run_id"], **scalars, **extra)


def parse_nvidia_smi_csv(output: str) -> list[dict[str, object]]:
    """Parse 4-column or 5-column nvidia-smi CSV output into structured GPU dicts."""
    gpus: list[dict[str, object]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            index, name, free, used, total = parts
            free_mib = int(free)
            used_mib = int(used)
            total_mib = int(total)
        elif len(parts) == 4:
            index, name, used, total = parts
            used_mib = int(used)
            total_mib = int(total)
            free_mib = total_mib - used_mib
        else:
            raise ValueError(f"unexpected nvidia-smi CSV line: {line!r}")
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_free_mib": free_mib,
                "memory_used_mib": used_mib,
                "memory_total_mib": total_mib,
            }
        )
    return gpus


def select_gpu_device(
    gpus: list[dict[str, object]],
    *,
    gpu_index: object = None,
    min_free_gpu_mib: int = 1024,
    preferred_gpu_arch: str | None = None,
    hostname: str = "remote",
) -> dict[str, object]:
    """Select and validate a target GPU, supporting automatic idle L20 selection."""
    arch_filter = (preferred_gpu_arch or "").strip().lower()
    if gpu_index is not None and str(gpu_index).strip().lower() not in {"", "auto"}:
        target_idx = int(gpu_index)  # type: ignore[arg-type]
        gpu = next((g for g in gpus if int(g["index"]) == target_idx), None)
        if gpu is None:
            raise RuntimeError(f"gpu {target_idx} not found on {hostname}")
        if arch_filter and arch_filter not in str(gpu["name"]).lower():
            raise RuntimeError(
                f"gpu {target_idx} on {hostname} is {gpu['name']!r}, expected arch {preferred_gpu_arch!r}"
            )
        free = int(gpu.get("memory_free_mib", int(gpu["memory_total_mib"]) - int(gpu["memory_used_mib"])))
        if free < min_free_gpu_mib:
            raise RuntimeError(
                f"gpu {target_idx} on {hostname} has {free} MiB free, need >= {min_free_gpu_mib} MiB"
            )
        return gpu

    eligible = [
        g
        for g in gpus
        if int(g.get("memory_free_mib", int(g["memory_total_mib"]) - int(g["memory_used_mib"])))
        >= min_free_gpu_mib
    ]
    if arch_filter:
        arch_matches = [g for g in eligible if arch_filter in str(g["name"]).lower()]
        if not arch_matches:
            raise RuntimeError(
                f"no GPU matching arch {preferred_gpu_arch!r} with >= {min_free_gpu_mib} MiB free on {hostname}"
            )
        eligible = arch_matches
    else:
        # Prefer L20 (>40GB) automatically when available among eligible GPUs
        l20_matches = [
            g
            for g in eligible
            if "l20" in str(g["name"]).lower() or int(g["memory_total_mib"]) >= 40000
        ]
        if l20_matches:
            eligible = l20_matches

    if not eligible:
        raise RuntimeError(
            f"no GPU on {hostname} has >= {min_free_gpu_mib} MiB free"
        )
    eligible.sort(
        key=lambda g: (
            int(g.get("memory_free_mib", int(g["memory_total_mib"]) - int(g["memory_used_mib"]))),
            -int(g["index"]),
        ),
        reverse=True,
    )
    return eligible[0]


@register("remote_preflight")
def remote_preflight(state: RunState) -> RunState:
    params = state["params"]
    exe = RemoteExecutor.from_env()
    try:
        hostname = exe.run_ok("hostname").strip()
        smi_out = exe.run_ok(
            "nvidia-smi --query-gpu=index,name,memory.free,memory.used,memory.total "
            "--format=csv,noheader,nounits"
        )
        gpus = parse_nvidia_smi_csv(smi_out)
        device = params.get("gpu_index")
        min_free = int(params.get("min_free_gpu_mib", 1024))
        preferred_arch = params.get("preferred_gpu_arch")
        selected_gpu = None
        if device is not None or preferred_arch is not None:
            selected_gpu = select_gpu_device(
                gpus,
                gpu_index=device,
                min_free_gpu_mib=min_free,
                preferred_gpu_arch=str(preferred_arch) if preferred_arch else None,
                hostname=hostname,
            )
            params["gpu_index"] = int(selected_gpu["index"])
        remote_root = _format(str(params["remote_root"]), state)
        exe.run_ok(f"mkdir -p {shlex.quote(remote_root)}")
        state["metrics"]["remote"] = {
            "hostname": hostname,
            "gpus": gpus,
            "selected_gpu": selected_gpu,
        }
    finally:
        exe.close()
    return state


@register("remote_submit")
def remote_submit(state: RunState) -> RunState:
    params = state["params"]
    exe = RemoteExecutor.from_env()
    try:
        remote_root = _format(str(params["remote_root"]), state)
        run_dir = f"{remote_root}/{state['run_id']}"
        qdir = shlex.quote(run_dir)
        probe = exe.run(
            f"if [ -f {qdir}/JOB.exit ]; then echo finished; "
            f"elif [ -f {qdir}/JOB.pid ] && kill -0 $(cat {qdir}/JOB.pid) 2>/dev/null; "
            f"then echo running; else echo absent; fi"
        ).stdout.strip()
        if probe == "absent":
            command = _format(str(params["command"]), state, remote_run_dir=run_dir)
            script_b64 = base64.b64encode(_RUN_SH.format(command=command).encode()).decode()
            exe.run_ok(
                f"mkdir -p {qdir} && echo {script_b64} | base64 -d > {qdir}/run.sh "
                f"&& chmod +x {qdir}/run.sh"
            )
            pid = (
                exe.run_ok(f"cd {qdir} && nohup ./run.sh >/dev/null 2>&1 </dev/null & echo $!")
                .strip()
                .splitlines()[-1]
            )
            exe.run_ok(f"echo {pid} > {qdir}/JOB.pid")
            probe = "submitted"
        state["remote_job"] = {"run_dir": run_dir, "state": probe}
    finally:
        exe.close()
    return state


@register("remote_monitor")
def remote_monitor(state: RunState) -> RunState:
    params = state["params"]
    run_dir = state["remote_job"]["run_dir"]
    poll = float(params.get("poll_seconds", 30))
    max_wait = float(params.get("max_wait_seconds", 4 * 3600))
    progress_rel = params.get("progress_file")
    exe = RemoteExecutor.from_env()
    deadline = time.monotonic() + max_wait
    try:
        while True:
            exit_code = exe.run(f"cat {run_dir}/JOB.exit 2>/dev/null || true").stdout.strip()
            if exit_code:
                rc = int(exit_code.splitlines()[-1])
                if rc != 0:
                    tail = exe.run(f"tail -20 {run_dir}/job.log 2>/dev/null || true").stdout
                    raise RuntimeError(f"remote job failed rc={rc}\n{tail}")
                break
            alive = exe.run(
                f"kill -0 $(cat {run_dir}/JOB.pid) 2>/dev/null && echo yes || echo no"
            ).stdout.strip()
            if alive == "no":
                time.sleep(min(poll, 5))  # grace: JOB.exit may land between checks
                retry = exe.run(f"cat {run_dir}/JOB.exit 2>/dev/null || true").stdout.strip()
                if not retry:
                    tail = exe.run(f"tail -20 {run_dir}/job.log 2>/dev/null || true").stdout
                    raise RuntimeError(f"remote job died without exit marker\n{tail}")
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"remote job still running after {max_wait}s: {run_dir}")
            time.sleep(poll)
        if progress_rel:
            lines = exe.run(
                f"wc -l < {run_dir}/{progress_rel} 2>/dev/null || echo 0"
            ).stdout.strip()
            state["metrics"]["remote_progress_lines"] = int(lines or 0)
        state["remote_job"]["state"] = "finished"
    finally:
        exe.close()
    return state


@register("remote_sync")
def remote_sync(state: RunState) -> RunState:
    params = state["params"]
    run_dir = state["remote_job"]["run_dir"]
    local_dir = Path(_format(str(params["local_dir"]), state))
    files = params.get("sync_files") or []
    exe = RemoteExecutor.from_env()
    try:
        for item in files:
            rel = str(item["path"])
            local_path = local_dir / rel
            exe.get_file(f"{run_dir}/{rel}", local_path)
            local_sha = hashlib.sha256(local_path.read_bytes()).hexdigest()
            remote_sha = exe.run_ok(f"sha256sum {run_dir}/{rel} | cut -d' ' -f1").strip()
            if local_sha != remote_sha:
                raise RuntimeError(f"sha256 mismatch after sync: {rel}")
            state["artifacts"].append(
                {
                    "kind": str(item.get("kind", "artifact")),
                    "uri": f"file://{local_path.resolve()}",
                    "sha256": local_sha,
                    "producer": "remote_sync",
                }
            )
    finally:
        exe.close()
    return state
