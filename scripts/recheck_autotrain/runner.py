#!/usr/bin/env python3
"""复检微调自动迭代的宿主机执行器：由 systemd 用户定时器每 2 分钟调用一次，按任务状态推进一步。

硬限制（Agent 无法改变）：
  - 只用 L20 GPU-c977647c；同一时间只有一个任务（训练或评测）；
  - 训练容器 24GB 内存 / 8 核，评测容器各 7GB / 8 核，均断网、结束即删、以 uid 1000 运行；
  - 启动前检查显存与宿主机可用内存；训练超过 4 小时、评测超过 5 小时即停止并记失败；
  - 存在 STOP 文件时停止本系统所有容器并拒绝推进。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

LIVE = Path(os.environ.get("AUTOTRAIN_LIVE", "/data/workspace/jzheng/recheck-autotrain/live"))
STATE = LIVE / "state"
RUNS = LIVE / "runs"
SRC = LIVE / "src"
ASSETS = LIVE / "assets"
GPU = "GPU-c977647c-baa1-4f69-87ba-1e384cfa6de3"
BASE_MODEL = "/data/workspace/jzheng/yoloe_service/data/ktd-qwen-vl-8b-instruct"
DATASET = "/data/workspace/jzheng/ktd-clean-source-qwen-20260930-v2/data"
EVAL_SETS = {"hard": "/data/workspace/jzheng/claude-vsg-resolution-ab-20260929",
             "random": "/data/workspace/jzheng/claude-vsg-random-ab-20260929"}
TRAIN_IMAGE = "ktd-qwen-runtime:clean-source-20260925-v1"
EVAL_IMAGE = "yoloe_service_test:vehicle-surface-v3-20260929-v1"
EVAL_ENV = str(ASSETS / "container.env")  # 清空 MQTT/MinIO/数据库等业务连接的环境变量
TRAIN_NEED_GPU_MIB, EVAL_NEED_GPU_MIB = 26000, 16000
TRAIN_MEM_GB, EVAL_MEM_GB = 24, 7
TRAIN_TIMEOUT_H, EVAL_TIMEOUT_H = 4, 5
TZ = dt.timezone(dt.timedelta(hours=8))
UID = f"{os.getuid()}:{os.getgid()}"


def now() -> str:
    return dt.datetime.now(TZ).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{now()} {message}", flush=True)


def sh(args: list[str], check: bool = True) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])}… 失败：{result.stderr.strip()[:500]}")
    return result.stdout


def gpu_free_mib() -> int:
    return int(sh(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", GPU]).strip())


def host_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def containers(job_id: str | None = None) -> list[str]:
    label = f"autotrain.job={job_id}" if job_id else "autotrain.job"
    return sh(["docker", "ps", "-q", "--filter", f"label={label}"]).split()


def save(job: dict, status: str | None = None, **fields) -> None:
    if status:
        job["status"] = status
        job.setdefault("history", []).append({"at": now(), "status": status, **fields})
    job.update(fields)
    path = STATE / "jobs" / f"{job['job_id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    log(f"{job['job_id']} -> {job['status']} {fields if fields else ''}")


def elapsed_hours(job: dict, key: str) -> float:
    started = dt.datetime.fromisoformat(job[key])
    return (dt.datetime.now(TZ) - started).total_seconds() / 3600


def base_run(name: str, job_id: str, mem_gb: int) -> list[str]:
    return ["docker", "run", "-d", "--rm", "--name", name, "--label", "owner=agent",
            "--label", f"autotrain.job={job_id}", f"--memory={mem_gb}g", f"--memory-swap={mem_gb}g",
            "--cpus=8", "--network", "none", "--gpus", f"device={GPU}", "--user", UID]


def start_training(job: dict) -> None:
    run = RUNS / job["job_id"]
    (run / "home").mkdir(parents=True, exist_ok=True)
    config = job["config"]
    args = ["--model", "/models/qwen3vl", "--train", "/workspace/data/train.jsonl",
            "--val", "/workspace/data/validation.jsonl", "--provenance-root", "/workspace/data",
            "--output-dir", "/workspace/out/training-run", "--epochs", str(config["epochs"]),
            "--max-length", "2048", "--min-pixels", "50176", "--max-pixels", "200704", "--min-free-gb", "20",
            "--train-sampling", config["train_sampling"], "--learning-rate", str(config["learning_rate"]),
            "--seed", "42"]
    if config["train_sampling"] == "capped_stratum":
        args += ["--stratum-weight-cap", str(config["stratum_weight_cap"])]
    inner = ("python3 /workspace/code/qwen3vl_qlora_train.py " + " ".join(args)
             + " > /workspace/out/train.log 2>&1; echo $? > /workspace/out/train.exit")
    sh(base_run(f"autotrain-train-{job['job_id']}", job["job_id"], TRAIN_MEM_GB)
       + ["-e", "HOME=/workspace/out/home", "-e", "PYTHONUNBUFFERED=1", "-w", "/workspace/code",
          "-v", f"{BASE_MODEL}:/models/qwen3vl:ro", "-v", f"{DATASET}:/workspace/data:ro",
          "-v", f"{SRC}/train:/workspace/code:ro", "-v", f"{run}:/workspace/out",
          "--entrypoint", "sh", TRAIN_IMAGE, "-c", inner])
    save(job, "training", training_started_at=now())


def check_training(job: dict) -> None:
    run = RUNS / job["job_id"]
    if containers(job["job_id"]):
        if elapsed_hours(job, "training_started_at") > TRAIN_TIMEOUT_H:
            for cid in containers(job["job_id"]):
                sh(["docker", "stop", "-t", "30", cid], check=False)
            save(job, "failed", failure="训练超时")
        return
    code = (run / "train.exit").read_text().strip() if (run / "train.exit").exists() else "missing"
    adapter = run / "training-run" / "adapter_model.safetensors"
    if code == "0" and adapter.exists():
        digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
        save(job, "trained", adapter_sha256=digest, trained_at=now())
    else:
        tail = (run / "train.log").read_text(errors="replace")[-800:] if (run / "train.log").exists() else ""
        save(job, "failed", failure=f"训练退出码 {code}", log_tail=tail)


def start_evaluation(job: dict) -> None:
    run = RUNS / job["job_id"]
    for shard in (0, 1):
        for name in EVAL_SETS:
            (run / "eval" / f"out-{name}-q{shard}").mkdir(parents=True, exist_ok=True)
        sh(base_run(f"autotrain-eval-{job['job_id']}-{shard}", job["job_id"], EVAL_MEM_GB)
           + ["--env-file", EVAL_ENV,
              "-v", f"{BASE_MODEL}:/opt/clean-source/base:ro",
              "-v", f"{run}/training-run:/opt/clean-source/adapter:ro", "-v", f"{SRC}/eval:/src:ro",
              "-v", f"{EVAL_SETS['hard']}:/hard:ro", "-v", f"{EVAL_SETS['random']}:/random:ro",
              "-v", f"{run}/eval:/out", "--entrypoint", "sh", EVAL_IMAGE,
              "/src/run_eval_shard.sh", str(shard), "2", GPU, job["adapter_sha256"]])
    save(job, "evaluating", evaluation_started_at=now())


def finish_evaluation(job: dict) -> None:
    run = RUNS / job["job_id"]
    ev = run / "eval"
    done = [ev / f"out-{s}-q{k}" / "done.json" for s in EVAL_SETS for k in (0, 1)]
    if not all(p.exists() for p in done):
        save(job, "failed", failure="评测分片未全部完成", missing=[str(p) for p in done if not p.exists()])
        return
    py = sys.executable
    parse_errors = 0
    for name, rows in (("hard", "rows.jsonl"), ("random", "sample.jsonl")):
        merged = ev / name / "merged_predictions.jsonl"
        sh([py, str(SRC / "analysis/merge_adapter_predictions.py"), "--base-predictions",
            *[str(p) for p in sorted((ASSETS / "base" / name).glob("*.jsonl"))],
            "--adapter-predictions", *[str(ev / f"out-{name}-q{k}" / "predictions.jsonl") for k in (0, 1)],
            "--out", str(merged)])
        sh([py, str(SRC / "analysis/analyze_resolution_ab.py"), "--predictions", str(merged),
            "--inputs", str(ASSETS / "inputs" / f"{name}-inputs.jsonl"), "--rows", str(ASSETS / "inputs" / f"{name}-{rows}"),
            "--labels", *[str(p) for p in sorted((ASSETS / "labels" / name).glob("*.csv"))], "--out", str(ev / name)])
        parse_errors += json.loads((ev / name / "report.json").read_text())["parse_errors"]
    sys.path.insert(0, str(SRC / "analysis"))
    from metrics import compare_to_v1  # noqa: E402
    metrics = compare_to_v1(ASSETS / "v1" / "random-per_candidate.csv", ev / "random" / "per_candidate.csv",
                            ASSETS / "v1" / "hard-per_candidate.csv", ev / "hard" / "per_candidate.csv", parse_errors)
    (STATE / "metrics").mkdir(exist_ok=True)
    metrics_path = STATE / "metrics" / f"{job['job_id']}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    sh([py, str(SRC / "analysis/make_samples.py"), "--metrics", str(metrics_path),
        "--out-dir", str(STATE / "samples" / job["job_id"])])
    save(job, "evaluated", evaluated_at=now(), passed_all_gates=metrics["passed_all_gates"])


def stop_everything(reason: str) -> None:
    for cid in containers():
        sh(["docker", "stop", "-t", "30", cid], check=False)
    for path in (STATE / "jobs").glob("*.json"):
        job = json.loads(path.read_text(encoding="utf-8"))
        if job["status"] in {"queued", "training", "eval_requested", "evaluating"}:
            save(job, "stopped", failure=reason)


def write_gpu_status(ok: bool, reason: str) -> None:
    (STATE / "status").mkdir(parents=True, exist_ok=True)
    (STATE / "status" / "gpu.json").write_text(json.dumps(
        {"gpu_ok": ok, "reason": reason, "gpu": GPU, "checked_at": now()}, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    if (STATE / "STOP").exists():
        stop_everything("STOP 文件")
        write_gpu_status(False, "STOP 文件存在")
        return
    jobs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((STATE / "jobs").glob("*.json"))]
    active = [j for j in jobs if j["status"] in {"queued", "training", "eval_requested", "evaluating"}]
    free, avail = gpu_free_mib(), host_available_gb()
    busy = any(j["status"] in {"training", "evaluating"} for j in active)
    write_gpu_status(busy or free >= EVAL_NEED_GPU_MIB,
                     f"本系统任务运行中" if busy else f"c977 空闲显存 {free}MiB，宿主机可用内存 {avail:.0f}GB")
    if not active:
        return
    job = active[0]
    status = job["status"]
    if status == "queued":
        if free < TRAIN_NEED_GPU_MIB or avail < 2 * TRAIN_MEM_GB:
            log(f"{job['job_id']} 等待资源：显存 {free}MiB，内存 {avail:.0f}GB")
            return
        start_training(job)
    elif status == "training":
        check_training(job)
    elif status == "eval_requested":
        if free < EVAL_NEED_GPU_MIB or avail < 4 * EVAL_MEM_GB:
            log(f"{job['job_id']} 评测等待资源：显存 {free}MiB，内存 {avail:.0f}GB")
            return
        start_evaluation(job)
    elif status == "evaluating":
        if containers(job["job_id"]):
            if elapsed_hours(job, "evaluation_started_at") > EVAL_TIMEOUT_H:
                for cid in containers(job["job_id"]):
                    sh(["docker", "stop", "-t", "30", cid], check=False)
                save(job, "failed", failure="评测超时")
            return
        finish_evaluation(job)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # 记录后退出，下次定时再试
        log(f"runner 异常：{type(exc).__name__}: {exc}")
        raise
