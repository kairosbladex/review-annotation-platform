"""复检微调模型自动迭代：供 Hermes 调用的受控 MCP 工具。

Hermes 只能通过这些工具读状态、提交实验和写报告；真正的训练与评测由宿主机上的 runner 执行，
资源限制（单卡、内存/CPU 上限、同时只跑一个任务）写死在 runner 中，本服务只写请求文件。

目录（AUTOTRAIN_ROOT）：
  STOP                        存在即停止：拒绝一切新提交与评测请求
  config/experiment_space.json 允许的实验参数与固定项（人工维护）
  config/policy.json           每日上限、截止日期（人工维护）
  jobs/<id>.json               任务状态（runner 更新 status）
  metrics/<id>.json            runner 写入的对比 v1 结果
  samples/<id>/<kind>.jpg|json runner 生成的错判样本拼图
  reports/<id>.md  ledger.jsonl  proposals.jsonl  audit.jsonl   Agent 写入
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(os.environ.get("AUTOTRAIN_ROOT", "/work"))
TZ = dt.timezone(dt.timedelta(hours=8))
ACTIVE = {"queued", "training", "eval_requested", "evaluating"}
SAMPLE_KINDS = {"real_car_regressed", "negative_still_released"}
mcp = FastMCP("autotrain")


def _now() -> dt.datetime:
    return dt.datetime.now(TZ)


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _audit(tool: str, **fields) -> None:
    _append(ROOT / "audit.jsonl", {"at": _now().isoformat(timespec="seconds"), "tool": tool, **fields})


def _jobs() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "jobs").glob("*.json"))]


def _job(job_id: str) -> dict:
    if not re.fullmatch(r"exp-\d{3}", job_id or ""):
        raise ValueError("job_id 格式应为 exp-NNN")
    path = ROOT / "jobs" / f"{job_id}.json"
    if not path.exists():
        raise ValueError(f"没有任务 {job_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _stop_reason() -> str | None:
    if (ROOT / "STOP").exists():
        return "STOP 文件存在：" + (ROOT / "STOP").read_text(encoding="utf-8").strip()[:200]
    deadline = _read_json(ROOT / "config/policy.json", {}).get("deadline")
    if deadline and _now().date().isoformat() > deadline:
        return f"已过截止日期 {deadline}"
    return None


def _config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True, ensure_ascii=False)


@mcp.tool()
def get_state() -> dict:
    """每轮第一步调用。返回是否要求停止、GPU 是否可用、各任务所处阶段、今日已提交数和最近台账。"""
    jobs = _jobs()
    policy = _read_json(ROOT / "config/policy.json", {})
    today = _now().date().isoformat()
    gpu = _read_json(ROOT / "status/gpu.json", {"gpu_ok": False, "reason": "runner 尚未写入 GPU 状态"})
    ledger = [json.loads(l) for l in (ROOT / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()] if (ROOT / "ledger.jsonl").exists() else []
    result = {
        "stop_requested": _stop_reason() is not None,
        "stop_reason": _stop_reason(),
        "gpu_ok": bool(gpu.get("gpu_ok")),
        "gpu_detail": gpu,
        "running_jobs": [j["job_id"] for j in jobs if j["status"] in ACTIVE],
        "trained_not_evaluated": [j["job_id"] for j in jobs if j["status"] == "trained"],
        "evaluated_not_reported": [j["job_id"] for j in jobs if j["status"] == "evaluated" and not j.get("reported")],
        "failed_jobs": [j["job_id"] for j in jobs if j["status"] == "failed"],
        "submitted_today": sum(1 for j in jobs if j.get("submitted_at", "").startswith(today)),
        "daily_limit": policy.get("daily_training_limit", 3),
        "ledger_tail": ledger[-5:],
    }
    _audit("get_state", stop=result["stop_requested"], running=result["running_jobs"])
    return result


@mcp.tool()
def job_status(job_id: str) -> dict:
    """查看某个任务的状态与进度。"""
    job = _job(job_id)
    _audit("job_status", job_id=job_id, status=job["status"])
    return job


@mcp.tool()
def list_experiment_space() -> dict:
    """列出允许调整的参数、固定不变的设置，以及已经跑过的配置（相同配置不能重复提交）。"""
    space = _read_json(ROOT / "config/experiment_space.json", {})
    tried = [{"job_id": j["job_id"], "config": j["config"], "status": j["status"]} for j in _jobs()]
    _audit("list_experiment_space")
    return {**space, "tried": tried}


@mcp.tool()
def submit_training(train_sampling: str, epochs: int, learning_rate: float, stratum_weight_cap: float,
                    hypothesis: str) -> dict:
    """提交一次训练。各参数必须取 list_experiment_space 中 knobs 列出的值；
    stratum_weight_cap 只在 train_sampling=capped_stratum 时有意义，其他抽样方式必须填 0。
    hypothesis 写明改了哪个参数、预期哪项数字怎么变。"""
    config = {"train_sampling": train_sampling, "epochs": epochs, "learning_rate": learning_rate,
              "stratum_weight_cap": stratum_weight_cap}
    reason = _stop_reason()
    if reason:
        _audit("submit_training", rejected=reason)
        return {"accepted": False, "reason": reason}
    space = _read_json(ROOT / "config/experiment_space.json", {})
    knobs = space.get("knobs", {})
    problems = [f"不允许的参数 {k}" for k in config if k not in knobs]
    problems += [f"{k} 的取值 {v!r} 不在允许范围 {knobs[k]}" for k, v in config.items() if k in knobs and v not in knobs[k]]
    problems += [f"缺少参数 {k}" for k in knobs if k not in config]
    if (train_sampling == "capped_stratum") != (stratum_weight_cap > 0):
        problems.append("只有 capped_stratum 需要 stratum_weight_cap>0，其他抽样方式填 0")
    jobs = _jobs()
    if any(j["status"] in ACTIVE for j in jobs):
        problems.append("已有任务在运行")
    if any(_config_key(j["config"]) == _config_key(config) for j in jobs):
        problems.append("相同配置已经跑过")
    policy = _read_json(ROOT / "config/policy.json", {})
    today = _now().date().isoformat()
    if sum(1 for j in jobs if j.get("submitted_at", "").startswith(today)) >= policy.get("daily_training_limit", 3):
        problems.append("今日训练次数已达上限")
    if len(hypothesis.strip()) < 10:
        problems.append("hypothesis 需写明改动与预期")
    if problems:
        _audit("submit_training", rejected=problems, config=config)
        return {"accepted": False, "reason": problems}
    job_id = f"exp-{len(jobs):03d}"
    job = {"job_id": job_id, "config": config, "hypothesis": hypothesis.strip()[:1000], "status": "queued",
           "submitted_at": _now().isoformat(timespec="seconds"), "reported": False, "history": []}
    (ROOT / "jobs").mkdir(parents=True, exist_ok=True)
    (ROOT / "jobs" / f"{job_id}.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    _audit("submit_training", job_id=job_id, config=config)
    return {"accepted": True, "job_id": job_id}


@mcp.tool()
def run_evaluation(job_id: str) -> dict:
    """对已训练完成（status=trained）的任务申请评测；runner 会在随机流量 200 与楼顶难例 108 上评测并对比 v1。"""
    reason = _stop_reason()
    job = _job(job_id)
    if reason:
        return {"accepted": False, "reason": reason}
    if job["status"] != "trained":
        return {"accepted": False, "reason": f"当前状态为 {job['status']}，只有 trained 可以申请评测"}
    job["status"] = "eval_requested"
    job["history"].append({"at": _now().isoformat(timespec="seconds"), "status": "eval_requested"})
    (ROOT / "jobs" / f"{job_id}.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    _audit("run_evaluation", job_id=job_id)
    return {"accepted": True}


@mcp.tool()
def get_evaluation(job_id: str) -> dict:
    """读取已评测任务与 v1 的对比结果（数字由程序计算，报告只能引用这里的数字）。"""
    job = _job(job_id)
    metrics = _read_json(ROOT / "metrics" / f"{job_id}.json")
    _audit("get_evaluation", job_id=job_id, available=metrics is not None)
    if metrics is None:
        return {"available": False, "status": job["status"]}
    return {"available": True, "config": job["config"], "hypothesis": job.get("hypothesis"), **metrics}


@mcp.tool()
def view_error_samples(job_id: str, kind: str) -> dict:
    """取错判样本拼图。kind 为 real_car_regressed（v1 自动确认、本次未确认的真车）或
    negative_still_released（本次仍被自动确认的误检）。返回 image_path，请用 vision_analyze 查看该图片。
    每行一个样本，从左到右：截图上下文、截图局部、高清原图上下文、高清原图局部。"""
    if kind not in SAMPLE_KINDS:
        raise ValueError(f"kind 只能是 {sorted(SAMPLE_KINDS)}")
    _job(job_id)
    meta = _read_json(ROOT / "samples" / job_id / f"{kind}.json")
    _audit("view_error_samples", job_id=job_id, kind=kind, available=meta is not None)
    if meta is None:
        return {"available": False}
    return {"available": True, "image_path": str(ROOT / "samples" / job_id / f"{kind}.jpg"), **meta}


@mcp.tool()
def write_report(job_id: str, markdown: str) -> dict:
    """保存本次实验的中文报告，并把任务标记为已报告。"""
    job = _job(job_id)
    if job["status"] != "evaluated":
        return {"saved": False, "reason": "只有已评测的任务可以写报告"}
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / f"{job_id}.md").write_text(markdown[:20000], encoding="utf-8")
    job["reported"] = True
    (ROOT / "jobs" / f"{job_id}.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    _audit("write_report", job_id=job_id, chars=len(markdown))
    return {"saved": True}


@mcp.tool()
def update_ledger(job_id: str, passed: bool, failed_gates: str, key_numbers: str, next_hypothesis: str) -> dict:
    """在实验台账追加一条记录。failed_gates 用逗号分隔（全部通过写“无”）；key_numbers 抄 get_evaluation 的关键数字；
    next_hypothesis 写下一步打算改什么、为什么。"""
    job = _job(job_id)
    _append(ROOT / "ledger.jsonl", {"at": _now().isoformat(timespec="seconds"), "job_id": job_id, "config": job["config"],
                                    "passed": passed, "failed_gates": failed_gates[:500],
                                    "key_numbers": key_numbers[:1500], "next_hypothesis": next_hypothesis[:1500]})
    _audit("update_ledger", job_id=job_id)
    return {"saved": True}


@mcp.tool()
def propose_playbook_change(proposal: str, evidence: str) -> dict:
    """提出对提示词“选实验的原则”的修改建议（仅建议，人工审核后才生效）。"""
    _append(ROOT / "proposals.jsonl", {"at": _now().isoformat(timespec="seconds"),
                                       "proposal": proposal[:3000], "evidence": evidence[:3000]})
    _audit("propose_playbook_change")
    return {"saved": True, "note": "已记录，等待人工审核"}


if __name__ == "__main__":
    mcp.run()
