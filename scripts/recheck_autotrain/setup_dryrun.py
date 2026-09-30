"""搭建模拟演练目录：以 v2（exp-000）的真实评测结果作为“刚评测完、尚未报告”的任务，
供不同大脑模型各跑一轮比较。不启动训练、不占 GPU（runner 不运行）。在服务器宿主机运行。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

POLICY = {"daily_training_limit": 3, "deadline": "2026-10-08"}
SPACE = {
    "knobs": {"train_sampling": ["balanced_stratum", "natural"], "epochs": [1, 2]},
    "knob_meaning": {
        "train_sampling": "balanced_stratum=按“类别|标签”等权抽样（误检被重复抽得多）；natural=按训练集自然比例",
        "epochs": "训练轮数；v2 为 balanced_stratum 训 3 轮后选第 1 轮",
    },
    "fixed": {"dataset": "ktd-clean-source-qwen-20260930-v2（训练 135 / 验证 47）", "learning_rate": 2e-5,
              "seed": 42, "gpu": "L20 GPU-c977647c（单卡，同一时间只跑一个任务）"},
    "planned_knobs_not_yet_available": ["误检权重封顶", "学习率档位", "是否加入备用真目标"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True, help="v2 对比 v1 的 metrics json")
    parser.add_argument("--samples", type=Path, required=True, help="make_samples 输出目录")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    state = args.out / "state"
    if args.out.exists():
        shutil.rmtree(args.out)
    for sub in ("config", "jobs", "metrics", "samples/exp-000", "reports", "status"):
        (state / sub).mkdir(parents=True, exist_ok=True)
    (state / "config/policy.json").write_text(json.dumps(POLICY, ensure_ascii=False, indent=2), encoding="utf-8")
    (state / "config/experiment_space.json").write_text(json.dumps(SPACE, ensure_ascii=False, indent=2), encoding="utf-8")
    (state / "status/gpu.json").write_text(json.dumps({"gpu_ok": True, "reason": "模拟演练"}, ensure_ascii=False), encoding="utf-8")
    job = {"job_id": "exp-000", "config": {"train_sampling": "balanced_stratum", "epochs": 1},
           "hypothesis": "v2：加入 14 个车类误检，按类别与标签等权抽样训练，预期减少误检被放过",
           "status": "evaluated", "submitted_at": "2026-09-30T08:18:00+08:00", "reported": False, "history": []}
    (state / "jobs/exp-000.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(args.metrics, state / "metrics/exp-000.json")
    for item in args.samples.iterdir():
        shutil.copy2(item, state / "samples/exp-000" / item.name)
    (state / "ledger.jsonl").write_text(json.dumps({
        "at": "2026-09-30T00:00:00+08:00", "job_id": "baseline-v1",
        "note": "v1 基线（高清输入，两路一致）：随机流量真车自动确认 179/180、误检放过 7/19；楼顶难例真车自动确认 63/67"},
        ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copytree(args.src, args.out / "src")
    print("ok", args.out)


if __name__ == "__main__":
    main()
