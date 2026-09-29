#!/usr/bin/env python3
"""准备复检低清/高清输入对比所需的图片（在服务器 CPU 上运行）。

两组输入只差干净帧的分辨率：
  low  : 线上同款 960×720 干净帧（新事件原图按 INTER_AREA 缩到 960×720 模拟）；
  high : 高清原片按单应矩阵投影回与低清帧完全相同的画面几何，放大 k 倍（视频 k=2，原图 k=4）。
系统截图（annotated）两组相同。归一化框坐标两组相同，因此裁剪区域一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CODEX_ROOT = Path("/data/workspace/jzheng/codex-ktd-real-monitoring-20260924")
OLD_HIRES = Path("/data/workspace/jzheng/claude-hires-review-20260929")
NEW_ROOT = Path("/data/workspace/jzheng/claude-vsg-new-events-20260929")
LOW_SIZE = (960, 720)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gray_ncc(a: np.ndarray, b: np.ndarray) -> float:
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def warp_high(src: np.ndarray, homography: list, low_w: int, low_h: int) -> tuple[np.ndarray, int]:
    k = max(1, round(src.shape[1] / low_w))
    m = np.diag([k, k, 1.0]) @ np.linalg.inv(np.array(homography, np.float64))
    return cv2.warpPerspective(src, m, (low_w * k, low_h * k), flags=cv2.INTER_CUBIC), k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.rows)
    old_jobs = {job["event_id"]: job for job in read_jsonl(OLD_HIRES / "jobs.jsonl")}
    new_jobs = {job["event_id"]: job for job in read_jsonl(NEW_ROOT / "package/jobs.jsonl")}
    for arm in ("low_clean", "high_clean", "annotated"):
        (args.out / arm).mkdir(parents=True, exist_ok=True)

    events = {}
    for row in rows:
        event = row["event_id"]
        if event in events:
            continue
        if row["package"] != "new-events-hires-package":
            low = cv2.imread(str(CODEX_ROOT / "work/clean_frames" / f"{event}.jpg"))
            ann = cv2.imread(str(CODEX_ROOT / "work/annotated" / f"{event}.jpg"))
            src = cv2.imread(str(OLD_HIRES / "out/frames" / f"{event}.jpg"))
            high, k = warp_high(src, old_jobs[event]["homography"], low.shape[1], low.shape[0])
            origin = "recorded_low_res"
        elif new_jobs[event]["source_kind"] == "video":
            low = cv2.imread(str(NEW_ROOT / "video-root/work/clean_frames" / f"{event}.jpg"))
            ann = cv2.imread(str(NEW_ROOT / "frames/annotated" / f"{event}.jpg"))
            src = cv2.imread(str(NEW_ROOT / "package/out/frames" / f"{event}.jpg"))
            high, k = warp_high(src, new_jobs[event]["homography"], low.shape[1], low.shape[0])
            origin = "recorded_low_res"
        else:
            src = cv2.imread(str(NEW_ROOT / "frames/clean" / f"{event}.jpg"))
            low = cv2.resize(src, LOW_SIZE, interpolation=cv2.INTER_AREA)
            ann = cv2.resize(cv2.imread(str(NEW_ROOT / "frames/annotated" / f"{event}.jpg")), LOW_SIZE,
                             interpolation=cv2.INTER_AREA)
            high, k = src, round(src.shape[1] / LOW_SIZE[0], 3)
            origin = "simulated_low_res_from_original"
        if ann.shape != low.shape:
            raise SystemExit(f"截图与低清帧尺寸不一致：{event}")
        cv2.imwrite(str(args.out / "low_clean" / f"{event}.png"), low)
        cv2.imwrite(str(args.out / "annotated" / f"{event}.png"), ann)
        cv2.imwrite(str(args.out / "high_clean" / f"{event}.jpg"), high, [cv2.IMWRITE_JPEG_QUALITY, 97])
        back = cv2.resize(high, (low.shape[1], low.shape[0]), interpolation=cv2.INTER_AREA)
        events[event] = {"event_id": event, "low_size": [low.shape[1], low.shape[0]],
                         "high_size": [high.shape[1], high.shape[0]], "scale": k,
                         "low_origin": origin, "high_vs_low_ncc": round(gray_ncc(back, low), 5)}
    (args.out / "inputs.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in events.values()),
                                           encoding="utf-8")
    scores = [v["high_vs_low_ncc"] for v in events.values()]
    print(json.dumps({"events": len(events), "min_ncc": min(scores), "scales": sorted({str(v["scale"]) for v in events.values()})},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
