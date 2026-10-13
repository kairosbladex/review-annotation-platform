#!/usr/bin/env python3
"""在服务器上把低清候选框投影回原始视频帧/静态原图，生成高清复核素材。

输入 jobs.jsonl 每行一个事件：
  {"event_id", "source_kind": "video"|"still", "source_path", "matched_time",
   "homography": 3x3 (低清标注图坐标 -> 原始坐标), "clean_frame",
   "candidates": [{"candidate_id", "slug", "bbox_px": [x1, y1, x2, y2]}]}

输出到 --out：
  frames/<event>.jpg                    匹配原帧（无框）
  frames/<event>.boxed.jpg              匹配原帧（画出本事件全部待复核框）
  crops/<slug>.jpg / <slug>.boxed.jpg   原生像素局部，放大显示（无框/有框）
  neighbors/<slug>.<offset>.jpg         相邻时刻同一原始像素区域（仅参考）
  clips/<slug>.mp4                      匹配时刻前后局部短视频（仅参考，需 --with-clips）
  results.jsonl                         每个候选的投影坐标与对齐核验分数
只读取原视频和原图，不修改任何源文件。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

NEIGHBOR_OFFSETS = (-1.0, -0.5, 0.5, 1.0)
CLIP_HALF_SECONDS = 2.0
SEARCH_HALF_SECONDS = 0.6
CROP_DISPLAY_MIN = 720


def read_video_frame(path: str, t: float) -> np.ndarray | None:
    # ffmpeg 输入端 -ss 在重新解码时是精确定位
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.6f}", "-i", path,
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    data = subprocess.run(cmd, capture_output=True, check=False).stdout
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def best_video_frame(path: str, t: float, clean: np.ndarray | None, h: np.ndarray,
                     width: int, height: int) -> tuple[np.ndarray | None, float, float | None]:
    """在记录时刻前后逐帧解码，取与低清标注图整体最吻合的一帧。"""
    if clean is None:
        return read_video_frame(path, t), t, None
    start = max(0.0, t - SEARCH_HALF_SECONDS)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-i", path, "-t", f"{2 * SEARCH_HALF_SECONDS:.3f}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "-"]
    data = subprocess.run(cmd, capture_output=True, check=False).stdout
    size = width * height * 3
    count = len(data) // size
    if count == 0:
        return read_video_frame(path, t), t, None
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=avg_frame_rate", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    num, _, den = probe.strip().partition("/")
    fps = float(num) / float(den or 1)
    ch, cw = clean.shape[:2]
    inv = np.linalg.inv(h)
    gb = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    best = (-2.0, 0)
    for i in range(count):
        frame = np.frombuffer(data, np.uint8, size, i * size).reshape(height, width, 3)
        warped = cv2.warpPerspective(frame, inv, (cw, ch), flags=cv2.INTER_AREA)
        score = ncc(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), gb) or -2.0
        if score > best[0]:
            best = (score, i)
    frame = np.frombuffer(data, np.uint8, size, best[1] * size).reshape(height, width, 3).copy()
    return frame, start + best[1] / fps, best[0]


def video_size(path: str) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def project(h: np.ndarray, box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    pts = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float64)
    return cv2.perspectiveTransform(pts, h).reshape(4, 2)


def context_rect(poly: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    half = max(side * 1.6, 96)
    rx1 = int(max(0, np.floor(cx - half)))
    ry1 = int(max(0, np.floor(cy - half)))
    rx2 = int(min(width, np.ceil(cx + half)))
    ry2 = int(min(height, np.ceil(cy + half)))
    return rx1, ry1, rx2, ry2


def enlarge(img: np.ndarray) -> tuple[np.ndarray, float]:
    scale = max(1.0, CROP_DISPLAY_MIN / max(img.shape[:2]))
    if scale == 1.0:
        return img.copy(), 1.0
    size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC), scale


def draw_outline(img: np.ndarray, poly: np.ndarray, thickness: int) -> np.ndarray:
    out = img.copy()
    # 框外扩几像素绘制，避免线条压住目标本身
    center = poly.mean(axis=0)
    pad = thickness + 2
    vec = poly - center
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1
    outer = poly + vec / norm * pad * 1.42
    cv2.polylines(out, [np.round(outer).astype(np.int32)], True, (0, 255, 255), thickness, cv2.LINE_AA)
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size < 16:
        return None
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return None if den == 0 else float((a * b).sum() / den)


def alignment_scores(src: np.ndarray, clean: np.ndarray | None, h: np.ndarray, box: list[float]) -> dict:
    if clean is None:
        return {"full_ncc": None, "roi_ncc": None}
    ch, cw = clean.shape[:2]
    warped = cv2.warpPerspective(src, np.linalg.inv(h), (cw, ch), flags=cv2.INTER_AREA)
    ga = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = box
    pad = max(12, int(max(x2 - x1, y2 - y1)))
    rx1, ry1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    rx2, ry2 = min(cw, int(np.ceil(x2)) + pad), min(ch, int(np.ceil(y2)) + pad)
    return {"full_ncc": ncc(ga, gb), "roi_ncc": ncc(ga[ry1:ry2, rx1:rx2], gb[ry1:ry2, rx1:rx2])}


def write_clip(path: str, t: float, rect: tuple[int, int, int, int], out: Path) -> bool:
    x1, y1, x2, y2 = rect
    w, h = (x2 - x1) // 2 * 2, (y2 - y1) // 2 * 2
    scale = max(1, int(np.ceil(CROP_DISPLAY_MIN / max(w, h))))
    start = max(0.0, t - CLIP_HALF_SECONDS)
    vf = f"crop={w}:{h}:{x1}:{y1},scale={w * scale}:{h * scale}:flags=bicubic"
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", path,
           "-t", f"{2 * CLIP_HALF_SECONDS:.3f}", "-vf", vf, "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    return subprocess.run(cmd, check=False).returncode == 0 and out.exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True, help="source_path/clean_frame 的相对根目录")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--with-clips", action="store_true", help="额外生成匹配时刻前后 4 秒局部短视频（默认只出图片）")
    args = parser.parse_args()
    for sub in ("frames", "crops", "neighbors") + (("clips",) if args.with_clips else ()):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    results = []
    for line in args.jobs.read_text(encoding="utf-8").splitlines():
        job = json.loads(line)
        event = job["event_id"]
        src_path = str(args.work_root / job["source_path"])
        h = np.array(job["homography"], np.float64)
        clean = cv2.imread(str(args.work_root / job["clean_frame"]), cv2.IMREAD_COLOR) if job.get("clean_frame") else None
        frame_time, search_ncc = job.get("matched_time"), None
        if job["source_kind"] == "video":
            vw, vh = video_size(src_path)
            frame, frame_time, search_ncc = best_video_frame(src_path, job["matched_time"], clean, h, vw, vh)
        else:
            frame = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if frame is None:
            for cand in job["candidates"]:
                results.append({"candidate_id": cand["candidate_id"], "event_id": event, "error": "无法读取原始来源"})
            continue
        fh, fw = frame.shape[:2]
        cv2.imwrite(str(args.out / "frames" / f"{event}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxed_full = frame.copy()
        thick = max(2, round(max(fw, fh) / 700))
        neighbor_frames = {}
        if job["source_kind"] == "video":
            for off in NEIGHBOR_OFFSETS:
                neighbor_frames[off] = read_video_frame(src_path, frame_time + off)

        for cand in job["candidates"]:
            slug = cand["slug"]
            poly = project(h, cand["bbox_px"])
            cx, cy = poly.mean(axis=0)
            if not (0 <= cx < fw and 0 <= cy < fh):
                # 匹配的原片没有拍到该框位置，不能当作该目标的高清来源
                results.append({"candidate_id": cand["candidate_id"], "event_id": event,
                                "error": "projected_outside_source"})
                continue
            boxed_full = draw_outline(boxed_full, poly, thick)
            rect = context_rect(poly, fw, fh)
            rx1, ry1, rx2, ry2 = rect
            crop, scale = enlarge(frame[ry1:ry2, rx1:rx2])
            local_poly = (poly - [rx1, ry1]) * scale
            cv2.imwrite(str(args.out / "crops" / f"{slug}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(args.out / "crops" / f"{slug}.boxed.jpg"),
                        draw_outline(crop, local_poly, 2), [cv2.IMWRITE_JPEG_QUALITY, 95])
            neighbors = []
            for off, nf in neighbor_frames.items():
                if nf is None:
                    continue
                ncrop, _ = enlarge(nf[ry1:ry2, rx1:rx2])
                name = f"{slug}.{off:+.1f}s.jpg"
                cv2.imwrite(str(args.out / "neighbors" / name), ncrop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                neighbors.append({"offset_seconds": off, "file": f"neighbors/{name}"})
            clip = None
            if args.with_clips and job["source_kind"] == "video" and write_clip(src_path, frame_time, rect, args.out / "clips" / f"{slug}.mp4"):
                clip = f"clips/{slug}.mp4"
            px1, py1 = poly.min(axis=0)
            px2, py2 = poly.max(axis=0)
            results.append({
                "candidate_id": cand["candidate_id"],
                "event_id": event,
                "source_kind": job["source_kind"],
                "source_path": job["source_path"],
                "recorded_match_time": job.get("matched_time"),
                "frame_time": None if frame_time is None else round(frame_time, 6),
                "frame_search_full_ncc": search_ncc,
                "source_size": [fw, fh],
                "projected_polygon": np.round(poly, 2).tolist(),
                "projected_box_size_px": [round(float(px2 - px1), 1), round(float(py2 - py1), 1)],
                "context_rect": list(rect),
                "display_scale": round(scale, 3),
                "frame": f"frames/{event}.jpg",
                "frame_boxed": f"frames/{event}.boxed.jpg",
                "crop": f"crops/{slug}.jpg",
                "crop_boxed": f"crops/{slug}.boxed.jpg",
                "neighbors": neighbors,
                "clip": clip,
                **alignment_scores(frame, clean, h, cand["bbox_px"]),
            })
        cv2.imwrite(str(args.out / "frames" / f"{event}.boxed.jpg"), boxed_full, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(event, "ok", flush=True)

    with (args.out / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
