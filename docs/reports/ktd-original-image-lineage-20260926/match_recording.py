"""只读检索录像候选帧；结果不构成同帧确认或复核结论。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def retrieve(shot_file, video_file, output):
    shot = cv2.imread(str(shot_file))
    if shot is None:
        raise ValueError("无法解码初审图")
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    gray = clahe.apply(cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY))
    red = ((shot[:, :, 2] > 110)
           & (shot[:, :, 2] > shot[:, :, 1] * 1.35)
           & (shot[:, :, 2] > shot[:, :, 0] * 1.35))
    mask = np.full(gray.shape, 255, np.uint8)
    mask[cv2.dilate(red.astype("uint8"), np.ones((9, 9), np.uint8)) > 0] = 0
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=.015)
    shot_keys, shot_descriptors = sift.detectAndCompute(gray, mask)
    if shot_descriptors is None:
        raise ValueError("初审图缺少可检索的背景特征")

    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise ValueError("无法读取录像或帧率")
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    candidates = []

    def measure(seconds):
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        if not ok:
            return
        resized = cv2.resize(frame, (shot.shape[1], shot.shape[0]))
        processed = clahe.apply(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
        keys, descriptors = sift.detectAndCompute(processed, None)
        if descriptors is None:
            return
        matches = cv2.BFMatcher().knnMatch(shot_descriptors, descriptors, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < .7 * n.distance]
        row = {"seconds": seconds, "matches": len(good), "inliers": 0,
               "medianReprojectionError": None, "homographyToResizedVideo": None}
        if len(good) >= 10:
            source = np.float32([shot_keys[m.queryIdx].pt for m in good])
            target = np.float32([keys[m.trainIdx].pt for m in good])
            matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3)
            if matrix is not None:
                residual = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2) - target
                row.update(inliers=int(inliers.sum()), homographyToResizedVideo=matrix.tolist(),
                           medianReprojectionError=float(np.median(np.linalg.norm(residual[inliers.ravel() > 0], axis=1))))
        candidates.append(row)

    try:
        for seconds in range(0, int(duration), 4):
            measure(seconds)
        candidates.sort(key=lambda row: row["inliers"], reverse=True)
        if not candidates or candidates[0]["inliers"] == 0:
            raise ValueError("未检索到有几何支持的候选帧")
        coarse = candidates[0]["seconds"]
        for seconds in np.arange(max(0, coarse - 4), min(duration, coarse + 4), .25):
            measure(float(seconds))
        candidates.sort(key=lambda row: row["inliers"], reverse=True)
        output.mkdir(parents=True, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_MSEC, candidates[0]["seconds"] * 1000)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(output / "candidate-frame.jpg"), frame)
        result = {"sameFrameConfirmed": False, "automaticRecheckAllowed": False,
                  "durationSeconds": duration, "topCandidates": candidates[:10],
                  "analysisResize": list(shot.shape[:2][::-1]),
                  "note": "仅背景匹配；另需时间戳、移动目标与相机通道一致性检查。"}
        (output / "candidate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotated_image", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    result = retrieve(args.annotated_image, args.recording, args.output_directory)
    print(json.dumps(result["topCandidates"][0], ensure_ascii=False))
