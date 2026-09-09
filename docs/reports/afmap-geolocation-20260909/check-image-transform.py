"""Measure image correspondence; this does not calibrate the camera."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("event_image", type=Path)
parser.add_argument("video_frame", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
event = cv2.imread(str(args.event_image), cv2.IMREAD_GRAYSCALE)
frame = cv2.imread(str(args.video_frame), cv2.IMREAD_GRAYSCALE)
if event is None or frame is None:
    raise SystemExit("Could not decode both input images")
cv2.setRNGSeed(20260909)
sift = cv2.SIFT_create(nfeatures=5000)
k1, d1 = sift.detectAndCompute(event, None)
k2, d2 = sift.detectAndCompute(frame, None)
matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
good = [a for a, b in matches if a.distance < 0.7 * b.distance]
if len(good) < 20:
    raise SystemExit(f"Insufficient image correspondence: {len(good)} matches")
src = np.float32([k1[m.queryIdx].pt for m in good])
dst = np.float32([k2[m.trainIdx].pt for m in good])
matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
if matrix is None or mask is None:
    raise SystemExit("Homography could not be estimated")
inliers = mask.ravel().astype(bool)
projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
errors = np.linalg.norm(projected[inliers] - dst[inliers], axis=1)
h, w = event.shape
landmarks = np.float32([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]])
mapped = cv2.perspectiveTransform(landmarks.reshape(-1, 1, 2), matrix).reshape(-1, 2)
result = {
    "eventSize": [w, h],
    "videoSize": [frame.shape[1], frame.shape[0]],
    "ratioTestMatches": len(good),
    "ransacInliers": int(inliers.sum()),
    "medianInlierErrorPixels": float(np.median(errors)),
    "p95InlierErrorPixels": float(np.percentile(errors, 95)),
    "eventToVideoHomography": matrix.tolist(),
    "eventCornersAndCenterMappedToVideo": mapped.tolist(),
    "sourceHashes": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [args.event_image, args.video_frame]
    },
    "limits": "Different captured frames; registration is not camera calibration or proof of the preprocessing implementation.",
}
text = json.dumps(result, indent=2) + "\n"
print(text, end="")
if args.output:
    args.output.write_text(text)
