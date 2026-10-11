"""Extract only the ten preselected source frames; run inside a capped systemd unit.

Reads existing video caches. No downloads, model calls, or business writes.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def refine(job, dest):
    """Bounded streaming frame search; never collect a decoded video in memory."""
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    clean = cv2.imread(job['clean_path'])
    reference = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
    reference -= reference.mean()
    norm = np.linalg.norm(reference)
    start = max(0, job['time'] - 0.6)
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-ss', str(start),
               '-i', job['source_path'], '-t', '1.2', '-threads', '1',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-vsync', '0', '-']
    width, height = job['native_size']
    size = width * height * 3
    inverse = np.linalg.inv(np.array(job['H']))
    best, chosen, count = -1.0, None, 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        try:
            while count < 90:
                data = process.stdout.read(size)
                if not data:
                    break
                if len(data) != size:
                    raise ValueError('incomplete decoded frame')
                frame = np.frombuffer(data, np.uint8).reshape(height, width, 3)
                warped = cv2.warpPerspective(frame, inverse, (clean.shape[1], clean.shape[0]), flags=cv2.INTER_AREA)
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
                gray -= gray.mean()
                score = float((gray * reference).sum() / (np.linalg.norm(gray) * norm))
                if score > best:
                    best, chosen = score, count
                    cv2.imwrite(str(dest), frame)
                count += 1
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
    if chosen is None:
        raise ValueError('no frame found')
    return {'search_start': start, 'frame_index': chosen, 'frames_examined': count, 'full_ncc': best,
            'time_note': 'frame identity verified against clean reference; no absolute capture-time claim'}

parser = argparse.ArgumentParser()
parser.add_argument('--jobs', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
rows = []
for job in json.loads(args.jobs.read_text()):
    dest = args.out / (job['event_id'] + '.png')
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
               '-ss', str(job['time']), '-i', job['source_path'],
               '-frames:v', '1', '-threads', '1', '-y', str(dest)]
    alignment = refine(job, dest) if job.get('refine') else None
    if alignment is None:
        subprocess.run(command, check=True, timeout=60, capture_output=True)
    rows.append({'event_id': job['event_id'], 'seek_seconds': job['time'],
                 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                 'bytes': dest.stat().st_size, 'alignment': alignment})
(args.out / 'manifest.json').write_text(json.dumps(rows, indent=2))
print(json.dumps({'extracted': len(rows), 'bytes': sum(r['bytes'] for r in rows)}))
