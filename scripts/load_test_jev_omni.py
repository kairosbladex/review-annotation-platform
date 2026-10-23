#!/usr/bin/env python3
"""Run 50 real double-view composite requests against Jev-Omni at concurrency 4."""

from __future__ import annotations

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("/bundle"))
    parser.add_argument("--manifest", type=Path, default=Path("/bundle/manifest_en.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--base-url", default="http://127.0.0.1:18010")
    parser.add_argument("--output", type=Path, default=Path("/state/logs/load_test_50.json"))
    args = parser.parse_args()

    rows = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break
    if len(rows) < args.limit:
        raise RuntimeError(f"only {len(rows)} manifest rows available")

    requests = []
    for row in rows:
        image = (args.bundle / row["composite"]).read_bytes()
        requests.append(
            {
                "candidate_id": row["candidate_id"],
                "state": row["state"],
                "question": row["question"],
                "options": list(row["options"]),
                "image_base64": base64.b64encode(image).decode("ascii"),
                "image_media_type": "image/jpeg",
                "modality": "image",
            }
        )

    def call(item: tuple[int, dict]) -> dict:
        index, payload = item
        started = time.perf_counter()
        response = client.post(
            f"{args.base_url.rstrip('/')}/v1/decide",
            json=payload,
            timeout=180.0,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        body = response.json()
        return {
            "index": index,
            "candidate_id": payload["candidate_id"],
            "latency_ms": elapsed_ms,
            "service_latency_ms": body.get("latency_ms"),
            "p_true": body.get("p_true"),
            "status": response.status_code,
        }

    results = []
    wall_start = time.perf_counter()
    with httpx.Client(timeout=180.0, trust_env=False) as client:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(call, item): item[0] for item in enumerate(requests)}
            for future in as_completed(futures):
                results.append(future.result())
    wall_seconds = time.perf_counter() - wall_start
    latencies = [item["latency_ms"] for item in results]
    service_latencies = [
        float(item["service_latency_ms"])
        for item in results
        if item["service_latency_ms"] is not None
    ]
    probabilities = [float(item["p_true"]) for item in results]
    summary = {
        "requests": len(results),
        "concurrency": args.concurrency,
        "wall_seconds": round(wall_seconds, 3),
        "qps": round(len(results) / wall_seconds, 3),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3),
            "p50": round(_percentile(latencies, 50), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "p99": round(_percentile(latencies, 99), 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "service_latency_ms": {
            "mean": round(sum(service_latencies) / len(service_latencies), 3),
            "p50": round(_percentile(service_latencies, 50), 3),
            "p95": round(_percentile(service_latencies, 95), 3),
            "p99": round(_percentile(service_latencies, 99), 3),
        },
        "p_true": {
            "min": round(min(probabilities), 6),
            "max": round(max(probabilities), 6),
        },
        "status_counts": {
            str(status): sum(1 for item in results if item["status"] == status)
            for status in sorted({item["status"] for item in results})
        },
        "results": sorted(results, key=lambda item: item["index"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "results"}, ensure_ascii=False, indent=2))
    return 0 if len(results) == args.limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
