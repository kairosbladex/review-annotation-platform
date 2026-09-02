#!/usr/bin/env python3
"""Run the KTD visual recheck with durable local outputs and provider fallback."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.services.ktd_recheck import env_with_files, run_recheck, sqlite_path_from_url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env.ktd.local"), help="KTD MySQL and source image env file")
    parser.add_argument("--model-env-file", type=Path, default=Path(".env.auto-annotation.local"), help="GPT routing env file")
    parser.add_argument("--database-url", default="sqlite:///./data/ktd-gpt-recheck.sqlite3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=1510, help="source rows to include")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--provider-order", default="gpt,grok,kimi")
    parser.add_argument("--batch-size", "--image-batch-size", dest="batch_size", type=int, default=10, help="images per model request, 1-20")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--pilot-images", type=int, default=None, help="process only this many unique images while keeping the full manifest")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.offset < 0:
        parser.error("limit must be positive and offset cannot be negative")
    if args.pilot_images is not None and args.pilot_images < 1:
        parser.error("pilot-images must be positive")
    output_dir = args.output_dir
    run_id = args.run_id or output_dir.name
    if not run_id or run_id in {".", ".."}:
        parser.error("run-id must be supplied or encoded in output-dir")
    env = env_with_files([args.env_file, args.model_env_file])
    summary = run_recheck(
        database_path=sqlite_path_from_url(args.database_url, cwd=Path.cwd()),
        output_dir=output_dir,
        run_id=run_id,
        env=env,
        provider_order=tuple(name.strip() for name in args.provider_order.split(",") if name.strip()),
        limit=args.limit,
        offset=args.offset,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        resume=args.resume,
        pilot_images=args.pilot_images,
    )
    print({
        "run_id": summary["run_id"],
        "status": summary["status"],
        "source_rows": summary["source_rows"],
        "unique_images": summary["unique_images"],
        "candidates": summary["candidates"],
        "completed_images": summary["completed_images"],
        "failed_images": summary["failed_images"],
        "provider_calls": summary["provider_calls"],
        "preflight_calls": summary["preflight_calls"],
        "remaining_images": summary.get("remaining_images", 0),
        "output_dir": str(output_dir),
    })
    return 0 if summary["status"] in {"succeeded", "pilot_complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
