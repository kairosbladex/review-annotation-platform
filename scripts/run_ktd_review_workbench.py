#!/usr/bin/env python3
"""Run the live test review UI on this computer only."""
import argparse
import os
import sys
from pathlib import Path

import uvicorn

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "backend"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=8194)
args = parser.parse_args()
required = ["HOST", "USER", "PASSWORD", "NAME"]
missing = ["KTD_REVIEW_DB_" + key for key in required if not os.getenv("KTD_REVIEW_DB_" + key)]
if missing:
    parser.error("Set server-side environment variables: " + ", ".join(missing))
if not (root / "frontend/dist/workbench.html").exists():
    parser.error("Build the frontend first: cd frontend && npm run build")
uvicorn.run("app.recheck_workbench:app", host="127.0.0.1", port=args.port, workers=1, access_log=False)
