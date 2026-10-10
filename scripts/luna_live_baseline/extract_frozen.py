#!/usr/bin/env python3
"""从 yoloe 仓库按固定提交提取冻结资料到**私有（gitignore）目录**，并核对哈希、写 provenance。

提交只提交来源 SHA/路径/哈希与提取脚本；items.json / labels.json / 提示词等原始资料
留在 `data/luna-live-baseline/private/`（该目录自我 gitignore），不入库。

用法：
  YOLOE_REPO=/path/to/yoloe-algorithm-service \
  python3 scripts/luna_live_baseline/extract_frozen.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

# 冻结来源（协调方指定的最终冻结资料）
FINAL_BRANCH = "feature/luna-class-prompt-release"
FINAL_COMMIT = "85d10aea9963042cb62c766c616ff4d0d8760931"
FINAL_ROOT = "data/luna-prompt-autotune-final-20261009"
FINAL_FILES = {
    "assets/labels.json": "c990580379d13f4a6c84715ee80cf8620e74aa80ca1df8634923ddcf1b6e252d",
    "assets/items.json": "392e88ff8997dad71eb054973c4403921d36ec032cd3e0be4cea70871dcfbaa6",
    "assets/build.json": "340b702740422c3902571aafc4900128cb756702bc5a5f74c553f1d8704c5fcc",
    "images-manifest.json": "06b5e8ab6edd4967616853576f208f3c8d730c3e8e96234290d34db98226613b",
    "MANIFEST.sha256.json": "91efb439aa83fbe567f668884cdd2095e21b22dddc7d9a9ee816d0fdc11ff4fb",
}
AUTOTUNE_PROMPTS = {
    "autotune_p000.json": f"{FINAL_ROOT}/state/prompts/p000.json",
    "autotune_p002.json": f"{FINAL_ROOT}/state/prompts/p002.json",
}

# 线上当前 worker（只读核实与发布 SHA256SUMS 一致）
WORKER_BRANCH = "feature/luna-worker-failure-log"
WORKER_COMMIT = "25a612f28403bed971db96430ab638508589b5f6"
WORKER_FILES = {
    "worker_25a612f/run_codex_luna_worker.py": ("scripts/run_codex_luna_worker.py",
        "b909136b16f0f357e58f9d3e221482e122195b34e7cec124bf0760aca59495e1"),
    "worker_25a612f/luna_prompts/p000.json": ("scripts/luna_prompts/p000.json",
        "0f265c686c73c751c545d440e3e3e5a1de7cc8206ca86c73e6710412f70e67e9"),
    "worker_25a612f/luna_prompts/p002.json": ("scripts/luna_prompts/p002.json",
        "8da46d8cf25ce51c92291e586561e0cedfa7c268dd3d03bb824f22e7366ffa45"),
    "worker_25a612f/luna_prompts/card-road-crack-20261010.json": ("scripts/luna_prompts/card-road-crack-20261010.json",
        "4ca5e02a16b99071eaa87b8846bd2d3039518ea6c71587e241b1a0dcbf31539a"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_show(repo: Path, commit: str, path: str) -> bytes:
    r = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"],
                       capture_output=True)
    if r.returncode:
        raise SystemExit(f"git show 失败 {commit}:{path}: {r.stderr.decode()[:200]}")
    return r.stdout


def extract(repo: Path, commit: str, rel: str, dest: Path) -> dict:
    data = git_show(repo, commit, rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return {"source": rel, "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(os.environ.get("YOLOE_REPO", "../yoloe-algorithm-service")))
    ap.add_argument("--dest", type=Path, default=Path("data/luna-live-baseline/private/frozen"))
    ap.add_argument("--provenance", type=Path, default=Path("data/luna-live-baseline/provenance.json"))
    args = ap.parse_args()

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"不是 git 仓库：{repo}")
    dest = args.dest
    files, mismatches = {}, []
    for rel, expected in FINAL_FILES.items():
        info = extract(repo, FINAL_COMMIT, f"{FINAL_ROOT}/{rel}", dest / rel)
        info["expected"] = expected
        info["ok"] = info["sha256"] == expected
        files[f"final/{rel}"] = info
        if not info["ok"]:
            mismatches.append(f"final/{rel}")
    for name, rel in AUTOTUNE_PROMPTS.items():
        info = extract(repo, FINAL_COMMIT, rel, dest / name)
        files[f"final/{rel}"] = info
    for name, (rel, expected) in WORKER_FILES.items():
        info = extract(repo, WORKER_COMMIT, rel, dest / name)
        info["expected"] = expected
        info["ok"] = info["sha256"] == expected
        files[f"worker/{rel}"] = info
        if not info["ok"]:
            mismatches.append(f"worker/{rel}")

    # 官方计分：提取到私有目录，并核对可入库纯代码副本。
    metrics_rel = f"{FINAL_ROOT}/src/metrics.py"
    info = extract(repo, FINAL_COMMIT, metrics_rel, dest / "official_metrics.py")
    copied = Path("scripts/luna_live_baseline/metrics_official.py")
    info["committed_copy"] = str(copied)
    info["committed_copy_sha256"] = sha256_file(copied) if copied.exists() else None
    info["committed_copy_matches"] = info["committed_copy_sha256"] == info["sha256"]
    files[f"final/{metrics_rel}"] = info
    if not info["committed_copy_matches"]:
        mismatches.append("official_metrics committed copy")

    prov = {
        "generated_by": "scripts/luna_live_baseline/extract_frozen.py",
        "final_dataset": {"repo": "yoloe-algorithm-service", "branch": FINAL_BRANCH,
                          "commit": FINAL_COMMIT, "root": FINAL_ROOT},
        "worker": {"repo": "yoloe-algorithm-service", "branch": WORKER_BRANCH,
                   "commit": WORKER_COMMIT, "root": "scripts"},
        "official_metrics": {
            "source": f"{FINAL_ROOT}/src/metrics.py",
            "committed_copy": "scripts/luna_live_baseline/metrics_official.py",
            "note": "评分口径与线上一致；复制为纯代码，便于计分复现。",
        },
        "files": files,
        "all_expected_ok": not mismatches,
        "private_dest": str(dest),
        "note": "items.json/labels.json/提示词等原始资料位于 self-gitignore 的 private/ 下，不入库。",
    }
    args.provenance.write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"提取到 {dest}，provenance 写入 {args.provenance}")
    print(f"  期望哈希全部一致: {prov['all_expected_ok']}")
    if mismatches:
        raise SystemExit(f"哈希不一致：{mismatches}")


if __name__ == "__main__":
    main()
