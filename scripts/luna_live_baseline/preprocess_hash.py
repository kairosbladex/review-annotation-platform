#!/usr/bin/env python3
"""在服务器上按线上 worker 原样跑一次图像预处理并记录哈希（不调用模型、不读凭据）。

做法：加载线上 worker 源码，替换 `subprocess.Popen` 为只做“抓取临时目录文件”的桩，
再调用其 `execute()`。预处理（source.jpg / located.jpg / crops.jpg）确实由线上代码完成，
随后桩在构造时复制这几个文件；`execute()` 因桩返回失败而抛错，被本脚本吞掉——
不会发起任何模型请求。`CODEX_REVIEW_HOME` 指向含假 auth.json 的临时目录，不碰真实账号材料。

输出：JSON，含每个样图的预热产物 sha256、提示词 sha256、候选编号校验结果。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

CAPTURE_DIR: Path | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CapturePopen:
    """只抓取 -C 目录里的预处理产物，不发请求。"""

    def __init__(self, cmd, **kwargs):
        self.pid = 0
        self.returncode = 1  # 触发 execute 的失败分支，避免任何网络调用
        self.stdin = self.stdout = self.stderr = None
        self.cmd = list(cmd)
        root = None
        if "-C" in cmd:
            root = Path(cmd[cmd.index("-C") + 1])
        self.captured = {}
        if root is not None and CAPTURE_DIR is not None:
            for name in ("source.jpg", "located.jpg", "crops.jpg", "clean.jpg"):
                p = root / name
                if p.exists():
                    dst = CAPTURE_DIR / name
                    shutil.copy2(p, dst)
                    self.captured[name] = sha256_file(dst)

    def communicate(self, *a, **k):
        return "", ""

    def wait(self, timeout=None):
        return 0


def load_worker(path: Path):
    spec = importlib.util.spec_from_file_location("luna_worker_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--items", type=Path, default=None, help="默认 assets/items.json")
    ap.add_argument("--prompt-version", default="p000-pure-class-cards")
    ap.add_argument("--units", default="", help="逗号分隔的 unit；默认自动挑选 4 个代表图")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    os.environ["CODEX_REVIEW_PROMPT_VERSION"] = args.prompt_version
    os.environ["CODEX_REVIEW_BINARY"] = "/bin/true"
    dummy_home = Path(tempfile.mkdtemp(prefix="probe-home-"))
    (dummy_home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "dummy-not-a-secret"}), encoding="utf-8")
    os.environ["CODEX_REVIEW_HOME"] = str(dummy_home)
    tmp_root = Path(tempfile.mkdtemp(prefix="probe-tmp-"))
    os.environ["CODEX_REVIEW_TMP_DIR"] = str(tmp_root)

    items_path = args.items or (args.assets / "items.json")
    items = json.loads(Path(items_path).read_text(encoding="utf-8"))
    mod = load_worker(args.worker)
    assert mod.PROMPT_DOCUMENT["version"] == args.prompt_version

    if args.units:
        wanted = [u.strip() for u in args.units.split(",") if u.strip()]
        units = [u for u in items if u["unit"] in wanted]
    else:
        # 代表图：纯违建（有卡）、纯车（无卡）、混合、以及候选最多的图
        pick = {}
        for u in items:
            names = {c["class_name"] for c in u["candidates"]}
            if "pure_building" not in pick and names == {"违建"}:
                pick["pure_building"] = u
            if "pure_car" not in pick and names == {"车"}:
                pick["pure_car"] = u
            if "mixed" not in pick and len(names) > 1:
                pick["mixed"] = u
        pick["max_candidates"] = max(items, key=lambda u: len(u["candidates"]))
        seen, units = set(), []
        for u in pick.values():
            if u["unit"] not in seen:
                seen.add(u["unit"])
                units.append(u)

    results = []
    global CAPTURE_DIR
    for u in units:
        CAPTURE_DIR = Path(tempfile.mkdtemp(prefix="probe-cap-"))
        cands = [{"target_index": i, "class_name": c["class_name"], "bbox": c["bbox"]}
                 for i, c in enumerate(u["candidates"])]
        payload = {"image": base64.b64encode((args.assets / "images" / f"{u['unit']}.jpg").read_bytes()).decode(),
                   "candidates": cands}
        prompt = mod.build_prompt(mod.PROMPT_DOCUMENT, cands)
        saved_popen = mod.subprocess.Popen
        mod.subprocess.Popen = CapturePopen
        try:
            mod.execute(payload, {})
        except Exception:
            pass
        finally:
            mod.subprocess.Popen = saved_popen
        cap = {name: sha256_file(CAPTURE_DIR / name)
               for name in ("source.jpg", "located.jpg", "crops.jpg") if (CAPTURE_DIR / name).exists()}
        results.append({
            "unit": u["unit"], "split": u["split"],
            "class_names": sorted({c["class_name"] for c in u["candidates"]}),
            "candidate_count": len(cands),
            "candidate_numbering_ok": [c["target_index"] for c in cands] == list(range(len(cands))),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_has_card_违建": "[违建]" in prompt,
            "prompt_has_card_道路裂纹": "[道路裂纹]" in prompt,
            "artifact_sha256": cap,
        })

    out = {
        "worker_file": str(args.worker),
        "worker_sha256": sha256_file(args.worker),
        "prompt_version": args.prompt_version,
        "python": os.sys.version.split()[0],
        "pil_version": getattr(__import__("PIL"), "__version__", None),
        "units": results,
        "notes": "预处理为线上源码原样执行；未发起模型请求；未读取真实账号材料。",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {args.out}：{len(results)} 图")
    for r in results:
        print(f"  {r['unit']} {r['class_names']} n={r['candidate_count']} "
              f"prompt={r['prompt_sha256'][:12]} src={r['artifact_sha256'].get('source.jpg','—')[:12]}")


if __name__ == "__main__":
    main()
