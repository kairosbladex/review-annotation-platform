#!/usr/bin/env python3
"""纯类别 / 混合路由一致性与候选编号检查（不调用模型）。

直接加载线上 worker 源码副本 `frozen/worker_25a612f/run_codex_luna_worker.py`，
复用其 `load_prompt` / `build_prompt` / `_class_key`，对冻结 items.json 的每张评测图
验证 p000-pure-class-cards 的路由：

  * 整图候选全为同一类且该类有卡（违建/道路裂纹）→ 插入该类卡；
  * 混合图、其他纯类、以及无标签池图 → 与 p000 逐字节一致（不插卡）；
  * 候选必须保持原编号 target_index=0..n-1，且提示词尾部的 JSON 能原样解析回候选。

退出码非 0 表示有用例失败。不接触凭据、网络或业务数据。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

DEFAULT_FROZEN = Path("data/luna-live-baseline/private/frozen")
WORKER_REL = "worker_25a612f/run_codex_luna_worker.py"
CARDS = {"违建", "道路裂纹"}

ALIASES = {"人": "人员", "小客车": "车", "货车": "卡车", "集装箱货车": "卡车",
           "有斗货车": "卡车", "带蓬布的卡车": "卡车", "大客车": "公交车",
           "路面裂纹": "道路裂纹", "路面病害": "道路裂纹"}
CLASSES = {"车", "人员", "违建", "光斑", "道路裂纹", "光伏板", "两轮车", "三轮车",
           "卡车", "公交车", "没戴头盔", "车牌", "其他"}


def load_worker_module(frozen: Path):
    worker_path = frozen / WORKER_REL
    if not worker_path.exists():
        raise SystemExit(f"缺少 worker 源码：{worker_path}")
    # import 时 worker 会用 CODEX_REVIEW_PROMPT_VERSION 载入提示词；
    # 这里指向复刻的线上版本，不读任何凭据。
    os.environ.setdefault("CODEX_REVIEW_PROMPT_VERSION", "p000-pure-class-cards")
    os.environ.setdefault("CODEX_REVIEW_BINARY", "/bin/true")
    os.environ.setdefault("CODEX_REVIEW_HOME", tempfile.mkdtemp(prefix="dummy-codex-home-"))
    os.environ.setdefault("CODEX_REVIEW_TMP_DIR", tempfile.mkdtemp(prefix="dummy-tmp-"))
    spec = importlib.util.spec_from_file_location("luna_worker_25a612f", worker_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def expected_keys(mod, names: list[str], cards: set[str]) -> list[str]:
    mapped = [mod._class_key(n) for n in names]
    if mapped and len(set(mapped)) == 1 and mapped[0] in cards:
        return [mapped[0]]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    ap.add_argument("--max-report", type=int, default=15)
    args = ap.parse_args()
    frozen = args.frozen
    mod = load_worker_module(frozen)
    items = json.loads((frozen / "items.json").read_text(encoding="utf-8"))

    p000 = mod.load_prompt("p000")
    pure_cards = mod.PROMPT_DOCUMENT
    assert pure_cards["version"] == "p000-pure-class-cards", pure_cards["version"]
    assert set(pure_cards["cards"]) == CARDS, set(pure_cards["cards"])

    failures: list[str] = []
    checks = 0

    # ---- 0. 别名映射 ----
    for raw, mapped in ALIASES.items():
        assert mod._class_key(raw) == mapped, (raw, mod._class_key(raw))
        checks += 1
    for raw in CLASSES:
        assert mod._class_key(raw) == raw, (raw, mod._class_key(raw))
        checks += 1
    assert mod._class_key("不明类别") == "其他"
    checks += 1

    pure_units = mixed_units = no_card_pure = 0
    for unit in items:
        names = [c["class_name"] for c in unit["candidates"]]
        mapped = [mod._class_key(n) for n in names]
        is_pure = len(set(mapped)) == 1
        has_card = bool(mapped) and is_pure and mapped[0] in CARDS

        cands = [{"target_index": i, "class_name": c["class_name"], "bbox": c["bbox"]}
                 for i, c in enumerate(unit["candidates"])]
        # 候选编号：必须原样、连续、按输入顺序。
        assert [c["target_index"] for c in cands] == list(range(len(cands))), unit["unit"]
        checks += 1

        text = mod.build_prompt(pure_cards, cands)
        base = mod.build_prompt(p000, cands)
        exp = expected_keys(mod, names, CARDS)
        keys = [k for k in CARDS if f"\n[{k}] " in text]
        if keys != exp:
            failures.append(f"路由不符 {unit['unit']} split={unit['split']} names={sorted(set(mapped))} "
                            f"got={keys} exp={exp}")

        # 提示词尾部 JSON 必须与候选逐字节一致（编号/顺序不丢失）。
        suffix = mod.PROMPT_SUFFIX
        if not text.endswith(suffix + json.dumps(cands, ensure_ascii=False)):
            failures.append(f"提示词尾部候选 JSON 不匹配 {unit['unit']}")
        else:
            payload = json.loads(text[len(text) - len(json.dumps(cands, ensure_ascii=False)):])
            if payload != cands:
                failures.append(f"候选 JSON 反解析不一致 {unit['unit']}")
        checks += 1

        if has_card:
            pure_units += 1
        elif is_pure:
            no_card_pure += 1
        else:
            mixed_units += 1
        # 无卡时必须与 p000 逐字节一致。
        if not exp and text != base:
            failures.append(f"无卡但与 p000 不一致 {unit['unit']}")

    # ---- 合成用例：纯道路裂纹（冻结 dev/holdout 无裂纹，故手工构造）----
    synth = [{"target_index": 0, "class_name": "道路裂纹", "bbox": [0.1, 0.1, 0.2, 0.2]},
             {"target_index": 1, "class_name": "路面裂纹", "bbox": [0.3, 0.1, 0.4, 0.2]}]
    text = mod.build_prompt(pure_cards, synth)
    if "[道路裂纹]" not in text:
        failures.append("合成纯裂纹图未插入道路裂纹卡")
    checks += 1
    # 混合图（裂纹+车）不得插卡，且与 p000 一致。
    mixed = synth + [{"target_index": 2, "class_name": "车", "bbox": [0.5, 0.1, 0.6, 0.2]}]
    text_mixed = mod.build_prompt(pure_cards, mixed)
    if any(f"[{k}] " in text_mixed for k in CARDS):
        failures.append("合成混合图错误插入了类别卡")
    if text_mixed != mod.build_prompt(p000, mixed):
        failures.append("合成混合图与 p000 不一致")
    checks += 2

    print(f"用例数 {checks}，失败 {len(failures)}")
    print(f"  纯类有卡图 {pure_units}，纯类无卡图 {no_card_pure}，混合图 {mixed_units}")
    if failures:
        for f in failures[: args.max_report]:
            print("  FAIL", f)
        raise SystemExit(1)
    print("路由一致性 / 候选编号检查：全部通过")


if __name__ == "__main__":
    main()
