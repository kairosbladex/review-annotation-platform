"""为自动迭代 Agent 生成错判样本拼图（在服务器宿主机运行）。

每行一个样本：截图上下文 | 截图局部 | 高清原图上下文 | 高清原图局部（黄框为目标）。
视图裁剪与复检服务四视图一致（上下文 12 倍框长、最小 320 像素；局部 4 倍、最小 96 像素）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

SETS = {
    "random": ("/REDACTED_RUNTIME_ROOT/claude-vsg-random-ab-20260929", "input/sample.jsonl"),
    "hard": ("/REDACTED_RUNTIME_ROOT/claude-vsg-resolution-ab-20260929", "input/rows.jsonl"),
}
TILE = 224
MAX_ROWS = 4  # Hermes 送图上限约 256KB，控制行数与 JPEG 质量避免再压缩


def crop(image: Image.Image, bbox: list[float], factor: float, min_side: int, box: bool) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    side = max(min_side * max(1.0, width / 960), max(x2 - x1, y2 - y1) * factor)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = int(cx - side / 2), int(cy - side / 2)
    tile = Image.new("RGB", (int(side), int(side)), (40, 40, 40))
    tile.paste(image.crop((max(0, left), max(0, top), min(width, left + int(side)), min(height, top + int(side)))),
               (max(0, -left), max(0, -top)))
    if box:
        ImageDraw.Draw(tile).rectangle((x1 - left, y1 - top, x2 - left, y2 - top), outline=(255, 220, 0),
                                       width=max(2, int(side / 150)))
    return tile.resize((TILE, TILE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    rows = {}
    for name, (root, rel) in SETS.items():
        for line in Path(root, rel).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["candidate_id"]] = (name, root, row)
    kinds = {
        "real_car_regressed": [("random", c) for c in metrics["random_traffic_200"]["real_paired"]["worse_ids"]]
        + [("hard", c) for c in metrics["rooftop_hard_108"]["real_paired"]["worse_ids"]],
        "negative_still_released": [("random", c) for c in metrics["random_traffic_200"]["negative_still_released_ids"]]
        + [("hard", c) for c in metrics["rooftop_hard_108"]["negative_still_released_ids"]],
    }
    human = {"real_car_regressed": ("HUMAN LABEL: REAL VEHICLE", "人工标签：全部是真车（v1 自动确认，本次没有确认）"),
             "negative_still_released": ("HUMAN LABEL: NOT A VEHICLE (false detection)",
                                         "人工标签：全部是误检，不是车（本次仍被自动确认为车）")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for kind, items in kinds.items():
        picked = items[:MAX_ROWS]
        sheet = Image.new("RGB", (4 * TILE + 8, max(1, len(picked)) * (TILE + 22)), "white")
        draw = ImageDraw.Draw(sheet)
        listing = []
        for index, (set_name, cid) in enumerate(picked):
            _, root, row = rows[cid]
            event = row["event_id"]
            annotated = Image.open(Path(root) / "images/annotated" / f"{event}.png").convert("RGB")
            clean = Image.open(Path(root) / "images/high_clean" / f"{event}.jpg").convert("RGB")
            views = [crop(annotated, row["bbox"], 12, 320, False), crop(annotated, row["bbox"], 4, 96, False),
                     crop(clean, row["bbox"], 12, 320, True), crop(clean, row["bbox"], 4, 96, True)]
            y = index * (TILE + 22)
            draw.text((4, y + 4), f"#{index + 1} {set_name} {cid}   {human[kind][0]}", fill="black")
            for col, view in enumerate(views):
                sheet.paste(view, (col * (TILE + 2), y + 20))
            listing.append({"row": index + 1, "eval_set": set_name, "candidate_id": cid,
                            "class_name": row.get("class_name")})
        sheet.save(args.out_dir / f"{kind}.jpg", quality=82)
        (args.out_dir / f"{kind}.json").write_text(json.dumps(
            {"human_label": human[kind][1], "total": len(items), "shown": len(picked), "rows": listing,
             "layout": "每行：截图上下文 | 截图局部 | 高清原图上下文 | 高清原图局部（黄框为目标）"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(kind, len(items), "shown", len(picked))


if __name__ == "__main__":
    main()
