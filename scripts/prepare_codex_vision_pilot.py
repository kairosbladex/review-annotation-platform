"""Freeze a small labelled review slice and existing skypin controls, offline."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(review_root, geo_root, out):
    out.mkdir(parents=True, exist_ok=False)
    pairs = [json.loads(line) for line in (review_root / "dataset-source/candidate_pairs.jsonl").read_text().splitlines()]
    labels = {r["candidate_id"]: r for r in csv.DictReader(
        (review_root / "labeling-package/annotations.csv").open(encoding="utf-8-sig"))}
    rows = sorted([r for r in pairs if r["partition"] == "validation"
                   and r.get("label_source") == "wxl_hires_web_20260930"],
                  key=lambda r: r["candidate_id"])
    if len(rows) != 12:
        raise ValueError("frozen slice changed; review selection before sending")
    manifest = []
    for row in rows:
        source_label = labels[row["origin_candidate_id"]]
        assert {"真目标": "true", "误检": "false"}[source_label["label"]] == row["human_label"]
        assert not source_label["eval_set"]
        name = "review-" + row["candidate_id"]
        dest = out / name
        dest.mkdir()
        image_path = review_root / "dataset-source" / row["clean_frame"]
        with Image.open(image_path) as source:
            im = source.convert("RGB")
        assert list(im.size) == row["clean_frame_size"]
        x1, y1, x2, y2 = row["projected_clean_bbox_px"]
        assert 0 <= x1 < x2 <= im.width and 0 <= y1 < y2 <= im.height
        cx, cy = (x1+x2)/2, (y1+y2)/2
        span = max(512, 4 * max(x2-x1, y2-y1))
        box = (int(max(0, cx-span/2)), int(max(0, cy-span/2)),
               int(min(im.width, cx+span/2)), int(min(im.height, cy+span/2)))
        context = im.crop(box)
        ImageDraw.Draw(context).rectangle((x1-box[0], y1-box[1], x2-box[0], y2-box[1]), outline="yellow", width=3)
        context.thumbnail((1000,1000))
        context.save(dest / "context.jpg", quality=95)
        pad = max(4, 0.1 * max(x2-x1, y2-y1))
        roi = im.crop((int(max(0,x1-pad)),int(max(0,y1-pad)),int(min(im.width,x2+pad)),int(min(im.height,y2+pad))))
        roi.thumbnail((640,640))
        roi.save(dest / "target.jpg", quality=95)
        prompt = (
            "判断第一张图黄色框内是否真实存在车辆。第二张是同一目标的无框局部图。"
            "只判断指定框内，邻近车辆不能算作该目标。检查车身、车窗、车轮等可见结构，"
            "不要把屋顶设施、棚顶、阴影、矩形斑块当车辆；也不要仅因出现在屋顶或场景少见就否决。"
            "true=有足够可见证据确认车辆，false=有足够证据确认不是车辆，"
            "uncertain=分辨率、遮挡或证据不足。不能因看不清就判false。"
            "返回decision和简短中文reason。只看所附图片，不调用工具，不读其他文件。"
        )
        request = {"task":"review", "model":"gpt-5.6-sol", "images":["context.jpg","target.jpg"], "prompt":prompt}
        (dest / "request.json").write_text(json.dumps(request,ensure_ascii=False))
        manifest.append({"job": name, "task":"review", "human_label":row["human_label"],
                         "source_task_id":row["source_task_id"], "candidate_id":row["candidate_id"],
                         "source_sha256":sha(image_path), "label_source":row["label_source"],
                         "images":{n:sha(dest/n) for n in request["images"]},
                         "request_sha256":sha(dest/"request.json")})
    for i in range(1,5):
        name = f"geo-case{i:02}"
        dest = out / name
        dest.mkdir()
        shutil.copyfile(geo_root / f"case{i:02}.jpg", dest / "comparison.jpg")
        prompt = (
            "这是无人机撒点视觉质检，图中SOURCE为目标照片，CANDIDATE A/B为两个候选。"
            "先凭道路、建筑等稳定地物核对是否同地，考虑旋转、尺度和拍摄日期。"
            "不同日期的车辆不是固定地标。黑色无影像区域不能判断。"
            "判断哪一候选更合理，不假设必有正确答案。"
            "decision只能是A/B/neither/uncertain，reason写简短中文可见依据和局限。"
            "两者明显不合理或不同地点选neither，不能可靠对应选uncertain。"
            "禁止编造经纬度或米级精度。只看所附图片，不调用工具，不读其他文件。"
        )
        request = {"task":"geolocation", "model":"gpt-5.6-sol", "images":["comparison.jpg"], "prompt":prompt}
        (dest / "request.json").write_text(json.dumps(request,ensure_ascii=False))
        manifest.append({"job":name,"task":"geolocation", "images":{"comparison.jpg":sha(dest/'comparison.jpg')},
                         "request_sha256":sha(dest/'request.json'),
                         "expected": "B" if i==1 else "neither_or_uncertain" if i==4 else None,
                         "absolute_truth_available":False})
    (out / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root",type=Path,required=True)
    parser.add_argument("--geo-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps({"jobs":len(prepare(args.review_root,args.geo_root,args.out))}))
