#!/usr/bin/env python3
"""Build a self-contained Codex-assisted human review batch from frozen candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_source(selection_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("candidate has no source_image")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = selection_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _bbox(row: Mapping[str, Any], size: tuple[int, int]) -> tuple[int, int, int, int]:
    value = row.get("bbox_px")
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{row.get('candidate_id')} has invalid bbox_px")
    try:
        x1, y1, x2, y2 = (int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{row.get('candidate_id')} has non-integer bbox_px") from exc
    width, height = size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"{row.get('candidate_id')} bbox_px is outside {width}x{height}: {value}"
        )
    return x1, y1, x2, y2


def _expanded_box(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    padding_ratio: float,
    minimum_padding: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    pad = max(minimum_padding, round(max(x2 - x1, y2 - y1) * padding_ratio))
    return max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)


def _display_scale(image: Image.Image, target_max: int = 720) -> tuple[Image.Image, float]:
    longest = max(image.size)
    if longest <= 0:
        raise ValueError("empty crop")
    scale = min(12.0, max(1.0, target_max / longest))
    if scale == 1.0:
        return image.copy(), scale
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.NEAREST,
    )
    return resized, scale


def _save_jpeg(image: Image.Image, path: Path, *, quality: int = 94) -> None:
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True)


def _make_views(
    *,
    source: Path,
    candidate_id: str,
    bbox: tuple[int, int, int, int],
    media_dir: Path,
    event_id: str,
) -> dict[str, Any]:
    image = Image.open(source).convert("RGB")
    source_copy = media_dir / f"source-original-{event_id}.jpg"
    if not source_copy.exists():
        shutil.copyfile(source, source_copy)

    overview_path = media_dir / f"overview-{candidate_id}.jpg"
    overview = image.copy()
    draw = ImageDraw.Draw(overview)
    line_width = max(3, min(image.size) // 240)
    draw.rectangle(bbox, outline=(255, 222, 0), width=line_width)
    draw.text((bbox[0] + 3, max(0, bbox[1] - 18)), candidate_id, fill=(255, 222, 0))
    overview.thumbnail((1440, 1080), Image.Resampling.LANCZOS)
    _save_jpeg(overview, overview_path)

    context_box = _expanded_box(bbox, image.size, padding_ratio=4.0, minimum_padding=64)
    context, _ = _display_scale(image.crop(context_box), target_max=760)
    context_path = media_dir / f"{candidate_id}-context.jpg"
    _save_jpeg(context, context_path)

    roi_box = _expanded_box(bbox, image.size, padding_ratio=1.2, minimum_padding=22)
    roi, roi_scale = _display_scale(image.crop(roi_box), target_max=640)
    roi_path = media_dir / f"{candidate_id}-roi.jpg"
    _save_jpeg(roi, roi_path)

    marked = roi.copy()
    draw = ImageDraw.Draw(marked)
    x1, y1, x2, y2 = bbox
    mx1 = round((x1 - roi_box[0]) * roi_scale)
    my1 = round((y1 - roi_box[1]) * roi_scale)
    mx2 = round((x2 - roi_box[0]) * roi_scale)
    my2 = round((y2 - roi_box[1]) * roi_scale)
    draw.rectangle(
        (mx1, my1, mx2, my2),
        outline=(255, 222, 0),
        width=max(3, round(roi_scale)),
    )
    marked_path = media_dir / f"{candidate_id}-roi-marked.jpg"
    _save_jpeg(marked, marked_path)

    return {
        "full_original_copy": f"media/{source_copy.name}",
        "overview": f"media/{overview_path.name}",
        "media": {
            "context": f"media/{context_path.name}",
            "roi": f"media/{roi_path.name}",
            "roi-marked": f"media/{marked_path.name}",
        },
        "derived_sha256": {
            "context": _sha256(context_path),
            "roi": _sha256(roi_path),
            "roi-marked": _sha256(marked_path),
        },
        "overview_sha256": _sha256(overview_path),
    }


def _contact_sheet(
    output_dir: Path, rows: Sequence[Mapping[str, Any]], *, page_size: int = 6
) -> None:
    sheets_dir = output_dir / "contact-sheets"
    sheets_dir.mkdir(exist_ok=True)
    for page_start in range(0, len(rows), page_size):
        page = rows[page_start : page_start + page_size]
        tiles: list[Image.Image] = []
        for row in page:
            marked = Image.open(output_dir / str(row["media"]["roi-marked"])).convert("RGB")
            marked.thumbnail((520, 360), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (540, 420), "white")
            tile.paste(marked, ((540 - marked.width) // 2, 45))
            draw = ImageDraw.Draw(tile)
            label = f"{row['candidate_id']}  {row['class_name']}  bbox={row['bbox_px']}"
            draw.text((12, 12), label, fill=(20, 26, 38))
            tiles.append(tile)
        columns = 2
        rows_count = (len(tiles) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * 540, rows_count * 420), (232, 235, 240))
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % columns) * 540, (index // columns) * 420))
        first = page_start + 1
        last = page_start + len(page)
        _save_jpeg(
            sheet,
            sheets_dir / f"candidates-{first:03d}-{last:03d}.jpg",
            quality=92,
        )


def _html(
    *, batch_id: str, title: str, subtitle: str, notice: str, data: list[dict[str, Any]]
) -> str:
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</script>", "<\\/script>"
    )
    batch_json = json.dumps(batch_id, ensure_ascii=False)
    title_html = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    subtitle_html = subtitle.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    notice_html = notice.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light;--bg:#f4f6f9;--panel:#fff;--ink:#18212f;--muted:#637083;--line:#dce2ea;--blue:#2854c5}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.top{background:#17243a;color:#fff;padding:24px max(20px,calc((100vw - 1280px)/2))}.top h1{margin:0 0 6px;font-size:22px}.top p{margin:4px 0;color:#d3dbea}.wrap{max-width:1280px;margin:18px auto;padding:0 18px}.notice{background:#fff7df;border:1px solid #efd88f;padding:14px 16px;border-radius:10px;margin-bottom:16px}.toolbar{position:sticky;top:0;z-index:5;display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:#ffffffed;border:1px solid var(--line);padding:12px;border-radius:10px;margin-bottom:15px;backdrop-filter:blur(8px)}button,select,input[type=search]{font:inherit}button{cursor:pointer;border:1px solid var(--line);background:white;color:var(--ink);padding:8px 11px;border-radius:8px}button:hover{border-color:#91a2bd}.toolbar button.primary{background:var(--blue);color:#fff;border-color:var(--blue)}.toolbar .count{font-weight:650;margin-right:auto}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin:14px 0;padding:16px;box-shadow:0 2px 10px #12213a0a}.card-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.card h2{margin:0;font-size:18px}.badge{display:inline-block;border-radius:99px;padding:3px 10px;font-weight:650}.badge.true{background:#e5f5ec;color:#11623d}.badge.false{background:#fdebed;color:#98333d}.badge.uncertain,.badge.hidden{background:#fff2d3;color:#805600}.ev{color:#42516a;margin:10px 0 4px}.meta{color:#637083;font-size:13px}.views{display:grid;grid-template-columns:1fr 1fr 1.2fr;gap:12px;margin:14px 0}.views figure{margin:0;min-width:0}.views img{display:block;width:100%;max-height:340px;object-fit:contain;background:#edf0f4;border:1px solid var(--line);border-radius:8px}.views figcaption{font-size:12px;color:#637083;margin:3px 2px}.viewbuttons{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0}.actions{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.actions button{font-weight:650}.actions .yes{border-color:#69b18c;color:#11623d}.actions .no{border-color:#dd8990;color:#98333d}.actions .hold{border-color:#dfc36e;color:#805600}.actions .issue{border-color:#9ca8b8;color:#42516a}.actions button.selected{outline:3px solid #2954c533;box-shadow:0 0 0 1px #2954c5 inset}textarea{width:100%;min-height:54px;border:1px solid var(--line);border-radius:8px;padding:9px;font:inherit;resize:vertical}.reviewstate{margin-top:8px;color:#637083;font-size:13px}.foot{color:#637083;font-size:13px;margin:20px 0 42px}@media(max-width:760px){.views{grid-template-columns:1fr 1fr}.views figure:last-child{grid-column:1/-1}.top{padding:18px}.wrap{padding:0 10px}}
</style></head><body><header class="top"><h1>__TITLE__</h1><p>__SUBTITLE__</p></header><main class="wrap"><section class="notice"><strong>确认规则</strong><br>__NOTICE__</section><nav class="toolbar"><span class="count" id="progress"></span><label>筛选 <select id="filter"><option value="all">全部候选</option><option value="pending">待确认</option><option value="done">已处理</option><option value="人员">人员</option><option value="车">车辆</option><option value="hidden">盲查项</option></select></label><input id="search" type="search" placeholder="查找候选编号"><button id="export" class="primary">导出全部人工确认 JSONL</button><button id="clear">清除本机记录</button></nav><p id="storage-warning" class="meta" role="status"></p><div id="cards"></div><p class="foot">原图、上下文和局部图均有 SHA-256。平台已有红框和文字仅用于定位，不是真值；黄色框由本页追加。Codex 建议与人工标签分开保存，本批结果只用于独立验证流程试跑，不构成准确率结论。</p></main><script>
const BATCH=__BATCH__,KEY=BATCH+':human-review',DATA=__DATA__;let state={};try{state=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){state={}}
const labels={true:'真',false:'假',uncertain:'不确定'};function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function saveState(){try{localStorage.setItem(KEY,JSON.stringify(state))}catch(e){document.getElementById('storage-warning').textContent='浏览器未允许本地保存；请完成后立即导出。'}}
function action(id,kind){const x=DATA.find(r=>r.candidate_id===id);if(kind==='confirm'&&!x.suggestion_visible_during_review)return;const actual=kind==='confirm'&&x.codex_suggestion==='uncertain'?'uncertain':kind;const label=actual==='issue'?null:actual==='confirm'?x.codex_suggestion:actual;state[id]={human_action:actual,human_label:label,reviewed:true,reviewed_at:new Date().toISOString(),human_comment:state[id]?.human_comment||''};saveState();render()}
function updateNote(id,value){state[id]={...(state[id]||{}),human_comment:value};saveState()}
function toggleMark(button){const image=button.closest('.card').querySelector('img.roi'),marked=image.dataset.marked,plain=image.dataset.plain;if(image.getAttribute('src')===marked){image.src=plain;button.textContent='显示黄色定位框'}else{image.src=marked;button.textContent='关闭黄色定位框'}}
function exportRows(){const pending=DATA.filter(x=>!state[x.candidate_id]?.reviewed);if(pending.length){alert(`还有 ${pending.length} 条未处理，请全部确认后再导出。`);return}const rows=DATA.map(x=>({schema_version:1,batch_id:BATCH,candidate_id:x.candidate_id,source_image_sha256:x.original_sha256,candidate_bbox_px:x.bbox_px,candidate_class:x.class_name,source_group:x.source_group,codex_suggestion:x.codex_suggestion,codex_evidence:x.visual_evidence,suggestion_visible_during_review:x.suggestion_visible_during_review,human_action:state[x.candidate_id].human_action,human_label:state[x.candidate_id].human_label,human_comment:state[x.candidate_id].human_comment||'',human_reviewed:true,reviewed_at:state[x.candidate_id].reviewed_at,label_source:'human_review_pending_audit',is_training_eligible:false,is_validation_eligible:false,dataset_partition:x.dataset_partition}));const blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/x-ndjson'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=BATCH+'-human-review.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function render(){const filter=document.getElementById('filter').value,q=document.getElementById('search').value.trim().toLowerCase(),done=DATA.filter(x=>state[x.candidate_id]?.reviewed).length,binary=DATA.filter(x=>['true','false'].includes(state[x.candidate_id]?.human_label)).length;document.getElementById('progress').textContent=`${done} / ${DATA.length} 已处理 · ${binary} 条人工二态标签`;const rows=DATA.filter(x=>{const s=state[x.candidate_id];if(filter==='pending'&&s?.reviewed)return false;if(filter==='done'&&!s?.reviewed)return false;if(filter==='hidden'&&x.suggestion_visible_during_review)return false;if(['人员','车'].includes(filter)&&x.class_name!==filter)return false;if(q&&!`${x.candidate_id} ${x.class_name}`.toLowerCase().includes(q))return false;return true});document.getElementById('cards').innerHTML=rows.map(x=>{const s=state[x.candidate_id]||{},hidden=!x.suggestion_visible_during_review,pred=hidden?'hidden':x.codex_suggestion,badge=hidden?'盲查：建议已隐藏':`Codex 建议：${labels[x.codex_suggestion]||x.codex_suggestion}`,evidence=hidden?'请先独立判断；该项的 Codex 建议和说明在评审期间不展示。':x.visual_evidence,on=k=>s.human_action===k?'selected':'',status=s.human_action==='issue'?'已标记框问题':s.reviewed?`已记录：${labels[s.human_label]||'不生成二态标签'}`:'等待人工确认',confirm=hidden?'':`<button class="yes ${on('confirm')}" onclick="action('${esc(x.candidate_id)}','confirm')">确认 Codex 建议</button>`;return `<article class="card"><div class="card-head"><h2>${esc(x.candidate_id)} · ${esc(x.class_name)}</h2><span class="badge ${esc(pred)}">${esc(badge)}</span><span class="meta">visibility ${esc(x.visibility)} · class ${esc(x.class_match)} · box ${esc(x.box_fit)}</span></div><p class="ev">${esc(evidence)}</p><div class="views"><figure><a href="${esc(x.full_original_copy)}" target="_blank" rel="noreferrer"><img src="${esc(x.overview)}" loading="lazy" alt="全图缩略"></a><figcaption>全图缩略（点击打开来源图）</figcaption></figure><figure><img src="${esc(x.media.context)}" loading="lazy" alt="候选上下文"><figcaption>候选上下文，放大显示</figcaption></figure><figure><img class="roi" src="${esc(x.media['roi-marked'])}" data-marked="${esc(x.media['roi-marked'])}" data-plain="${esc(x.media.roi)}" loading="lazy" alt="候选局部"><figcaption>候选局部，黄色定位框可切换</figcaption></figure></div><div class="viewbuttons"><button onclick="toggleMark(this)">关闭黄色定位框</button><span class="meta">像素框 ${esc(JSON.stringify(x.bbox_px))} · 原图 SHA-256 ${esc(x.original_sha256.slice(0,16))}…</span></div><div class="actions">${confirm}<button class="yes ${on('true')}" onclick="action('${esc(x.candidate_id)}','true')">判断为真</button><button class="no ${on('false')}" onclick="action('${esc(x.candidate_id)}','false')">判断为假</button><button class="hold ${on('uncertain')}" onclick="action('${esc(x.candidate_id)}','uncertain')">仍无法判断</button><button class="issue ${on('issue')}" onclick="action('${esc(x.candidate_id)}','issue')">框有问题</button></div><textarea placeholder="可选：遮挡、类别或框问题说明" oninput="updateNote('${esc(x.candidate_id)}',this.value)">${esc(s.human_comment||'')}</textarea><div class="reviewstate">${status}${s.reviewed_at?' · '+esc(s.reviewed_at):''}</div></article>`}).join('')||'<p>没有符合筛选条件的候选。</p>'}
document.getElementById('filter').onchange=render;document.getElementById('search').oninput=render;document.getElementById('export').onclick=exportRows;document.getElementById('clear').onclick=()=>{if(confirm('清除此批次在当前浏览器保存的人工复核记录？')){try{localStorage.removeItem(KEY)}catch(e){}state={};render()}};render();
</script></body></html>'''
    return (
        template.replace("__TITLE__", title_html)
        .replace("__SUBTITLE__", subtitle_html)
        .replace("__NOTICE__", notice_html)
        .replace("__BATCH__", batch_json)
        .replace("__DATA__", data_json)
    )


def build_batch(
    *,
    selection_path: Path,
    output_dir: Path,
    batch_id: str,
    title: str,
    subtitle: str,
    notice: str,
    expected_count: int | None,
    source_dataset: str | None,
    limitations: Sequence[str],
    companion_files: Sequence[Path],
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    rows = _read_jsonl(selection_path)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"selection has {len(rows)} rows; expected {expected_count}")
    if not rows:
        raise ValueError("selection is empty")
    ids = [str(row.get("candidate_id") or "") for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique and non-empty")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    media_dir = output_dir / "media"
    media_dir.mkdir()

    manifest_rows: list[dict[str, Any]] = []
    prelabels: list[dict[str, Any]] = []
    ui_rows: list[dict[str, Any]] = []
    for source_row in rows:
        candidate_id = str(source_row["candidate_id"])
        source = _resolved_source(selection_path, source_row.get("source_image"))
        digest = _sha256(source)
        declared_digest = source_row.get("original_sha256")
        if declared_digest not in (None, "") and str(declared_digest) != digest:
            raise ValueError(f"{candidate_id} source image SHA-256 mismatch")
        image = Image.open(source)
        image_size = image.size
        bbox = _bbox(source_row, image_size)
        views = _make_views(
            source=source,
            candidate_id=candidate_id,
            bbox=bbox,
            media_dir=media_dir,
            event_id=str(source_row["event_id"]),
        )
        split = str(source_row.get("split") or "")
        if not split:
            raise ValueError(f"{candidate_id} has no split")
        bbox_norm = [
            bbox[0] / image_size[0],
            bbox[1] / image_size[1],
            bbox[2] / image_size[0],
            bbox[3] / image_size[1],
        ]
        manifest = {
            "schema_version": 1,
            "batch_id": batch_id,
            "candidate_id": candidate_id,
            "event_id": str(source_row["event_id"]),
            "event_no": source_row.get("event_no"),
            "event_time": source_row.get("event_time"),
            "task_id": source_row.get("source_task_id"),
            "source_task_id": source_row.get("source_task_id"),
            "source_group": str(source_row["source_group"]),
            "source_group_kind": str(source_row.get("source_group_kind") or "task_id"),
            "source_capture_group": source_row.get("source_capture_group"),
            "capture_group_status": source_row.get("capture_group_status"),
            "class_name": str(source_row["class_name"]),
            "bbox_px": list(bbox),
            "bbox_norm": bbox_norm,
            "bbox_format": "xyxy_top_left",
            "image_size": list(image_size),
            "source_indexes": list(source_row.get("source_indexes", [source_row.get("source_index", 0)])),
            "source_confidences": list(source_row.get("source_confidences", [source_row.get("confidence")])),
            "source_original": str(source_row.get("source_original") or source),
            "source_target_file": source_row.get("source_target_file"),
            "source_object_key": str(source_row["source_object_key"]),
            "original_sha256": digest,
            "split": split,
            "training_eligible": False,
            "validation_eligible": False,
            "locked_test_eligible": False,
            "human_label": None,
            **views,
            "source_feature_count": source_row.get("source_feature_count", 1),
            "source_row_ordinals": list(source_row.get("source_row_ordinals", [])),
            "source_properties_indexes": list(source_row.get("source_properties_indexes", [])),
            "duplicate_source_rows": int(source_row.get("duplicate_source_rows", 0)),
            "source_overlay_status": source_row.get(
                "source_overlay_status",
                "platform_rendered_class_text_and_red_box_present_not_ground_truth",
            ),
            "suggestion_visible_during_review": bool(
                source_row.get("suggestion_visible_during_review", True)
            ),
        }
        prelabel = {
            "schema_version": 1,
            "batch_id": batch_id,
            "candidate_id": candidate_id,
            "source_image_sha256": digest,
            "candidate_bbox_px": list(bbox),
            "source_group": manifest["source_group"],
            "source_group_kind": manifest["source_group_kind"],
            "candidate_class": manifest["class_name"],
            "codex_suggestion": str(source_row["codex_suggestion"]),
            "visibility": str(source_row["visibility"]),
            "class_match": str(source_row["class_match"]),
            "box_fit": str(source_row["box_fit"]),
            "visual_evidence": str(source_row["visual_evidence"]),
            "label_source": "codex_pseudo_prelabel",
            "is_human_gold": False,
            "human_label": None,
            "human_reviewed": False,
            "dataset_partition": split,
            "training_eligible": False,
            "validation_eligible": False,
            "locked_test_eligible": False,
            "blind_test": not manifest["suggestion_visible_during_review"],
            "suggestion_visible_during_review": manifest[
                "suggestion_visible_during_review"
            ],
            "codex_model": str(
                source_row.get("codex_model")
                or "Codex GPT-6 visual assessment in current conversation; exact serving revision unavailable"
            ),
            "model_revision": str(source_row.get("model_revision") or "unknown"),
            "prompt_version": str(
                source_row.get("prompt_version") or "ktd-candidate-visual-prelabel-v1"
            ),
            "review_date": str(source_row.get("review_date") or "2026-09-24"),
            "views": dict(manifest["derived_sha256"]),
            "source_task_id": manifest["source_task_id"],
            "source_capture_group": manifest["source_capture_group"],
            "source_overlay_warning": "red platform class text and boxes are display artifacts, not truth",
        }
        ui_row = {
            "candidate_id": candidate_id,
            "bbox_px": manifest["bbox_px"],
            "class_name": manifest["class_name"],
            "original_sha256": digest,
            "source_group": manifest["source_group"],
            "source_task_id": manifest["source_task_id"],
            "source_capture_group": manifest["source_capture_group"],
            "media": manifest["media"],
            "overview": manifest["overview"],
            "full_original_copy": manifest["full_original_copy"],
            "codex_suggestion": prelabel["codex_suggestion"],
            "visibility": prelabel["visibility"],
            "class_match": prelabel["class_match"],
            "box_fit": prelabel["box_fit"],
            "visual_evidence": prelabel["visual_evidence"],
            "event_id": manifest["event_id"],
            "suggestion_visible_during_review": prelabel[
                "suggestion_visible_during_review"
            ],
            "dataset_partition": split,
        }
        manifest_rows.append(manifest)
        prelabels.append(prelabel)
        ui_rows.append(ui_row)

    _write_jsonl(output_dir / "candidate_inventory.jsonl", manifest_rows)
    _write_jsonl(output_dir / "pilot_manifest.jsonl", manifest_rows)
    _write_jsonl(output_dir / "codex_prelabels.jsonl", prelabels)
    _write_jsonl(output_dir / "validation_review_queue.jsonl", manifest_rows)
    _contact_sheet(output_dir, manifest_rows)
    for companion in companion_files:
        companion = companion.resolve()
        if not companion.is_file():
            raise FileNotFoundError(companion)
        shutil.copyfile(companion, output_dir / companion.name)

    splits = sorted({str(row["split"]) for row in manifest_rows})
    summary = {
        "schema_version": 1,
        "batch_id": batch_id,
        "source_dataset": source_dataset,
        "candidate_rows": len(manifest_rows),
        "source_images": len({row["original_sha256"] for row in manifest_rows}),
        "source_task_groups": len({row["source_group"] for row in manifest_rows}),
        "source_capture_groups": sorted(
            {
                str(row["source_capture_group"])
                for row in manifest_rows
                if row.get("source_capture_group") not in (None, "")
            }
        ),
        "unknown_capture_group_rows": sum(
            row.get("source_capture_group") in (None, "") for row in manifest_rows
        ),
        "class_counts": dict(sorted(Counter(row["class_name"] for row in manifest_rows).items())),
        "prelabel_counts": dict(
            sorted(Counter(row["codex_suggestion"] for row in prelabels).items())
        ),
        "hidden_suggestion_rows": sum(
            not row["suggestion_visible_during_review"] for row in prelabels
        ),
        "split": splits[0] if len(splits) == 1 else splits,
        "label_status": "codex_pseudo_prelabels_ready_human_confirmation_pending",
        "human_confirmed_rows": 0,
        "gold_rows": 0,
        "validation_ready": False,
        "formal_training_ready": False,
        "accuracy_evaluated": False,
        "limitations": list(dict.fromkeys(str(item) for item in limitations if str(item).strip())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(
        _html(
            batch_id=batch_id,
            title=title,
            subtitle=subtitle,
            notice=notice,
            data=ui_rows,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# KTD Codex 辅助人工确认批次\n\n"
        f"- 批次：`{batch_id}`\n"
        f"- 候选：{len(manifest_rows)} 条\n"
        f"- 类别：{summary['class_counts']}\n"
        f"- 来源任务组：{summary['source_task_groups']}\n"
        f"- 建议隐藏盲查：{summary['hidden_suggestion_rows']} 条\n"
        "- 人工确认前不属于训练、验证金标或锁定测试金标。\n"
        "- 页面只写浏览器 localStorage，导出 JSONL 后仍需审计。\n"
        "- 本批不用于声明准确率。\n",
        encoding="utf-8",
    )

    receipt_files = [selection_path]
    receipt_files.extend(
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "input_receipt.json"
    )
    receipt = {
        "schema_version": 1,
        "batch_id": batch_id,
        "files": [
            {
                "path": str(path),
                "relative_to_output": (
                    str(path.relative_to(output_dir)) if output_dir in path.parents else None
                ),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in receipt_files
        ],
    }
    (output_dir / "input_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", required=True)
    parser.add_argument("--notice", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--source-dataset")
    parser.add_argument("--limitation", action="append", default=[])
    parser.add_argument("--companion-file", type=Path, action="append", default=[])
    args = parser.parse_args()
    summary = build_batch(
        selection_path=args.selection,
        output_dir=args.output_dir.resolve(),
        batch_id=args.batch_id,
        title=args.title,
        subtitle=args.subtitle,
        notice=args.notice,
        expected_count=args.expected_count,
        source_dataset=args.source_dataset,
        limitations=args.limitation,
        companion_files=args.companion_file,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
