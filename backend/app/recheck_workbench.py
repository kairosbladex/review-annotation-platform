"""Local, project-scoped viewer of live KTD test events. No annotation writes.

Run separately from the annotation service. Database transactions are read only;
the sole business write is an explicit single-event POST to the test recheck API.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx
import pymysql
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "frontend/dist"
PROJECT = os.getenv("KTD_REVIEW_PROJECT_ID", "2c95908e958f3b75019593551f520126")
TEST_ORIGIN = "https://lxtest.zhuoyilingxi.com:8088"
DATA = Path(os.getenv("KTD_REVIEW_DATA_DIR", str(ROOT / "data/ktd-review-workbench")))
TLS_VERIFY = os.getenv("KTD_REVIEW_VERIFY_TLS", "true").lower() != "false"
SUBMIT_ENABLED = os.getenv("KTD_REVIEW_ALLOW_SUBMIT", "false").lower() == "true"
DB_CONFIG = {
    "host": os.environ["KTD_REVIEW_DB_HOST"],
    "port": int(os.getenv("KTD_REVIEW_DB_PORT", "3306")),
    "user": os.environ["KTD_REVIEW_DB_USER"],
    "password": os.environ["KTD_REVIEW_DB_PASSWORD"],
    "database": os.environ["KTD_REVIEW_DB_NAME"],
    "charset": "utf8mb4", "connect_timeout": 8, "read_timeout": 25,
    "cursorclass": pymysql.cursors.DictCursor,
}
DATA.mkdir(parents=True, exist_ok=True, mode=0o700)
(DATA / "images").mkdir(exist_ok=True, mode=0o700)
(DATA / "submissions").mkdir(exist_ok=True, mode=0o700)
app = FastAPI(title="KTD test recheck workbench", docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
submit_lock = threading.Lock()
active_submissions: set[str] = set()


def query(sql: str, args=()):
    """Fresh read-only transaction; no client SQL or identifiers are accepted."""
    with pymysql.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("START TRANSACTION READ ONLY")
            cur.execute(sql, args)
            rows = cur.fetchall()
        conn.rollback()
    return rows


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def decode(value, default=None):
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value) if value else default
    except (ValueError, TypeError):
        return default


def event_id(value):
    if not re.fullmatch(r"[a-fA-F0-9]{32}", value):
        raise HTTPException(400, "事件编号格式无效")
    return value


def support_codes():
    # Matches TaskEventServiceImpl -> project state=1 and composite warning join.
    rows = query("""SELECT DISTINCT c.early_warn_code AS code
        FROM bo_early_warning_customize c JOIN bo_early_warning w
        ON w.early_warn_code=c.early_warn_code AND w.ai_platform_code=c.ai_platform_code
        WHERE c.project_id=%s AND c.del_flag=0 AND c.state='1' AND w.support_recheck=1""", (PROJECT,))
    return {x["code"] for x in rows}


def status_of(row, codes):
    return (row.get("recheck_status") or "0") if row["upload_data_type"] in codes else "4"


def event_row(eid):
    rows = query("SELECT * FROM bo_task_event WHERE id=%s AND project_id=%s AND del_flag=0", (event_id(eid), PROJECT))
    if not rows:
        raise HTTPException(404, "当前测试项目中没有这条事件")
    return rows[0]


def brief(row, codes):
    return {
        "id": row["id"], "eventNo": row.get("event_no") or row["id"],
        "typeCode": row["upload_data_type"], "typeName": row.get("upload_data_type_name") or "未分类",
        "eventTime": str(row.get("event_time") or ""), "recheckTime": str(row.get("recheck_time") or ""),
        "status": status_of(row, codes), "storedStatus": row.get("recheck_status"),
        "supported": row["upload_data_type"] in codes,
        "hasOriginalRelation": bool(row.get("has_original")),
        "thumbnail": f"/workbench-api/events/{row['id']}/thumbnail",
    }


@app.middleware("http")
async def local_boundary(request: Request, call_next):
    if request.url.path.startswith("/workbench-api"):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "仅允许本机页面访问"}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "仅允许本机页面访问"}, status_code=403)
        if request.method == "POST" and (not origin or request.headers.get("x-workbench-action") != "recheck"):
            return JSONResponse({"detail": "请从对照页面提交复核"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/workbench-api") and "/images/" not in request.url.path and not request.url.path.endswith("/thumbnail"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(pymysql.MySQLError)
async def db_error(_request, _error):
    return JSONResponse({"detail": "测试数据源暂时无法访问，请稍后刷新"}, status_code=503)


@app.get("/workbench-api/events")
def list_events(status: str = "0", search: str = "", type_code: str = "", source: str = "all",
                page: int = Query(1, ge=1), limit: int = Query(24, ge=1, le=60)):
    if status not in {"all", "0", "1", "2", "3", "4"} or source not in {"all", "linked", "missing"}:
        raise HTTPException(400, "筛选条件无效")
    codes = support_codes()
    # Fetch compact metadata only; never load thousands of MEDIUMTEXT results for a list.
    rows = query("""SELECT e.id,e.event_no,e.upload_data_type,e.upload_data_type_name,e.event_time,
        e.recheck_status,e.recheck_time
        FROM bo_task_event e WHERE e.project_id=%s AND e.del_flag=0
        ORDER BY e.event_time DESC,e.id""", (PROJECT,))
    linked = {r["busi_id"] for r in query("""SELECT DISTINCT a.busi_id FROM sys_annex_source_target r
        JOIN sys_annex_file a ON a.id=r.target_id AND a.busi_type_code='SHOTIMG' AND a.del_flag=0
        JOIN bo_task_event e ON e.id=a.busi_id AND e.project_id=%s AND e.del_flag=0
        WHERE r.busi_type IN ('WAR_TO_AI','AR_TO_AI')""", (PROJECT,))}
    for row in rows:
        row["has_original"] = row["id"] in linked
    items = [brief(r, codes) for r in rows]
    counts = {key: sum(x["status"] == key for x in items) for key in ["0", "1", "2", "3", "4"]}
    types = sorted({(x["typeCode"], x["typeName"]) for x in items if x["typeCode"] and
                    (status == "all" or x["status"] == status or x["typeCode"] == type_code)}, key=lambda t: t[1])
    term = search.strip().lower()[:120]
    filtered = [x for x in items if (status == "all" or x["status"] == status)
                and (not type_code or x["typeCode"] == type_code)
                and (not term or term in (x["eventNo"] + x["id"] + x["typeName"]).lower())
                and (source == "all" or x["hasOriginalRelation"] == (source == "linked"))]
    total = len(filtered)
    return {"items": filtered[(page - 1) * limit:page * limit], "total": total, "page": page,
            "limit": limit, "counts": counts, "allCount": len(items), "capturedAt": now(),
            "projectName": "空天地一体化测试平台", "types": [{"code": c, "name": n} for c, n in types],
            "submitEnabled": SUBMIT_ENABLED}


def shots_for(row):
    shots = query("""SELECT id,file_path,imgxy,image_resolution,busi_type_code FROM sys_annex_file
        WHERE busi_id=%s AND del_flag=0 AND busi_type_code='SHOTIMG' ORDER BY id""", (row["id"],))
    if not shots:
        shots = query("""SELECT id,file_path,imgxy,image_resolution,busi_type_code FROM sys_annex_file
            WHERE busi_id=%s AND del_flag=0 AND busi_type_code='PRIMARYIMG' ORDER BY id""", (row["id"],))
    return shots


def original_for(shot):
    rels = query("SELECT source_id,busi_type FROM sys_annex_source_target WHERE target_id=%s AND busi_type IN ('WAR_TO_AI','AR_TO_AI')", (shot["id"],))
    ids = set()
    for r in rels:
        edges = query("SELECT source_id,target_id FROM sys_annex_source_target WHERE busi_type='WAR_TO_AR' AND (source_id=%s OR target_id=%s)", (r["source_id"], r["source_id"]))
        normalized = {x["source_id"] for x in edges if x["target_id"] == r["source_id"] or x["source_id"] == r["source_id"]}
        if not normalized and r["busi_type"] == "WAR_TO_AI":
            normalized = {r["source_id"]}
        if len(normalized) != 1:
            return None, "原图关联存在歧义，暂不配对"
        ids.update(normalized)
    if len(ids) != 1:
        return None, "尚未找到关联原图" if not ids else "关联了多张原图，暂不配对"
    rows = query("SELECT id,file_path,image_resolution FROM sys_annex_file WHERE id=%s AND del_flag=0", (next(iter(ids)),))
    if not rows or not rows[0]["file_path"]:
        return None, "原图附件记录缺失"
    return rows[0], "已找到关联原图；需对照画面确认是否为同一场景"


def finite(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def normalized_box(value):
    if not isinstance(value, list) or len(value) != 4:
        return None
    b = [finite(x) for x in value]
    if any(x is None for x in b) or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
        return None
    return b


def candidates_for(raw_boxes, result, manifest):
    raw_boxes = raw_boxes if isinstance(raw_boxes, list) else []
    records = result.get("results") or result.get("resultList") or []
    records = records if isinstance(records, list) else []
    dim = re.findall(r"\d+", str(manifest or ""))
    size = [float(x) for x in dim[:2]] if len(dim) >= 2 else [0, 0]
    out = []
    for i in range(max(len(records), len(raw_boxes))):
        initial = raw_boxes[i] if i < len(raw_boxes) and isinstance(raw_boxes[i], dict) else {}
        rec = records[i] if i < len(records) and isinstance(records[i], dict) else {}
        audit = rec.get("cleanSource") or {}
        recovery = audit.get("geometryRecovery") or {}
        box = normalized_box(rec.get("original_bbox") or rec.get("originalBbox"))
        geometry = "复核保存的位置" if box else "缺少完整框坐标"
        if recovery.get("applied"):
            geometry = "已从初审框恢复位置"
        elif box and audit.get("submittedMatchQuality", {}).get("bboxSource") == "legacy_default_extent":
            geometry = "历史估计框，仅供定位"
        if not box:
            fw, fh = finite(initial.get("frame_width")) or size[0], finite(initial.get("frame_height")) or size[1]
            x, y = finite(initial.get("x")), finite(initial.get("y"))
            w, h = finite(initial.get("width")), finite(initial.get("height"))
            if fw and fh and x is not None and y is not None and w and h:
                box = normalized_box([max(0, x/fw), max(0, y/fh), min(1, (x+w)/fw), min(1, (y+h)/fh)])
                if box:
                    geometry = "初审记录的完整框"
        decision = str(rec.get("decision") or "pending").lower()
        if decision not in {"confirmed", "rejected", "uncertain"}:
            decision = "pending"
        out.append({"index": i + 1, "className": initial.get("class_name") or rec.get("original_class") or rec.get("originalClass") or "未记录类别",
                    "initialConfidence": finite(initial.get("confidence", rec.get("original_confidence", rec.get("originalConfidence")))),
                    "bbox": box, "geometry": geometry, "decision": decision,
                    "reason": rec.get("reasoning") or "尚未生成复核结果", "reasonCode": rec.get("reasonCode"),
                    "model": rec.get("modelVersion"), "cleanApplied": audit.get("applied") is True,
                    "cleanRequested": audit.get("requested") is True,
                    "pairPassed": (audit.get("matchQuality") or {}).get("passed"),
                    "baseDecision": audit.get("baseDecision"), "adapterDecision": audit.get("adapterDecision"),
                    "personManual": audit.get("personAutoConfirmEnabled") is False and decision == "uncertain" and audit.get("baseDecision") == "CONFIRMED",
                    "audit": audit})
    return out


@app.get("/workbench-api/events/{eid}")
def event_detail(eid: str):
    row = event_row(eid)
    codes = support_codes()
    shots = shots_for(row)
    images = []
    event_result = decode(row.get("recheck_result"), {})
    event_result = event_result if isinstance(event_result, dict) else {}
    for shot in shots:
        original, note = original_for(shot)
        records = query("""SELECT recheck_result,recheck_time FROM bo_task_event_image_recheck_result
            WHERE event_id=%s AND annex_id=%s AND del_flag=0 ORDER BY recheck_time DESC,id DESC LIMIT 1""", (eid, shot["id"]))
        result = decode(records[0]["recheck_result"], {}) if records else {}
        result = result if isinstance(result, dict) else {}
        if not result and (event_result.get("sourceAnnexId") == shot["id"] or len(shots) == 1):
            result = event_result
        raw_boxes = decode(shot.get("imgxy"), [])
        if not raw_boxes and len(shots) == 1:
            raw_boxes = decode(row.get("imgxy"), [])
        candidates = candidates_for(raw_boxes, result, shot.get("image_resolution"))
        images.append({"id": shot["id"], "role": shot["busi_type_code"],
            "annotatedUrl": f"/workbench-api/events/{eid}/images/{shot['id']}",
            "originalUrl": f"/workbench-api/events/{eid}/images/{original['id']}" if original else None,
            "sourceNote": note, "originalId": original["id"] if original else None,
            "recordedResolution": shot.get("image_resolution"), "candidates": candidates,
            "recheckTime": str(records[0]["recheck_time"] or "") if records else str(row.get("recheck_time") or ""),
            "hasResult": bool(result), "summary": result.get("summary"), "rawResult": result})
    receipt = submission_state(eid, row)
    return {**brief(row, codes), "images": images, "capturedAt": now(), "submission": receipt,
            "submitEnabled": SUBMIT_ENABLED,
            "platformUrl": TEST_ORIGIN + "/afmap/#/eventCenter",
            "resultIsHistorical": status_of(row, codes) != "2" and any(x["hasResult"] for x in images)}


@app.get("/workbench-api/events/{eid}/export")
def export_event(eid: str):
    detail = event_detail(eid)
    name = re.sub(r"[^\w.-]", "_", detail["eventNo"]) + "-复核对照.json"
    return Response(json.dumps(detail, ensure_ascii=False, indent=2), media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename*=UTF-8''" + quote(name)})


def stored_image(path: str):
    parsed = urlparse(path)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme != "https" or parsed.netloc != urlparse(TEST_ORIGIN).netloc:
            raise HTTPException(422, "图片地址不属于当前测试环境")
        url = path
    else:
        p = "/" + path.lstrip("/")
        if p.startswith(("/local-src/", "/minio-src/")):
            url = TEST_ORIGIN + quote(p, safe="/%")
        else:
            url = TEST_ORIGIN + ("/local-src" if p.startswith("/upload/") else "/minio-src") + quote(p, safe="/%")
    if not urlparse(url).path.startswith(("/local-src/", "/minio-src/")) or ".." in urlparse(url).path:
        raise HTTPException(422, "图片路径无效")
    cache = DATA / "images" / hashlib.sha256(url.encode()).hexdigest()
    # Short cache avoids permanently hiding replaced test attachments.
    if cache.exists() and time.time() - cache.stat().st_mtime < 3600:
        content = cache.read_bytes()
    else:
        try:
            with httpx.stream("GET", url, verify=TLS_VERIFY, timeout=40, follow_redirects=False, trust_env=False) as response:
                if response.status_code != 200:
                    raise HTTPException(404, "测试环境图片不可用")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 64 * 1024 * 1024:
                        raise HTTPException(413, "图片过大")
                    chunks.append(chunk)
                content = b"".join(chunks)
            with Image.open(io.BytesIO(content)) as image:
                # DJI JPEG originals often carry MPO metadata; the first JPEG
                # frame is browser-decodable without recompressing source bytes.
                if image.format not in {"JPEG", "MPO", "PNG", "WEBP"}:
                    raise ValueError("unsupported image")
                image.verify()
            temp = cache.with_suffix(f".{threading.get_ident()}.tmp")
            temp.write_bytes(content)
            temp.replace(cache)
        except (httpx.HTTPError, ValueError, OSError, Image.DecompressionBombError):
            raise HTTPException(502, "图片读取失败，请稍后重试") from None
    return content


def image_response(content, thumbnail=False):
    with Image.open(io.BytesIO(content)) as image:
        if thumbnail:
            image.thumbnail((160, 110))
            out = io.BytesIO()
            image.convert("RGB").save(out, "JPEG", quality=78)
            return Response(out.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "private,max-age=60"})
        mime = "image/jpeg" if image.format == "MPO" else Image.MIME.get(image.format, "image/jpeg")
    return Response(content, media_type=mime, headers={"Cache-Control": "private,max-age=60"})


@app.get("/workbench-api/events/{eid}/thumbnail")
def thumbnail(eid: str):
    shots = shots_for(event_row(eid))
    if not shots:
        raise HTTPException(404, "没有初审图片")
    return image_response(stored_image(shots[0]["file_path"]), True)


@app.get("/workbench-api/events/{eid}/images/{aid}")
def event_image(eid: str, aid: str):
    event_id(aid)
    shots = shots_for(event_row(eid))
    for shot in shots:
        if aid == shot["id"]:
            return image_response(stored_image(shot["file_path"]))
        original, _ = original_for(shot)
        if original and aid == original["id"]:
            return image_response(stored_image(original["file_path"]))
    raise HTTPException(404, "图片不属于当前事件")


def result_fingerprint(row):
    return hashlib.sha256(str(row.get("recheck_result") or "").encode()).hexdigest()


def receipt_file(eid):
    return DATA / "submissions" / (event_id(eid) + ".json")


def save_receipt(eid, value):
    path = receipt_file(eid)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False))
    temp.chmod(0o600)
    temp.replace(path)


def submission_state(eid, row):
    path = receipt_file(eid)
    if not path.exists():
        return None
    receipt = decode(path.read_text(), {})
    if row.get("recheck_status") == "1":
        state = "running"
    elif (str(row.get("recheck_time") or "") != receipt.get("beforeTime") or result_fingerprint(row) != receipt.get("beforeHash")) and row.get("recheck_status") in {"2", "3"}:
        state = "completed" if row["recheck_status"] == "2" else "failed"
    else:
        state = receipt.get("state", "awaiting")
    return {"state": state, "submittedAt": receipt.get("submittedAt"), "message": receipt.get("message")}


def submit_worker(eid, receipt):
    try:
        headers = {"projectId": PROJECT, "timestamp": str(int(time.time() * 1000)), "Content-Length": "0"}
        token = os.getenv("KTD_REVIEW_BEARER_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
        response = httpx.post(TEST_ORIGIN + "/gis_server_a/recheckAudit/task/event/recheck/" + eid,
                              headers=headers, timeout=85, verify=TLS_VERIFY, follow_redirects=False, trust_env=False)
        body = decode(response.text, {})
        body = body if isinstance(body, dict) else {}
        if response.status_code in {401, 403, 404} or (response.status_code == 200 and (body.get("success") is False or body.get("code") not in {None, 0, 200, "0", "200"})):
            receipt.update(state="rejected", message="测试平台拒绝了请求，请到平台检查登录或事件状态")
        else:
            receipt.update(state="awaiting", message="请求已发送，正在等待测试平台保存结果；请勿重复提交")
        receipt["httpStatus"] = response.status_code
    except httpx.HTTPError:
        receipt.update(state="awaiting", message="连接已结束，平台可能仍在复核；页面会继续查询结果，请勿重复提交")
    finally:
        save_receipt(eid, receipt)
        with submit_lock:
            active_submissions.discard(eid)


@app.post("/workbench-api/events/{eid}/recheck", status_code=202)
def submit_recheck(eid: str):
    if not SUBMIT_ENABLED:
        raise HTTPException(403, "当前页面为只读模式")
    row = event_row(eid)
    if row["upload_data_type"] not in support_codes():
        raise HTTPException(409, "当前项目未为此类型启用复核")
    # Completed results are inspectable; this UI does not silently rerun them.
    if row.get("recheck_status") not in {None, "", "0", "3"}:
        raise HTTPException(409, "该事件已完成或正在复核，请刷新查看")
    with submit_lock:
        previous = submission_state(eid, row)
        if active_submissions or (previous and previous["state"] not in {"failed", "rejected"}):
            raise HTTPException(409, "已有请求在处理或等待平台回写，请勿重复提交")
        if query("SELECT id FROM bo_task_event WHERE project_id=%s AND del_flag=0 AND recheck_status='1' LIMIT 1", (PROJECT,)):
            raise HTTPException(409, "测试平台已有事件正在复核，请稍后再提交")
        receipt = {"submittedAt": now(), "beforeTime": str(row.get("recheck_time") or ""),
                   "beforeHash": result_fingerprint(row), "state": "awaiting", "message": "已提交，等待平台处理"}
        save_receipt(eid, receipt)
        active_submissions.add(eid)
        threading.Thread(target=submit_worker, args=(eid, receipt), daemon=True).start()
    return {"state": "awaiting", "message": "已提交当前事件，页面将自动更新复核结果"}


@app.get("/")
@app.get("/workbench.html")
def index():
    return FileResponse(DIST / "workbench.html", headers={"Cache-Control": "no-store"})


app.mount("/assets", StaticFiles(directory=DIST / "assets", check_dir=False), name="assets")
