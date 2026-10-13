"""未曝光人员的人工待标入口。

只接收初审原框或单独列出的人工补框。不调用模型，不预填标签，
不把历史回归样本、参考图或时间接近的任务算成新的独立验证。
页面只监听本机，结果只写当前清单目录。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LABELS = ('是人员', '不是人员', '看不清')
BOX_ORIGINS = ('initial_review', 'human_added')
FRAME_FIELDS = ('clean_image_sha256', 'source_pts_ns', 'stream_epoch', 'geometry_evidence')
MODEL_FIELDS = (
    'raw_decision', 'model_decision', 'decision', 'reasoning', 'reason', 'baseline',
    'baseline_reason', 'luna', 'confidence', 'business_decision', 'observed_object',
    'image_limitation', 'prompt', 'outcome', 'result',
)
CSV_FIELDS = (
    'candidate_id', 'task_id', 'event_id', 'box_origin', 'class_name', 'bbox',
    'image_sha256', 'image_file', 'frame_binding', 'frame_gap', 'label', 'note', 'reviewer',
)
API_FIELDS = CSV_FIELDS
_SHA = re.compile(r'^[0-9a-f]{64}$')
_SAFE_IMAGE = re.compile(r'^images/[A-Za-z0-9._-]+$')
_PAGE = Path(__file__).with_name('person_label_page.html')
_WRITE_LOCK = threading.Lock()


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ''


def _present(value) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value not in (None, '', [], {})


def legal_bbox(bbox) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    return 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


def normalize_exclusions(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}

    def values(name):
        found = raw.get(name) or []
        if not isinstance(found, (list, tuple, set)):
            return set()
        return {_text(item) for item in found if _text(item)}

    return {
        'task_ids': values('task_ids'),
        'unit_ids': values('unit_ids'),
        'event_ids': values('event_ids'),
        'image_sha256': {item.lower() for item in values('image_sha256')},
    }


def _blocked_tasks(candidates, exclusions) -> set[str]:
    blocked = set(exclusions['task_ids'])
    for item in candidates:
        if not isinstance(item, dict):
            continue
        task = _text(item.get('task_id'))
        if not task:
            continue
        sha = _text(item.get('image_sha256')).lower()
        if (_text(item.get('unit_id')) in exclusions['unit_ids']
                or _text(item.get('event_id')) in exclusions['event_ids']
                or (sha and sha in exclusions['image_sha256'])):
            blocked.add(task)
    return blocked


def _image_ready(item, image_root: Path | None) -> bool:
    relative = _text(item.get('image_file'))
    if image_root is None or not _SAFE_IMAGE.fullmatch(relative):
        return False
    path = (image_root / relative).resolve()
    return path.is_relative_to(image_root.resolve()) and path.is_file()


def _frame_gaps(item) -> list[str]:
    gaps = []
    if _text(item.get('frame_binding')) != 'same_frame_verified':
        gaps.append('没有同帧绑定')
    evidence = item.get('frame_evidence') if isinstance(item.get('frame_evidence'), dict) else {}
    missing = [name for name in FRAME_FIELDS if not _present(evidence.get(name))]
    if missing:
        gaps.append('同帧依据不足：' + '、'.join(missing))
    return gaps


def _model_fields(item) -> list[str]:
    found = []
    for name in MODEL_FIELDS:
        value = item.get(name)
        if value not in (None, '', [], {}):
            found.append(name)
    return found


def prepare_queue(candidates, exclusions=None, *, image_root: Path | None = None) -> dict:
    """筛出可人工验证的人员框。标签保持空白，人工补框不计入初审原框。"""
    exclusions = normalize_exclusions(exclusions)
    candidates = candidates if isinstance(candidates, list) else []
    blocked = _blocked_tasks(candidates, exclusions)
    seen = set()
    initial, human_added, held, rejected = [], [], [], []
    for item in candidates:
        if not isinstance(item, dict):
            rejected.append({'reason': '记录不是对象'})
            continue
        candidate_id = _text(item.get('candidate_id'))
        task_id = _text(item.get('task_id'))
        if not candidate_id:
            rejected.append({'reason': '缺少候选编号'})
            continue
        if candidate_id in seen:
            rejected.append({'candidate_id': candidate_id, 'task_id': task_id, 'reason': '候选编号重复'})
            continue
        seen.add(candidate_id)
        if task_id and task_id in blocked:
            rejected.append({'candidate_id': candidate_id, 'task_id': task_id, 'reason': '任务与已曝光来源重叠'})
            continue
        if _text(item.get('class_name')) != '人员':
            rejected.append({'candidate_id': candidate_id, 'task_id': task_id, 'reason': '不是人员初审框'})
            continue
        model_fields = _model_fields(item)
        if model_fields:
            rejected.append({'candidate_id': candidate_id, 'task_id': task_id, 'reason': '带有自动结论'})
            continue
        if _text(item.get('label')):
            rejected.append({'candidate_id': candidate_id, 'task_id': task_id, 'reason': '已带标签'})
            continue
        origin = _text(item.get('box_origin'))
        gaps = []
        if origin not in BOX_ORIGINS:
            gaps.append('框来源未标明，不能把补框当成初审原框')
        if not task_id:
            gaps.append('缺少任务或航次编号')
        if not _text(item.get('event_id')):
            gaps.append('缺少事件编号')
        if not legal_bbox(item.get('bbox')):
            gaps.append('框坐标不合法')
        sha = _text(item.get('image_sha256')).lower()
        if not _SHA.fullmatch(sha):
            gaps.append('缺少可核验图片摘要')
        if not _image_ready(item, image_root):
            gaps.append('缺少可核验图片')
        gaps.extend(_frame_gaps(item))
        row = {
            'candidate_id': candidate_id,
            'task_id': task_id,
            'event_id': _text(item.get('event_id')),
            'box_origin': origin,
            'class_name': '人员',
            'bbox': json.dumps(item.get('bbox'), ensure_ascii=False, separators=(',', ':')) if legal_bbox(item.get('bbox')) else '',
            'image_sha256': sha,
            'image_file': _text(item.get('image_file')) if _image_ready(item, image_root) else '',
            'frame_binding': 'same_frame_verified' if not _frame_gaps(item) else 'missing',
            'frame_gap': '；'.join(gaps),
            'label': '',
            'note': '',
            'reviewer': '',
        }
        if gaps:
            held.append(row)
        elif origin == 'human_added':
            human_added.append(row)
        else:
            initial.append(row)
    return {
        'initial_review': initial,
        'human_added': human_added,
        'held': held,
        'rejected': rejected,
        'summary': summarize(initial, human_added, held, rejected),
    }


def summarize(initial, human_added, held, rejected) -> dict:
    labels = {label: sum(row['label'] == label for row in initial) for label in LABELS}
    labels['未标'] = sum(not row['label'] for row in initial)
    covered = labels['是人员'] > 0 and labels['不是人员'] > 0 and labels['未标'] == 0 and bool(initial)
    reasons = {}
    for row in rejected:
        reasons[row['reason']] = reasons.get(row['reason'], 0) + 1
    holds = {}
    for row in held:
        holds[row['frame_gap']] = holds.get(row['frame_gap'], 0) + 1
    return {
        'purpose': '未曝光人员人工验证。历史回归样本不进入本清单，看不清不计入正负例。',
        'initial_review_ready': len(initial),
        'human_added_ready': len(human_added),
        'held': len(held),
        'rejected': len(rejected),
        'labels': labels,
        'positive_negative_covered': covered,
        'independent_validation_ready': covered,
        'historical_regression_separate': True,
        'rejected_reasons': reasons,
        'hold_reasons': holds,
    }


def apply_label(rows, candidate_id, label, note='', reviewer=''):
    if label not in LABELS and label != '':
        raise ValueError('标注选项无效')
    matches = [row for row in rows if row.get('candidate_id') == candidate_id]
    if len(matches) != 1:
        raise ValueError('找不到唯一候选框')
    matches[0]['label'] = label
    matches[0]['note'] = note.replace('\r\n', '\n').strip()[:500]
    matches[0]['reviewer'] = _text(reviewer)[:40]
    return matches[0]


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        if tuple(reader.fieldnames) != CSV_FIELDS:
            raise ValueError('待标清单字段与模板不一致')
        return [{field: row.get(field) or '' for field in CSV_FIELDS} for row in reader]


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in CSV_FIELDS})
    temporary.replace(path)


def write_package(out: Path, prepared: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'annotations.csv', prepared['initial_review'])
    write_csv(out / 'human_added.csv', prepared['human_added'])
    payload = dict(prepared['summary'], held_rows=prepared['held'], rejected_rows=prepared['rejected'])
    (out / 'gaps.json').write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (out / 'index.html').write_text(_PAGE.read_text(encoding='utf-8'), encoding='utf-8')


def public_row(row) -> dict:
    return {field: row.get(field, '') for field in API_FIELDS}


class Handler(BaseHTTPRequestHandler):
    package = Path('.')

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_json(self, payload, status=HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split('?', 1)[0]
        package = self.package.resolve()
        if path in {'/', '/index.html'}:
            page = (package / 'index.html').read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if path == '/api/rows':
            self.send_json({'rows': [public_row(row) for row in read_csv(package / 'annotations.csv')]})
            return
        if path == '/api/human-rows':
            self.send_json({'rows': [public_row(row) for row in read_csv(package / 'human_added.csv')]})
            return
        if path == '/api/status':
            self.send_json(json.loads((package / 'gaps.json').read_text(encoding='utf-8')))
            return
        relative = path.lstrip('/')
        if not _SAFE_IMAGE.fullmatch(relative):
            self.send_json({'error': 'not found'}, HTTPStatus.NOT_FOUND)
            return
        file_path = (package / relative).resolve()
        if not file_path.is_relative_to(package) or not file_path.is_file():
            self.send_json({'error': 'not found'}, HTTPStatus.NOT_FOUND)
            return
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        path = self.path.split('?', 1)[0]
        if path not in {'/api/label', '/api/human-label'}:
            self.send_json({'error': 'not found'}, HTTPStatus.NOT_FOUND)
            return
        csv_path = self.package / ('annotations.csv' if path == '/api/label' else 'human_added.csv')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 100_000:
                raise ValueError('请求体无效')
            payload = json.loads(self.rfile.read(length))
            with _WRITE_LOCK:
                rows = read_csv(csv_path)
                saved = apply_label(
                    rows, _text(payload.get('candidate_id')), _text(payload.get('label')),
                    str(payload.get('note') or ''), str(payload.get('reviewer') or ''))
                write_csv(csv_path, rows)
            self.send_json({'ok': True, 'candidate_id': saved['candidate_id'], 'label': saved['label']})
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            self.send_json({'error': str(exc)}, HTTPStatus.BAD_REQUEST)


def serve(package: Path, port: int) -> None:
    read_csv(package / 'annotations.csv')
    read_csv(package / 'human_added.csv')
    handler = type('BoundHandler', (Handler,), {'package': package.resolve()})
    server = ThreadingHTTPServer(('127.0.0.1', port), handler)
    print(f'人员待标入口: http://127.0.0.1:{port}/')
    print(f'清单目录: {package}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n待标入口已停止')
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description='准备或打开未曝光人员人工待标入口')
    parser.add_argument('command', choices=('prepare', 'serve'))
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--exclude', type=Path)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--port', type=int, default=8798)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.out is None:
            raise SystemExit('prepare 必须给出 --out')
        candidates = json.loads(args.candidates.read_text(encoding='utf-8')) if args.candidates else []
        exclusions = json.loads(args.exclude.read_text(encoding='utf-8')) if args.exclude else {}
        prepared = prepare_queue(candidates, exclusions, image_root=args.image_root)
        write_package(args.out, prepared)
        print(json.dumps(prepared['summary'], ensure_ascii=False))
        return
    if args.package is None:
        raise SystemExit('serve 必须给出 --package')
    serve(args.package, args.port)


if __name__ == '__main__':
    main()
