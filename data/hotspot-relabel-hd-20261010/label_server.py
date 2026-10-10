"""光斑原图复标页（仅本机 127.0.0.1）。不显示原标签和模型结论，结果只写本目录 annotations.csv。

用法（仓库根目录）：python3 data/hotspot-relabel-hd-20261010/label_server.py --port 8801
图片在 gitignore 的 images/compare 与 images/context，由服务器任务目录同步而来。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / 'annotations.csv'
BOXES_PATH = HERE / 'boxes.json'
LABELS = {'是光斑', '不是光斑', '看不清'}
LOCK = threading.Lock()
FIELDS = ['key', 'label', 'note', 'labeled_at']


def boxes():
    payload = json.loads(BOXES_PATH.read_text())
    rows = list(payload['boxes'])
    random.Random(payload.get('seed', 20261010)).shuffle(rows)
    return rows


def load():
    if not CSV_PATH.exists():
        return {}
    with CSV_PATH.open(encoding='utf-8-sig') as handle:
        return {row['key']: row for row in csv.DictReader(handle) if row.get('key')}


def save(rows):
    tmp = CSV_PATH.with_suffix('.tmp')
    order = [row['key'] for row in boxes()]
    with tmp.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for key in order:
            if key in rows:
                writer.writerow({field: rows[key].get(field, '') for field in FIELDS})
    tmp.replace(CSV_PATH)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self.send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif path == '/api/rows':
            done = load()
            rows = []
            for row in boxes():
                saved = done.get(row['key'], {})
                rows.append({
                    'key': row['key'],
                    'label': saved.get('label', ''),
                    'note': saved.get('note', ''),
                    'sample_source': row.get('sample_source') or 'new',
                    'duplicate_of': row.get('duplicate_of') or '',
                    'source_kind': row.get('source_kind') or '',
                    'source_width': row.get('source_width') or 0,
                    'source_height': row.get('source_height') or 0,
                    'source_visual': row.get('source_visual') or '',
                })
            self.send(200, json.dumps({'rows': rows}, ensure_ascii=False).encode(), 'application/json')
        elif path.startswith('/images/') and '..' not in path:
            image = HERE / path.lstrip('/')
            if image.is_file():
                self.send(200, image.read_bytes(), 'image/jpeg')
            else:
                self.send(404, b'not found', 'text/plain')
        else:
            self.send(404, b'not found', 'text/plain')

    def do_POST(self):
        if urlparse(self.path).path != '/api/label':
            return self.send(404, b'not found', 'text/plain')
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        known = {row['key'] for row in boxes()}
        if body.get('key') not in known or body.get('label') not in LABELS:
            return self.send(400, b'bad request', 'text/plain')
        with LOCK:
            rows = load()
            rows[body['key']] = {
                'key': body['key'],
                'label': body['label'],
                'note': (body.get('note') or '')[:200],
                'labeled_at': dt.datetime.now().isoformat(timespec='seconds'),
            }
            save(rows)
        self.send(200, b'{"ok":true}', 'application/json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8801)
    args = parser.parse_args()
    print(f'光斑原图复标页: http://127.0.0.1:{args.port}/\n共 {len(boxes())} 框，已标 {len(load())}。结果只写 {CSV_PATH}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
