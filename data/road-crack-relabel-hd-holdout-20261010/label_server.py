"""道路裂纹原帧复标页（复验 17 事件）（仅本机 127.0.0.1）。不显示原标签和模型结论，结果只写本目录 annotations.csv。

用法（仓库根目录）：python3 data/road-crack-relabel-hd-holdout-20261010/label_server.py --port 8801
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
LABELS = {'是裂纹', '不是裂纹', '看不清'}
LOCK = threading.Lock()


def keys():
    ks = sorted(p.stem for p in (HERE / 'images/compare').glob('*.jpg'))
    random.Random(20261011).shuffle(ks)
    return ks


def load():
    if not CSV_PATH.exists(): return {}
    with CSV_PATH.open(encoding='utf-8-sig') as f:
        return {r['key']: r for r in csv.DictReader(f)}


def save(rows):
    tmp = CSV_PATH.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['key', 'label', 'note', 'labeled_at'])
        w.writeheader()
        for k in keys():
            if k in rows: w.writerow(rows[k])
    tmp.replace(CSV_PATH)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype):
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self.send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif path == '/api/rows':
            done = load()
            rows = [{'key': k, 'label': done.get(k, {}).get('label', ''), 'note': done.get(k, {}).get('note', '')} for k in keys()]
            self.send(200, json.dumps({'rows': rows}, ensure_ascii=False).encode(), 'application/json')
        elif path.startswith('/images/') and '..' not in path:
            p = HERE / path.lstrip('/')
            if p.is_file(): self.send(200, p.read_bytes(), 'image/jpeg')
            else: self.send(404, b'not found', 'text/plain')
        else:
            self.send(404, b'not found', 'text/plain')

    def do_POST(self):
        if urlparse(self.path).path != '/api/label': return self.send(404, b'not found', 'text/plain')
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        if body.get('key') not in set(keys()) or body.get('label') not in LABELS:
            return self.send(400, b'bad request', 'text/plain')
        with LOCK:
            rows = load()
            rows[body['key']] = {'key': body['key'], 'label': body['label'], 'note': (body.get('note') or '')[:200],
                                 'labeled_at': dt.datetime.now().isoformat(timespec='seconds')}
            save(rows)
        self.send(200, b'{"ok":true}', 'application/json')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8801); a = ap.parse_args()
    print(f'道路裂纹原帧复标页（复验 17 事件）: http://127.0.0.1:{a.port}/\n共 {len(keys())} 框，已标 {len(load())}。结果只写 {CSV_PATH}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
