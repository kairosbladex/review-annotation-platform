"""道路裂纹卡评测打分（本地，不调用模型）。

需要本机人工标注导入结果 data/road-crack-labeling-20261009/labels-road-crack.json（按标注包规则不入库）。
用法（仓库根目录）：python3 data/road-crack-card-eval-20261010/score.py
"""
from __future__ import annotations
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ['crack-p001-a', 'crack-p001-b', 'crack-v2-a', 'crack-v2-b']


def load(run):
    out, failed = {}, 0
    for f in sorted((HERE / 'runs' / run / 'out').glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        failed += not r['success']
        for x in r['results']: out[x['key']] = (r['unit'], x['raw'])
    return out, failed


def main():
    labels = json.loads((ROOT / 'data/road-crack-labeling-20261009/labels-road-crack.json').read_text(encoding='utf-8'))
    split = json.loads((HERE / 'split.json').read_text(encoding='utf-8'))
    res = {}
    for run in RUNS:
        preds, failed = load(run); res[run] = preds
        for part in ('test', 'design'):
            c = collections.Counter()
            for k, lab in labels.items():
                if k not in preds or split[preds[k][0]] != part: continue
                raw = preds[k][1]; c[raw] += 1; c['gold_' + lab['label']] += 1
                ok = (raw == 'true') == (lab['label'] == 'true'); c['right'] += ok
                if not ok: c['true_rejected' if lab['label'] == 'true' else 'false_released'] += 1
            n = c['true'] + c['false'] + c['uncertain']
            print(f"{run:13s} {part:6s} 准确率 {c['right'] / n:.1%}  真裂纹被驳回 {c['true_rejected']}/{c['gold_true']}  "
                  f"非裂纹被放过 {c['false_released']}/{c['gold_false']}  是/不是/看不清 {c['true']}/{c['false']}/{c['uncertain']}  失败图 {failed}")
    for a, b in (('crack-p001-a', 'crack-p001-b'), ('crack-v2-a', 'crack-v2-b')):
        flips = sum((res[a][k][1] == 'true') != (res[b][k][1] == 'true') for k in labels if k in res[a] and k in res[b])
        print(f'{a} vs {b} 判真/判假翻转 {flips}/{len(labels)}')


if __name__ == '__main__':
    main()
