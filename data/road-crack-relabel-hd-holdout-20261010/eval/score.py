"""17 事件复验打分（本地，不调用模型）：python3 data/road-crack-relabel-hd-holdout-20261010/eval/score.py"""
import collections
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = ['shot-p000-a', 'shot-p000-b', 'shot-card-a', 'shot-card-b', 'clean-p000-a', 'clean-p000-b', 'clean-card-a', 'clean-card-b']


def main():
    gold = {}
    for r in csv.DictReader(open(HERE.parent / 'annotations.csv', encoding='utf-8-sig')):
        k = r['key'][:r['key'].rfind('_')] + ':' + r['key'][r['key'].rfind('_') + 1:]
        gold[k] = r['label']
    res = {}
    for run in RUNS:
        out, failed = {}, 0
        for f in (HERE / 'runs' / run / 'out').glob('*.json'):
            d = json.loads(f.read_text(encoding='utf-8')); failed += not d['success']
            for x in d['results']: out[x['key']] = x['raw']
        res[run] = (out, failed)
    binary = [k for k, v in gold.items() if v != '看不清' and all(k in res[r][0] for r in RUNS)]
    unclear = [k for k, v in gold.items() if v == '看不清' and all(k in res[r][0] for r in RUNS)]
    nt = sum(gold[k] == '是裂纹' for k in binary)
    print(f'有明确标签且 8 次都有结果的框 {len(binary)}（是 {nt} / 不是 {len(binary) - nt}）；人工看不清 {len(unclear)}')
    for run in RUNS:
        out, failed = res[run]; c = collections.Counter()
        for k in binary:
            raw = out[k]; c[raw] += 1
            ok = (raw == 'true') == (gold[k] == '是裂纹'); c['right'] += ok
            if not ok: c['TR' if gold[k] == '是裂纹' else 'FR'] += 1
        u = collections.Counter(out[k] for k in unclear)
        print(f"{run:13s} 准确率 {c['right'] / len(binary):.1%}  真裂纹被驳回 {c['TR']}/{nt}  非裂纹被放过 {c['FR']}/{len(binary) - nt}  "
              f"是/不是/看不清 {c['true']}/{c['false']}/{c['uncertain']}  人工看不清框的模型结论 是{u['true']}/不是{u['false']}/看不清{u['uncertain']}  失败图 {failed}")


if __name__ == '__main__':
    main()
