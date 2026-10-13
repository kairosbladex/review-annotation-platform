#!/usr/bin/env python3
"""对比 Luna 看 960×720 带框截图 与 看原视频同一时刻 1920×1080 原帧 的复检结果（同一提示词、同一 worker）。

金标与截图结果取 data/luna-eval-20261008/per-box.json；原帧结果取 --hd-dir（每事件一个 JSON）。
看不清的金标框、原帧缺失（投影出画面）的框不计分，单独列出。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

TRUE = {'真目标', '真车'}


def state(raw: str, truth: bool, uncertain_as: str) -> str:
    if raw == 'uncertain' and uncertain_as == 'manual':
        return 'manual'
    pred = raw == 'true'
    return 'right' if pred == truth else 'wrong'


def summarize(rows: list[dict], key: str) -> dict:
    out = {}
    for policy in ('false', 'manual'):
        c = collections.Counter(state(r[key], r['truth'], policy) for r in rows)
        auto = c['right'] + c['wrong']
        out[f'uncertain_as_{policy}'] = dict(right=c['right'], wrong=c['wrong'], manual=c['manual'],
                                             accuracy=round(c['right'] / auto, 3) if auto else None)
    out['raw_vs_gold'] = {f'{raw}/{"真" if t else "假"}': n for (raw, t), n in sorted(
        collections.Counter((r[key], r['truth']) for r in rows).items())}
    out['true_rejected'] = sum(r['truth'] and r[key] != 'true' for r in rows)
    out['false_passed'] = sum((not r['truth']) and r[key] == 'true' for r in rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-box', type=Path, default=Path('data/luna-eval-20261008/per-box.json'))
    ap.add_argument('--hd-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='report.json')
    ap.add_argument('--per-box-out', type=Path, required=True)
    args = ap.parse_args()

    gold = {r['candidate_id']: r for r in json.loads(args.per_box.read_text())}
    hd, failed = {}, []
    for p in sorted(args.hd_dir.glob('*.json')):
        item = json.loads(p.read_text())
        resp = item['response']
        if not resp.get('success'):
            failed.append(item['event_id'])
            continue
        for cid, res in zip(item['candidate_ids'], resp['results']):
            hd[cid] = dict(raw=res['rawDecision'], reasoning=res['reasoning'],
                           observed=res.get('observedObject'), limitation=res.get('imageLimitation'),
                           elapsed=item['elapsed'])

    rows, missing = [], []
    for cid, g in gold.items():
        if not g['scorable']:
            continue
        if cid not in hd:
            missing.append(cid)
            continue
        rows.append(dict(candidate_id=cid, group=g['group'], class_name=g['class_name'], gold=g['gold'],
                         truth=g['gold'] in TRUE, shot=g['luna_raw'], hd=hd[cid]['raw'],
                         hd_reasoning=hd[cid]['reasoning'], hd_observed=hd[cid]['observed'],
                         hd_limitation=hd[cid]['limitation'], shot_reasoning=g['luna_reasoning']))

    report = dict(scored_boxes=len(rows), missing_hd=missing, failed_events=failed)
    for group in ('person', 'vehicle'):
        sub = [r for r in rows if r['group'] == group]
        flip = collections.Counter(
            f"{state(r['shot'], r['truth'], 'false')}->{state(r['hd'], r['truth'], 'false')}" for r in sub)
        report[group] = dict(n=len(sub), screenshot=summarize(sub, 'shot'), hd=summarize(sub, 'hd'),
                             screenshot_to_hd_online_rule=dict(flip))
    times = sorted({v['elapsed'] for v in hd.values()})
    if times:
        report['hd_request_seconds'] = dict(median=round(times[len(times) // 2], 1),
                                            p90=round(times[int(len(times) * .9)], 1), max=round(times[-1], 1))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    args.per_box_out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
