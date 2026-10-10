#!/usr/bin/env python3
"""用更正后标签重算三处 Luna 评测。

评分函数直接复用 luna-prompt-autotune 的官方 src/metrics.py，口径与线上一致：
只有 true 放过，false 和 uncertain 都算驳回。

用法（先按说明把三份数据解到本地）：
    python3 rescore.py \
        --autotune /tmp/luna_final \
        --clean /tmp/luna_clean \
        --building /tmp/luna_bld \
        --road-crack-labels ../../data/road-crack-labeling-20261009/labels-road-crack.json

数据准备：
    # 逐图原始输出 + holdout 结果
    git -C <yoloe-repo> archive feature/luna-class-prompt-release \
        data/luna-prompt-autotune-final-20261009 | tar -x -C /tmp/luna_final --strip-components=2
    tar xzf /tmp/luna_final/raw-runs.tar.gz -C /tmp/luna_final/runs --strip-components=1
    # 违建单卡
    git -C <yoloe-repo> archive tests/luna-building-card-ablation \
        data/luna-building-card-ablation-20261010 | tar -x -C /tmp/luna_bld --strip-components=2
    # 干净图
    git archive tests/luna-clean-frame-eval \
        data/luna-clean-frame-eval-20261009 | tar -x -C /tmp/luna_clean --strip-components=2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 本轮人工复核确认的 3 条更正，全部 false -> true，且全部落在 dev。
CORRECTIONS = {
    'released_fp:61b6a8b48d6846b1aca70f777558012d:004': 'true',
    'released_fp:f7041ee1c1aa4feb952d95667e47b235:001': 'true',
    'released_fp:61b6a8b48d6846b1aca70f777558012d:005': 'true',
}
VERSIONS = ['p000', 'p000b', 'p001', 'p002', 'p003', 'p004', 'p005']
BUILDING_RUNS = ['baseline-a', 'baseline-b', 'building-a', 'building-b']


def pct(x) -> str:
    return '—' if x is None else f'{x * 100:.2f}%'


def res_from_outdir(d: Path) -> dict:
    res = {}
    for f in sorted(d.glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        for x in r.get('results', []):
            res[x['key']] = x['raw']
    return res


def res_from_decisions(path: Path) -> dict:
    d = json.loads(path.read_text(encoding='utf-8'))
    return {k: v['raw'] for k, v in d['decisions'].items()}


def show(title: str, m: dict, classes=None) -> None:
    print(f'\n--- {title}')
    for cls, c in m['by_class'].items():
        if classes and cls not in classes:
            continue
        print(f"    {cls:6s} n={c['n']:4d} acc={pct(c['accuracy']):>8s} "
              f"true_rej={c['true_rejected']:3d} false_rel={c['false_released']:3d} "
              f"unc={c['uncertain']:3d}")
    o = m['overall']
    print(f"    {'总体':6s} n={o['n']:4d} acc={pct(o['accuracy']):>8s} "
          f"true_rej={o['true_rejected']:3d} false_rel={o['false_released']:3d} "
          f"unc={o['uncertain']:3d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--autotune', type=Path, default=Path('/tmp/luna_final'))
    ap.add_argument('--clean', type=Path, default=Path('/tmp/luna_clean'))
    ap.add_argument('--building', type=Path, default=Path('/tmp/luna_bld'))
    ap.add_argument('--road-crack-labels', type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.autotune / 'src'))
    import metrics  # noqa: E402

    labels = json.loads((args.autotune / 'assets/labels.json').read_text(encoding='utf-8'))
    new_labels = {k: dict(v) for k, v in labels.items()}
    for k, v in CORRECTIONS.items():
        assert new_labels[k]['label'] == 'false', f'{k} 原标签不是 false，更正前提不成立'
        new_labels[k]['label'] = v

    print('=' * 90)
    print('更正框所在划分（必须是 dev 才会不影响 holdout）')
    print('=' * 90)
    for k in CORRECTIONS:
        print(f'  {k}: split={labels[k]["split"]} class={labels[k]["class_name"]}')
    holdout_keys = {k for k, v in labels.items() if v['split'] == 'holdout'}
    print(f'  更正框是否落在 holdout: {bool(set(CORRECTIONS) & holdout_keys)}')

    dev_res = {v: res_from_outdir(args.autotune / f'runs/{v}/out') for v in VERSIONS}

    print()
    print('=' * 90)
    print('自检：旧标签复现原始记录')
    print('=' * 90)
    checks = [('p000 holdout', metrics.score(dev_res['p000'], labels, 'holdout')['overall']['accuracy'], 0.7562),
              ('p000 dev', metrics.score(dev_res['p000'], labels, 'dev')['overall']['accuracy'], 0.7299),
              ('p000b dev', metrics.score(dev_res['p000b'], labels, 'dev')['overall']['accuracy'], 0.7124),
              ('p001 dev', metrics.score(dev_res['p001'], labels, 'dev')['overall']['accuracy'], 0.7608),
              ('p000b-holdout', metrics.score(
                  res_from_decisions(args.autotune / 'human/final-20261009/p000b-holdout.json'),
                  labels, 'holdout')['overall']['accuracy'], 0.7675)]
    for name, got, exp in checks:
        flag = '✅' if abs(got - exp) < 5e-5 else f'❌ 期望 {exp}'
        print(f'  {name}: {got} {flag}')

    print()
    print('=' * 90)
    print('holdout 两次运行（旧 vs 新）')
    print('=' * 90)
    hold = [('p000', dev_res['p000'])] + [
        (r, res_from_decisions(args.autotune / f'human/final-20261009/{r}.json'))
        for r in ['p000b-holdout', 'p002-holdout-a', 'p002-holdout-b']]
    for name, res in hold:
        o = metrics.score(res, labels, 'holdout')
        n = metrics.score(res, new_labels, 'holdout')
        show(f'holdout {name} 旧', o)
        print(f'    新旧是否一致: {o == n}')

    print()
    print('=' * 90)
    print('违建单卡四轮（holdout，旧 vs 新）')
    print('=' * 90)
    for rid in BUILDING_RUNS:
        res = res_from_decisions(args.building / f'results/{rid}.json')
        o = metrics.score(res, labels, 'holdout')
        n = metrics.score(res, new_labels, 'holdout')
        show(f'违建 {rid} 旧', o, classes=['违建'])
        print(f'    新旧是否一致: {o == n}')

    print()
    print('=' * 90)
    print('干净图各条件（旧 vs 新）')
    print('=' * 90)
    perbox = json.loads((args.clean / 'per-box.json').read_text(encoding='utf-8'))
    conds = json.loads((args.clean / 'report.json').read_text(encoding='utf-8'))['conditions']
    sub = {r['key'] for r in perbox}
    so = {k: v for k, v in labels.items() if k in sub}
    sn = {k: v for k, v in new_labels.items() if k in sub}
    for c in conds:
        res = {x['key']: x['decisions'][c] for x in perbox}
        for split in ['holdout', 'dev']:
            o = metrics.score(res, so, split)
            n = metrics.score(res, sn, split)
            show(f'干净图 {c} [{split}] 旧', o, classes=['车', '人员'])
            print(f'    新旧是否一致: {o == n}')

    print()
    print('=' * 90)
    print('dev 迭代轮次（更正真正落在此处）')
    print('=' * 90)
    for v in VERSIONS:
        o = metrics.score(dev_res[v], labels, 'dev')['overall']
        n = metrics.score(dev_res[v], new_labels, 'dev')['overall']
        print(f"  {v}: n={o['n']} 旧 {pct(o['accuracy'])} (right={o['right']}) -> "
              f"新 {pct(n['accuracy'])} (right={n['right']})  位移 {n['right'] - o['right']:+d}"
              f"  false_rel {o['false_released']} -> {n['false_released']}")

    print()
    print('=' * 90)
    print('道路裂纹（无标签池 197 框）')
    print('=' * 90)
    rc = json.loads(args.road_crack_labels.read_text(encoding='utf-8'))
    gt = sum(1 for v in rc.values() if v['label'] == 'true')
    print(f'  框数 {len(rc)}，金标：是裂纹 {gt}、不是裂纹 {len(rc) - gt}')
    print(f"  {'版本':8s} {'判是':>5s} {'判不是':>6s} {'看不清':>6s} {'准确率':>8s} "
          f"{'判是精确率':>10s} {'真裂纹召回':>10s} {'非裂纹拒绝':>10s} "
          f"{'真目标被驳回':>12s} {'非目标被放过':>12s}")
    for v in VERSIONS:
        res = dev_res[v]
        tt = tf = ft = ff = ut = uf = 0
        for k, lab in rc.items():
            raw = res.get(k)
            g = lab['label'] == 'true'
            if raw == 'true':
                tt, tf = (tt + 1, tf) if g else (tt, tf + 1)
            elif raw == 'false':
                ft, ff = (ft + 1, ff) if g else (ft, ff + 1)
            else:
                ut, uf = (ut + 1, uf) if g else (ut, uf + 1)
        n = tt + tf + ft + ff + ut + uf
        o = metrics.score(res, rc, 'unlabeled')['overall']
        prec = tt / (tt + tf) if tt + tf else None
        rec = tt / (tt + ft + ut) if tt + ft + ut else None
        ncr = (ff + uf) / (tf + ff + uf) if tf + ff + uf else None
        print(f"  {v:8s} {tt + tf:5d} {ft + ff:6d} {ut + uf:6d} {pct(o['accuracy']):>8s} "
              f"{pct(prec):>10s} {pct(rec):>10s} {pct(ncr):>10s} "
              f"{o['true_rejected']:12d} {o['false_released']:12d}")


if __name__ == '__main__':
    main()
