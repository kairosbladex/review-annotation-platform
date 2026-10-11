"""Recompute the frozen 24-positive/17-negative diagnostic cohort, including coverage."""
import argparse
import collections
import json
from pathlib import Path


def score(units, records):
    counts=collections.Counter()
    for unit in units:
        data=records.get(unit['unit'])
        if not data or not data.get('success'):
            counts['missing']+=len(unit['candidates'])
            continue
        result=data.get('result',data.get('raw_body'))
        items=result['results']
        assert len(items)==len(unit['candidates'])
        assert all(type(x['target_index']) is int for x in items)
        assert {x['target_index'] for x in items}==set(range(len(items)))
        by_index={x['target_index']:x for x in items}
        for i,candidate in enumerate(unit['candidates']):
            decision=by_index[i]['decision']
            assert decision in ('true','false','uncertain')
            gold=candidate['gold']
            counts['covered']+=1
            counts[f'{gold}:{decision}']+=1
    for key in ('covered','missing','true:true','true:false','true:uncertain','false:true','false:false','false:uncertain'):
        counts[key]+=0
    return dict(counts,tp=counts['true:true'],fp=counts['false:true'],
                diagnosis_gate=counts['covered']==41 and counts['true:true']>=7 and counts['false:true']==0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    units=json.loads((args.root/'experiment-units.json').read_text())
    groups={}
    for r in (1,2):
        records={u['unit']:json.loads((args.root/'raw'/f'{u["unit"]}-r{r}.json').read_text()) for u in units}
        groups[f'A{r}']=score(units,records)
    entries=collections.defaultdict(dict)
    for path in (args.root/'ablation').glob('*.json'):
        data=json.loads(path.read_text());entries[data['entry']][data['unit']]=data
    for entry,records in sorted(entries.items()):
        groups[entry]=score(units,records)
    if args.out:args.out.write_text(json.dumps(groups,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(groups,ensure_ascii=False))


if __name__=='__main__':main()
