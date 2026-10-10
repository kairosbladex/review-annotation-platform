"""Compact progress/resource view, without inspecting model predictions."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];run=root/'runs/holdout'
r=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {'done':0,'total':348}
samples=[json.loads(l) for l in (run/'resources.jsonl').read_text().splitlines()] if (run/'resources.jsonl').exists() else []
records=[json.loads(p.read_text()) for p in (run/'out').glob('*.json')]
r.update(valid=sum(x['success'] for x in records),failed=sum(not x['success'] for x in records),retry_attempts=sum(len(x['attempts'])-1 for x in records),complete=(run/'complete.json').exists(),scored=(root/'holdout-report.json').exists())
if samples:r.update(memory_mib=round(samples[-1]['memory_current']/2**20,1),peak_memory_mib=round(max(s['memory_current'] for s in samples)/2**20,1),host_available_pct=round(samples[-1]['available_fraction']*100,1),swap_bytes=samples[-1]['swap_current'],host_swapout_delta=samples[-1]['pswpout']-samples[0]['pswpout'])
print(json.dumps(r,ensure_ascii=False,indent=2))
