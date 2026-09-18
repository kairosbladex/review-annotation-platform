import json,hashlib,requests,datetime
from pathlib import Path
from urllib.parse import urlparse
p=Path(__file__).resolve().parent;rows=json.loads((p/'ar-source-records.json').read_text());sel=json.loads((p.parent/'pv-batch10-20260918/selection.json').read_text());receipt=[]
for r in sel[:5]:
 folder=p/r['id'];folder.mkdir(exist_ok=True);found={}
 for rec in rows:
  if rec.get('event')!='overlay_image' or r['taskId'] not in json.dumps(rec):continue
  im=rec['image'];t=im.get('shot_time_bj','').replace('T',' ')[:19]
  if t==r['eventTime']:found[im['source_detail']]=rec
 for url,rec in found.items():
  assert urlparse(url).netloc=='lxtest.zhuoyilingxi.com:8088'
  ch='W' if '_W_' in url else 'T';res=requests.get(url,timeout=60);res.raise_for_status();b=res.content;sha=hashlib.sha256(b).hexdigest();assert sha.startswith(rec['image']['sha256_16']);(folder/f'raw-{ch}.jpg').write_bytes(b);(folder/f'ar-{ch}.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2));receipt.append(dict(event_id=r['id'],channel=ch,sha256=sha,url=url,event_time=r['eventTime'],time_match=True));print(r['id'][:8],ch,len(b),flush=True)
  for kind in ['kml','telemetry']:
   u=rec[kind]['source_detail'];target=p/(r['taskId']+('.kml' if kind=='kml' else '.jsonl'))
   if not target.exists():
    assert urlparse(u).netloc=='lxtest.zhuoyilingxi.com:8088';q=requests.get(u,timeout=60);q.raise_for_status();target.write_bytes(q.content)
(p/'raw-recovery.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
