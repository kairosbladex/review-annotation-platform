from query import *
import requests,hashlib,concurrent.futures,io
from urllib.parse import urlsplit,quote
from PIL import Image
p=WORK;jobs=json.loads((p/'download-jobs.json').read_text());dest=p/'images';dest.mkdir(exist_ok=True)
def url(path):
 if path.startswith('https://'):return path
 return 'https://lxtest.zhuoyilingxi.com:8088'+('/local-src' if path.startswith('/upload/') else '/minio-src')+quote(path,safe='/%')
def one(job):
 out={'event_id':job['event']['id'],'files':{}}
 for kind in ['shot','raw']:
  a=job[kind];target=dest/(a['id']+'.jpg')
  try:
   with requests.get(url(a['file_path']),timeout=(8,25),stream=True) as r:
    r.raise_for_status();data=bytearray()
    for block in r.iter_content(65536):
     data.extend(block)
     if len(data)>20*1024*1024:raise ValueError('image exceeds 20MiB limit')
   im=Image.open(io.BytesIO(data));size=im.size;im.verify();target.write_bytes(data)
   out['files'][kind]={'ok':True,'path':str(target.relative_to(p)),'bytes':len(data),'size':size,'sha256':hashlib.sha256(data).hexdigest()}
  except Exception as e:out['files'][kind]={'ok':False,'error_type':type(e).__name__,'http_status':getattr(getattr(e,'response',None),'status_code',None)}
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,jobs))
(p/'downloads.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps({'events':len(results),'both_downloaded':sum(all(v['ok'] for v in r['files'].values()) for r in results),'failures':[{ 'event':r['event_id'],'kind':k,**v} for r in results for k,v in r['files'].items() if not v['ok']]},ensure_ascii=False))
