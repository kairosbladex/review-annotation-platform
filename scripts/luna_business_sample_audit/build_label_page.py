"""Build a local-only blind annotation page; outputs/images remain git-ignored."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
from PIL import Image

def run(out,fetch_recent):
 p=out/'private';rows=json.loads((p/'diagnostic.json').read_text());recent=json.loads((p/'recent-reserved.json').read_text()); failures=[]
 for r in recent:
  src=r.get('source_files',{}).get('screenshot');dest='media/'+hashlib.sha256(r['event_id'].encode()).hexdigest()[:20]+'-recent.jpg'; f=p/dest
  if src and fetch_recent and not f.exists():
   try:
    url='https://lxtest.zhuoyilingxi.com:8088/minio-src'+src
    with urllib.request.urlopen(url,timeout=15) as res:
     b=res.read(8*1024*1024+1)
     if len(b)>8*1024*1024:raise ValueError('image_too_large')
    f.write_bytes(b)
    with Image.open(f) as im:im.verify()
   except Exception as e:failures.append({'sample_id':r['sample_id'],'error':type(e).__name__});continue
  if not f.exists():continue
  b=r['box'];fw,fh=b['frame_width'],b['frame_height']
  rows.append(dict(r,files={'screenshot':dest},bbox=[b['x']/fw,b['y']/fh,(b['x']+b['width'])/fw,(b['y']+b['height'])/fh],clean_note='没有独立原帧；原图附件与截图相同或未核验。不能用于新盲测。'))
 safe=[];media=[]
 for r in rows:
  rr={k:r[k] for k in ['sample_id','queue','class_name','files','bbox','clean_note']}
  for k in ['target_bbox_px','annotated_source_box','clean_source_box','source_frame_size']:
   if k in r:rr[k]=r[k]
  for f in r['files'].values():
   b=(p/f).read_bytes()
   with Image.open(p/f) as im:im.verify()
   media.append({'path':f,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
  safe.append(rr)
 template=Path(__file__).with_name('label_page.html').read_text();payload=json.dumps(safe,ensure_ascii=False).replace('</','<\\/')
 (p/'index.html').write_text(template.replace('/*DATA*/[]',payload));(p/'media-manifest.json').write_text(json.dumps(media,indent=2))
 summary={'rows':len(safe),'media_unique':len({x['path'] for x in media}),'missing_recent':failures,'labels_prefilled':0,'page_has_model_predictions':False}
 (out/'label-package-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(summary)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--fetch-recent',action='store_true');x=a.parse_args();run(x.out,x.fetch_recent)
