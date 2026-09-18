from pathlib import Path
import json,hashlib,shutil,math
import numpy as np
from PIL import Image
p=Path('/Users/wangxiaolin/PycharmProjects/review-annotation-platform/data/pv-attribution-20260918');old=p.parent/'pv-batch10-20260918';out=p/'review';out.mkdir(exist_ok=True)
cor={x['event_id']:x for x in json.loads((p/'raw-image-correspondence.json').read_text())};items=[]
for x in json.loads((old/'attempt3/batch-input.json').read_text()):
 eid=x['event_id'];f=out/eid;f.mkdir(exist_ok=True)
 im=Image.open(p/eid/'clean-review.jpg') if eid in cor else Image.open(old/eid/'input.jpg')
 if eid in cor:
  h=np.linalg.inv(np.array(cor[eid]['raw_resized_to_event']));(f/'coordinate-transform.json').write_text(json.dumps({'event_to_clean_centers':h.tolist(),'clean_to_event_centers':np.linalg.inv(h).tolist(),'source_correspondence':'../../raw-image-correspondence.json','box_convention':'pixel edges; minus 0.5 before center transform, plus 0.5 after','not_catalog_mapping':True}))
  for d in x['detections']:
   a,b,c,e=d['bbox_xyxy'];q=np.array([[a,b],[c,b],[c,e],[a,e]])-.5;q=np.c_[q,np.ones(4)]@h.T;q=q[:,:2]/q[:,2:]+.5
   d['bbox_xyxy']=[float(q[:,0].min()),float(q[:,1].min()),float(q[:,0].max()),float(q[:,1].max())]
 im.save(f/'input.jpg',quality=95);x['image_sha256']=hashlib.sha256((f/'input.jpg').read_bytes()).hexdigest();x['image_source']='recovered_same_exposure_unannotated_W' if eid in cor else 'frozen_original_thermal';x['geometry']={'status':'UNDETERMINED','reason':'AR_PRIOR_WITHOUT_INDEPENDENT_MAPPING_CHECK','mapping_verified':False};x['crops']=[]
 for d in x['detections']:
  a,b,c,e=d['bbox_xyxy'];cx=(a+c)/2;cy=(b+e)/2;w,h=im.size;l=max(0,min(w-256,round(cx-128)));t=max(0,min(h-256,round(cy-128)));fn=f"crop-{d['index']}.png";im.crop((l,t,l+256,t+256)).save(f/fn);x['crops'].append(dict(index=d['index'],file=fn,source_bounds=[l,t,l+256,t+256],sha256=hashlib.sha256((f/fn).read_bytes()).hexdigest()))
 # Remove all legacy asset selection and labels from actual vision context.
 for k in ['matrix_id','matrix_code','catalog_count','catalog_rows','catalog_columns']:x.pop(k,None)
 items.append(x)
(out/'batch-input.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
shutil.copy(old/'attempt3/mcp_batch.py',out/'mcp_batch.py');shutil.copy(old/'attempt3/run_hermes.py',out/'run_hermes.py')
s=(out/'mcp_batch.py').read_text().replace("'catalog_count','catalog_rows','catalog_columns',",'')
s=s.replace("summary['schema_valid']=isinstance(parsed,dict)","summary['schema_valid']=isinstance(parsed,dict)")
s=s.replace("t['anomaly_visible'] in ('yes','no','uncertain') for t", "t['anomaly_visible'] in ('yes','no','uncertain') and isinstance(t['evidence'],str) and isinstance(t['uncertainty'],str) and (t['component_polygon'] is None or (isinstance(t['component_polygon'],list) and len(t['component_polygon'])==4 and all(isinstance(q,list) and len(q)==2 and all(type(v) in (int,float) and __import__('math').isfinite(v) for v in q) for q in t['component_polygon']))) for t")
(out/'mcp_batch.py').write_text(s)
s=(out/'run_hermes.py').read_text().replace('1至5是带旧叠加的可见光预警照片','1至5是已追溯并通过像素配准核对的无标注可见光原始照片').replace('几何候选不是业务确认','目录AR初始投影未经独立检查，不能推出范围外；几何候选不是业务确认')
(out/'run_hermes.py').write_text(s)
shutil.copy('/private/tmp/prepare_pv_new.py',p/'prepare_review.py')
print(len(items),sum(len(x['detections']) for x in items))
