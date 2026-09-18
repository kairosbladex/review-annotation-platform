from pathlib import Path
import json,math,hashlib
import numpy as np
import cv2,jsonschema
from PIL import Image,ImageDraw,ImageFont
p=Path(__file__).resolve().parent
# Schema is read from the exact submitted request, not from model text.
def polygon_check(poly,box,size):
 if poly is None:return {'status':'unknown','reason':'NO_COMPONENT_BOUNDARY'}
 a=np.asarray(poly,float);w,h=size
 if a.shape!=(4,2) or not np.isfinite(a).all():return {'status':'rejected','reason':'INVALID_POINTS'}
 q=a.astype('float32');area=abs(cv2.contourArea(q))
 if area<1 or not cv2.isContourConvex(q):return {'status':'rejected','reason':'NONCONVEX_OR_EMPTY'}
 if (a[:,0]<0).any() or (a[:,0]>w).any() or (a[:,1]<0).any() or (a[:,1]>h).any():return {'status':'rejected','reason':'OUTSIDE_IMAGE'}
 x1,y1,x2,y2=box;b=np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]);ba=(x2-x1)*(y2-y1);inter,_=cv2.intersectConvexConvex(q,b);iou=inter/(area+ba-inter)
 if iou>.8 or area<=ba*1.2:return {'status':'rejected','reason':'COPIES_DETECTION'}
 if inter/ba<.999:return {'status':'rejected','reason':'DOES_NOT_CONTAIN_TARGET'}
 return {'status':'coordinate_valid_only','reason':'VISUAL_BOUNDARY_NOT_INDEPENDENTLY_VERIFIED','area_ratio':area/ba}
font='/System/Library/Fonts/STHeiti Light.ttc';F=lambda n:ImageFont.truetype(font,n)
items=json.loads((p/'batch-input.json').read_text());results=[]
for item in items:
 folder=p/item['event_id'];s=json.loads((folder/'vision-summary.json').read_text());req=json.loads((folder/'vision-request.json').read_text());valid=False
 try:
  jsonschema.validate(s.get('parsed'),req['format']);ts=s['parsed']['targets'];assert sorted(t['index'] for t in ts)==[d['index'] for d in item['detections']];valid=True
 except (jsonschema.ValidationError,AssertionError):pass
 checks=[]
 for det in item['detections']:
  target=next((t for t in s.get('parsed',{}).get('targets',[]) if t.get('index')==det['index']),{}) if valid else {}
  checks.append(dict(index=det['index'],anomaly_visible=target.get('anomaly_visible','uncertain'),boundary=polygon_check(target.get('component_polygon'),det['bbox_xyxy'],item['image_size']),evidence=target.get('evidence'),uncertainty=target.get('uncertainty')))
 out=dict(number=item['number'],event_id=item['event_id'],matrix_code=item['matrix_code'],image_source=item['image_source'],target_count=len(checks),transport_complete=s.get('transport_complete',False),generation_complete=s.get('generation_complete',False),schema_valid=valid,targets=checks,geometry=item['geometry'],mapping_verified=False,business_accepted=False,seconds=s['seconds'])
 out['visual_acceptance']='MODEL_OBSERVATION_ONLY' if item['image_source']=='frozen_original' and valid else 'UNKNOWN'
 out['rejection_reasons']=[]
 if not valid:out['rejection_reasons'].append('VISION_SCHEMA_INVALID')
 if item['image_source']!='frozen_original':out['rejection_reasons'].append('PREEXISTING_ANNOTATIONS_NOT_INDEPENDENT_EVIDENCE')
 for t in s.get('parsed',{}).get('targets',[]):
  det=next((d for d in item['detections'] if d['index']==t.get('index')),None)
  if det and t.get('component_polygon')==det['bbox_xyxy']:out['rejection_reasons'].append('VISION_BOUNDARY_COPIES_DETECTION')
 (folder/'validated-result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));results.append(out)
 im=Image.open(folder/'input.jpg').convert('RGB');d=ImageDraw.Draw(im)
 for det in item['detections']:
  b=det['bbox_xyxy'];d.rectangle(b,outline='#ff355d',width=3);d.text((max(0,b[0]-5),max(0,b[1]-27)),str(det['index']),font=F(23),fill='#ffff00',stroke_width=2,stroke_fill='black')
 for g in item['geometry'].get('detections',[]):
  if g.get('candidate'):
   region=next(x for x in item['geometry'].get('projected_regions',[]) if x['id']==g['candidate']['id']);q=[tuple(z) for z in region['polygon']];d.line(q+[q[0]],fill='#00ffff',width=3)
 im.thumbnail((720,550));height=im.height+150;canvas=Image.new('RGB',(720,height),'#101b29');canvas.paste(im,((720-im.width)//2,58));d=ImageDraw.Draw(canvas)
 d.text((16,12),f"{item['number']:02d}  {item['event_id'][:8]}  |  {item['matrix_code']}  |  {len(checks)} 个目标",font=F(23),fill='white')
 candidates=[f"目标{g['index']}: H{g['candidate']['row']}/L{g['candidate']['column']} 候选" for g in item['geometry'].get('detections',[]) if g.get('candidate')]
 text='；'.join(candidates) if candidates else '组件编号：unknown（无可信映射）'
 d.text((16,im.height+70),text,font=F(21),fill='#ffcc77');d.text((16,im.height+103),('可见光带旧叠加：模型观察不可独立采信' if item['image_source']!='frozen_original' else '红框：原检测目标；青框：辅助组件候选；未业务验收'),font=F(19),fill='#ddddee')
 canvas.save(folder/'review.png')
(p/'validated-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps([dict(number=x['number'],schema=x['schema_valid'],visible=[t['anomaly_visible'] for t in x['targets']],reason=x['geometry'].get('reason'),seconds=x['seconds']) for x in results],ensure_ascii=False,indent=2))
