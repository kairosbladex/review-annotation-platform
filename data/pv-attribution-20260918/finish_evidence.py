from pathlib import Path
import json,hashlib,math
from PIL import Image,ImageDraw,ImageFont
from shapely.geometry import Polygon,box
P=Path(__file__).resolve().parent
R=lambda f:json.loads(f.read_text())
W=lambda f,x:f.write_text(json.dumps(x,ensure_ascii=False,indent=2))
font=ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc',22);small=ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc',18)
items=R(P/'review/batch-input.json');models={x['event_id']:x for x in R(P/'review/batch-results.json')['events']};cat=R(P/'project-catalog.json');comparison=R(P/'comparison.json');validation=[];rot=[];gallery=[]
for x in items:
 eid=x['event_id'];f=P/eid;im=Image.open(P/'review'/eid/'input.jpg').convert('RGB');w,h=im.size;ar=R(f/'absolute-projection.json');pts=ar['points'];d=ImageDraw.Draw(im);candidate=[];model=models[eid];polygons=[];scope=R(f/'attribution-input.json')['task_matrix_ids']
 for i,m in enumerate(cat['matrices']):
  q=pts[i*4:i*4+4]
  if len(q)!=4 or any(z is None or z.get('u') is None or z.get('v') is None for z in q):continue
  q=[(z['u'],z['v']) for z in q];g=Polygon(q)
  if not g.is_valid:continue
  polygons.append((m,g))
  if g.intersects(box(0,0,w,h)):
   color='#ffcc55' if m['id'] in scope else '#aa88ff';d.line(q+[q[0]],fill=color,width=3);anchor=min(q,key=lambda z:z[1]);d.text((max(0,min(w-180,anchor[0])),max(0,min(h-25,anchor[1]))),m['code']+' [AR]',font=small,fill=color)
 for det in x['detections']:
  a,b,c,e=det['bbox_xyxy'];g=box(a,b,c,e);d.rectangle([a,b,c,e],outline='#ff3355',width=3);d.text((c+4,b-15),f"T{det['index']}",font=font,fill='#ff3355');candidate.append(dict(index=det['index'],unverified_ar_intersections=[m['id'] for m,p in polygons if p.intersects(g)],nearest_boundary_px=min((p.distance(g) for _,p in polygons),default=None),not_final_attribution=True))
  rb=[w-c,h-e,w-a,h-b];back=[w-rb[2],h-rb[3],w-rb[0],h-rb[1]];rot.append(dict(event_id=eid,index=det['index'],box=det['bbox_xyxy'],rotated_180_box=rb,max_roundtrip_error=max(abs(v-z) for v,z in zip(back,det['bbox_xyxy'])),identity_verification='NOT_EVALUABLE_NO_VERIFIED_COMPONENT'))
 # Model outputs validated separately from AR geometry. No polygons are guessed for nulls.
 checks=[]
 for t in model.get('parsed',{}).get('targets',[]):
  q=t.get('component_polygon');reason='NO_COMPONENT_BOUNDARY' if q is None else 'SCHEMA_INVALID';geo=False
  if isinstance(q,list) and len(q)==4 and all(isinstance(v,list) and len(v)==2 for v in q):
   g=Polygon(q);orig=next((z for z in x['detections'] if z['index']==t.get('index')),None)
   if orig:
    detection=box(*orig['bbox_xyxy']);copied=g.symmetric_difference(detection).area<=max(1,detection.area*.02)
    geo=g.is_valid and box(0,0,w,h).covers(g) and g.covers(detection) and not copied;reason='GEOMETRY_PROPOSAL_ONLY' if geo else ('COPIED_DETECTION' if copied else 'INVALID_OR_UNSUPPORTED_POLYGON')
  elif isinstance(q,list) and len(q)==4 and all(type(v) in (float,int) for v in q):
   orig=next((z for z in x['detections'] if z['index']==t.get('index')),None)
   if orig and max(abs(a-b) for a,b in zip(q,orig['bbox_xyxy']))<.01:reason='SCHEMA_INVALID_AND_COPIED_DETECTION'
  checks.append(dict(index=t.get('index'),geometry_valid=geo,reason=reason,visual_result_usable=model['schema_valid'] and model['generation_complete']))
 W(f/'ar-candidate-audit.json',candidate);W(f/'visual-validation.json',checks)
 canvas=Image.new('RGB',(w,h+122),'#111d29');canvas.paste(im,(0,82));dr=ImageDraw.Draw(canvas);dr.text((15,8),f"第{x['number']}例 | {'无标注可见光原图' if x['number']<=5 else '热成像原图'} | {len(x['detections'])}个目标 | 最终归属：未知",font=font,fill='white');dr.text((15,40),'红：冻结目标  黄：任务所选方阵 AR 先验  紫：其他维护方阵 AR 先验',font=small,fill='#ffcc55');dr.text((15,h+88),'投影未独立验收；不确定半径未知，不绘制虚假的误差范围或组件边界。',font=small,fill='white');canvas.save(f/'overlay.jpg',quality=94)
 validation.append(dict(number=x['number'],event_id=eid,transport=model['transport_complete'],generation=model['generation_complete'],schema=model['schema_valid'],targets=checks,mapping=False,business=False))
 for det in x['detections']:
  pass
 # Contact sheet, two full inputs with readable stage labels per image.
 gallery.append(canvas)
for i in range(0,10,2):
 sheet=Image.new('RGB',(1440,1250),'#111d29');dr=ImageDraw.Draw(sheet)
 for j,c in enumerate(gallery[i:i+2]):
  c.thumbnail((1440,620));sheet.paste(c,((1440-c.width)//2,j*625))
 sheet.save(P/f'gallery-{i//2+1}.jpg',quality=93)
W(P/'visual-validation.json',validation);W(P/'rotation-coordinate-check.json',rot)
W(P/'acceptance.json',dict(events=10,targets=23,transport=sum(x['transport'] for x in validation),generation=sum(x['generation'] for x in validation),schema=sum(x['schema'] for x in validation),valid_component_geometry=sum(t['geometry_valid'] for x in validation for t in x['targets']),mapping_verified=0,business_accepted=0,rotation_coordinate_checks=len(rot),rotation_component_identity='NOT_EVALUABLE',accuracy=None,no_business_writes=True))
print('galleries=5',R(P/'acceptance.json'))
