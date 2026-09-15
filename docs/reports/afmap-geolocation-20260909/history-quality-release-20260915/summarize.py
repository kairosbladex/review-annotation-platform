"""仅统计有对应参考点的误差，不将坐标移动或从无到有计作误差下降。"""
import json,hashlib,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[4]
repair=root/'data/history-frame-repair-20260915/prepared.json'
reference=root/'data/cross-event-accuracy-20260915/summary.json'
b=json.loads(repair.read_text());ref=json.loads(reference.read_text());byid={r['event_id']:r for r in ref['events']};rows=[];before=[];after=[]
for eid,v in b['events'].items():
 row=dict(event_id=eid,event_time=v['event_time'],targets=v['targets'],repair_reason=v['reason'],object_ground_truth_available=False)
 if 'limited_validated_frame_intrinsics' in v['reason']:
  r=byid[eid];old=r['variants']['historical'];new=r['variants']['native'];a=old['errors_m'];c=new['errors_m'];assert len(a)==len(c)==8
  row.update(evidence='same_frame_road_reference_only',reference_points=8,before_median_m=statistics.median(a),after_median_m=statistics.median(c),before_p95_m=old['p95_m'],after_p95_m=new['p95_m'],median_reduction_pct=(1-statistics.median(c)/statistics.median(a))*100,reference_points_improved=sum(y<x for x,y in zip(a,c)),reference_points_worsened=sum(y>x for x,y in zip(a,c)))
  before+=a;after+=c
 else:row['evidence']='restored_without_original_coordinates' if v['reason'].startswith('recover_') else 'corrected_boxes_without_ground_truth'
 rows.append(row)
result={'scope':'40 repaired events; ground-truth object error unavailable; road references are correlated observations within one flight, not independent control points','source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [repair,reference]},'total_events':len(rows),'targets':sum(r['targets'] for r in rows),'object_error_evaluable_events':0,'road_reference_evaluable_events':2,'road_reference_improved_events':2,'road_reference_worsened_events':0,'road_reference_at_least_79pct_events':sum(r.get('median_reduction_pct',0)>=79 for r in rows),'object_error_undetermined_events':40,'without_road_reference_events':38,'restored_events':29,'realbox_only_events':9,'pooled_reference_observations':len(before),'pooled_before_median_m':statistics.median(before),'pooled_after_median_m':statistics.median(after),'pooled_median_reduction_pct':100*(1-statistics.median(after)/statistics.median(before)),'reference_observations_improved':sum(y<x for x,y in zip(before,after)),'events':rows}
Path(__file__).with_name('accuracy-statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('events','source_sha256')},ensure_ascii=False))
