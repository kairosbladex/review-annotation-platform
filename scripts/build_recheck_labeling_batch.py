#!/usr/bin/env python3
"""Select unlabeled high-resolution boxes locally; sources remain read-only."""
import argparse
import collections
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from consolidate_recheck_gold import read_rows, write_rows, iou

def build(root, gold_path, released_path, out):
    gold = read_rows(gold_path)
    by_hash = collections.defaultdict(list)
    for g in gold: by_hash[g['image_sha256']].append(g['bbox'])
    vsg=root/'data/vehicle-surface-gate-eval-20260929'
    pool=read_rows(vsg/'holdout/pool.jsonl')
    released={}
    for e in json.loads(released_path.read_text()):
        try: items=json.loads(e['recheck_result']).get('resultList',[])
        except (ValueError,TypeError):continue
        released[e['id']]=[x.get('originalBbox') for x in items if x.get('decision')=='confirmed' and x.get('originalBbox')]
    sources={}
    for jobs in sorted(root.glob('data/**/jobs.jsonl')):
        if not (jobs.parent/'results.jsonl').exists():continue
        for j in read_rows(jobs):
            frame=jobs.parent/'frames'/f"{j['event_id']}.jpg"
            if not frame.exists() or not j.get('homography'):continue
            with Image.open(frame) as im: size=im.size
            if min(size)<1000:continue
            sources.setdefault(j['event_id'],(frame,size,j['homography'],str(jobs)))
    rooftop_events={a.split(':',1)[1] for g in gold if 'rooftop_hard_case' in g['population'] for a in g['group_aliases'] if a.startswith('event:')}
    available=collections.defaultdict(list);excluded=collections.Counter();seen=collections.defaultdict(list)
    rng=random.Random(20260930);rng.shuffle(pool)
    for r in pool:
        if r['class_name']!='车':continue
        eid=r['event_id']
        if eid not in sources:excluded['no_verified_hires']+=1;continue
        frame,size,h,source=sources[eid];digest=hashlib.sha256(frame.read_bytes()).hexdigest()
        x1,y1,x2,y2=r['bbox_px'];points=np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]])@np.asarray(h).T
        points=points[:,:2]/points[:,2:];box=[float(points[:,0].min()/size[0]),float(points[:,1].min()/size[1]),float(points[:,0].max()/size[0]),float(points[:,1].max()/size[1])]
        if not all(0<=x<=1 for x in box):excluded['outside_original']+=1;continue
        if any(iou(box,g)>=0.5 for g in by_hash[digest]+seen[digest]):excluded['duplicate_gold_or_batch']+=1;continue
        is_released=any(iou(r['bbox'],b)>=0.5 for b in released.get(eid,[]))
        pop='released_by_recheck' if is_released else 'rooftop_hard_case' if eid in rooftop_events else 'random_traffic'
        seen[digest].append(box)
        available[pop].append({'candidate_id':r['candidate_id'],'event_id':eid,'class_name':'车','population':pop,'bbox':box,'image_sha256':digest,'source_image':str(frame),'source_size':list(size),'geometry_source':source})
    # Released package also has a few boxes not in the screened holdout pool.
    package=root/'data/recheck-released-fp-20260930/labeling-package'
    labels={r['candidate_id']:r['label'] for r in csv.DictReader((package/'annotations.csv').open(encoding='utf-8-sig'))}
    for r in read_rows(package/'results.jsonl'):
        if labels.get(r['candidate_id']):continue
        frame=package/r['frame'];size=r['source_size']
        if not frame.exists() or min(size)<1000:continue
        digest=hashlib.sha256(frame.read_bytes()).hexdigest();points=r['projected_polygon']
        box=[min(p[0] for p in points)/size[0],min(p[1] for p in points)/size[1],max(p[0] for p in points)/size[0],max(p[1] for p in points)/size[1]]
        if any(iou(box,b)>=0.5 for b in by_hash[digest]+seen[digest]):continue
        seen[digest].append(box)
        available['released_by_recheck'].append({'candidate_id':r['candidate_id'],'event_id':r['event_id'],'class_name':'车','population':'released_by_recheck','bbox':box,'image_sha256':digest,'source_image':str(frame),'source_size':size,'geometry_source':str(package/'results.jsonl')})
    selected=[];counts={}
    for pop,count in [('rooftop_hard_case',80),('released_by_recheck',80),('random_traffic',40)]:
        selected+=available[pop][:count];counts[pop]={'available':len(available[pop]),'selected':min(count,len(available[pop])),'requested':count}
    out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True)
    annotations=[]
    for r in selected:
        slug=hashlib.sha256(r['candidate_id'].encode()).hexdigest()[:20];r['slug']=slug
        # Symlink the original; no second copy of the dataset or originals.
        original=out/'images'/f'{slug}.jpg'
        if not original.exists():original.symlink_to(r['source_image'])
        with Image.open(r['source_image']) as im:
            im=im.convert('RGB');w,h=im.size;box=[r['bbox'][i]*[w,h,w,h][i] for i in range(4)]
            crop=im.crop((max(0,box[0]-80),max(0,box[1]-80),min(w,box[2]+80),min(h,box[3]+80)))
            crop.save(out/'images'/f'{slug}-crop.jpg',quality=95)
            im.thumbnail((1600,1600));scale=im.width/w;ImageDraw.Draw(im).rectangle([x*scale for x in box],outline='yellow',width=2);im.save(out/'images'/f'{slug}-context.jpg',quality=95)
        annotations.append({'candidate_id':r['candidate_id'],'event_id':r['event_id'],'class_name':'车','label':'','note':'','label_origin':'','labeled_by':'','labeled_at':'','slug':slug})
    with (out/'annotations.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=list(annotations[0]) if annotations else ['candidate_id','label','note']);wr.writeheader();wr.writerows(annotations)
    write_rows(out/'selected.jsonl',selected)
    rates={p:sum(g['label']=='false' for g in gold if p in g['population'] and g['class_name']=='车')/max(1,sum(g['label'] in ['true','false'] for g in gold if p in g['population'] and g['class_name']=='车')) for p in counts}
    expected=sum(v['selected']*rates[p] for p,v in counts.items())
    report={'counts':counts,'excluded':dict(excluded),'historical_false_rates':rates,'expected_new_false':round(expected,1),'expected_locked_false_upper_estimate':round(expected*0.7,1),'selection':'seeded pool selection; population determined from source events; not model suggestion','database_writes':0}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gold',type=Path,required=True);p.add_argument('--released',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.root,a.gold,a.released,a.out)
