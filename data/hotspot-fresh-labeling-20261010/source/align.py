import json,pathlib,cv2,numpy as np,hashlib,os
cv2.setNumThreads(1)
p=pathlib.Path(os.environ.get('HOTSPOT_SEARCH_DIR',str(pathlib.Path(__file__).resolve().parents[1])));jobs=json.loads((p/'download-jobs.json').read_text());downloads={r['event_id']:r for r in json.loads((p/'downloads.json').read_text())};out=[]
for j in jobs:
 eid=j['event']['id'];d=downloads[eid]
 if not all(v['ok'] for v in d['files'].values()):out.append({'event_id':eid,'qualified':False,'reason':'source_not_available'});continue
 a=cv2.imread(str(p/d['files']['shot']['path']),0);b=cv2.imread(str(p/d['files']['raw']['path']),0)
 detector=cv2.SIFT_create(nfeatures=2500);ka,da=detector.detectAndCompute(a,None);kb,db=detector.detectAndCompute(b,None)
 pairs=cv2.BFMatcher().knnMatch(da,db,k=2);good=[m for m,n in pairs if m.distance<.7*n.distance];rng=np.random.default_rng(20261010);rng.shuffle(good)
 if len(good)<40:out.append({'event_id':eid,'qualified':False,'reason':'too_few_matches','matches':len(good)});continue
 cut=int(len(good)*.7);train=good[:cut];check=good[cut:];src=np.float32([ka[m.queryIdx].pt for m in train]);dst=np.float32([kb[m.trainIdx].pt for m in train]);H,mask=cv2.findHomography(src,dst,cv2.RANSAC,3)
 if H is None:out.append({'event_id':eid,'qualified':False,'reason':'no_transform'});continue
 cs=np.float32([ka[m.queryIdx].pt for m in check]).reshape(-1,1,2);ct=np.float32([kb[m.trainIdx].pt for m in check]);pr=cv2.perspectiveTransform(cs,H).reshape(-1,2);err=np.linalg.norm(pr-ct,axis=1);median=float(np.median(err));p90=float(np.quantile(err,.9));inliers=int(mask.sum());ratio=float(mask.mean());goodfit=inliers>=30 and ratio>=.5 and median<=2.5 and p90<=5
 points=json.loads(j['shot']['imgxy']);mapped=[]
 for i,v in enumerate(points):
  if v.get('class_name')!='光斑' or not all(k in v for k in ('x','y')):continue
  raw=cv2.perspectiveTransform(np.float32([[[v['x'],v['y']]]]),H)[0,0];inside=0<=raw[0]<b.shape[1] and 0<=raw[1]<b.shape[0]
  mapped.append({'index':i,'screenshot_point':[v['x'],v['y']],'raw_point':[float(x) for x in raw],'inside':bool(inside)})
 out.append({'event_id':eid,'qualified':bool(goodfit),'reason':'pass' if goodfit else 'independent_alignment_check_failed','H':H.tolist(),'screenshot_size':[a.shape[1],a.shape[0]],'raw_size':[b.shape[1],b.shape[0]],'matches':len(good),'fit_inliers':inliers,'fit_inlier_ratio':ratio,'check_points':len(check),'check_median_px':median,'check_p90_px':p90,'points':mapped})
(p/'alignment.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'events':len(out),'qualified':sum(r['qualified'] for r in out),'points_in_qualified':sum(sum(pt['inside'] for pt in r.get('points',[])) for r in out if r['qualified']),'failures':[{k:r[k] for k in ['event_id','reason']} for r in out if not r['qualified']]}))
