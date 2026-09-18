import json,hashlib,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image
p=Path(__file__).resolve().parent;old=p.parent/'pv-batch10-20260918';sel=json.loads((old/'selection.json').read_text());results=[]
for r in sel[:5]:
 folder=p/r['id'];raw=Image.open(folder/'raw-W.jpg').convert('RGB');event=Image.open(old/r['id']/'original.jpg').convert('RGB');a=np.array(raw.resize(event.size));b=np.array(event);sift=cv2.SIFT_create(nfeatures=3000,enable_precise_upscale=True);ka,da=sift.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),None);kb,db=sift.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),None);pairs=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.65*n.distance];pts=np.float32([ka[m.queryIdx].pt for m in pairs]);dst=np.float32([kb[m.trainIdx].pt for m in pairs]);fit=np.arange(len(pts))%5!=0;H,mask=cv2.findHomography(pts[fit],dst[fit],cv2.RANSAC,2);check=~fit;err=np.linalg.norm(cv2.perspectiveTransform(pts[check,None,:],H)[:,0]-dst[check],axis=1);res=dict(event_id=r['id'],source_size=list(raw.size),event_size=list(event.size),matches=len(pairs),fit_inliers=int(mask.sum()),heldout=len(err),heldout_p95=float(np.percentile(err,95)),raw_resized_to_event=H.tolist(),event_to_raw=(np.diag([raw.width/event.width,raw.height/event.height,1])@np.linalg.inv(H)).tolist(),full_raw_sha256=hashlib.sha256((folder/'raw-W.jpg').read_bytes()).hexdigest(),pixel_convention='OpenCV pixel centers; convert +/- 0.5 for pixel-edge boxes')
 Image.fromarray(a).save(folder/'clean-review.jpg');results.append(res);print(r['id'][:8],len(pairs),res['heldout_p95'])
(p/'raw-image-correspondence.json').write_text(json.dumps(results,indent=2))
