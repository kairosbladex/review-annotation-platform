"""Isolated evidence-based altitude ablation; catalog contains measured nonzero vertex elevations."""
from pathlib import Path
import json,subprocess,base64
p=Path(__file__).resolve().parent
for item in json.loads((p/'review/batch-input.json').read_text()):
 f=p/item['event_id'];r=json.loads((f/'ar-replay-request.json').read_text());form=r['input']['form'];pars=json.loads(form['overlay_params_json']);pars['kml_alt_mode']='absolute';form['overlay_params_json']=json.dumps(pars)
 (f/'absolute-request.json').write_text(json.dumps(r));res=subprocess.run(['/Users/wangxiaolin/PycharmProjects/review-annotation-platform/.venv/bin/python','-B','-m','spatial.replay'],cwd='/Users/wangxiaolin/PycharmProjects/drone-diagnostics-integration',input=json.dumps(r),text=True,capture_output=True,timeout=120);(f/'absolute-stderr.txt').write_text(res.stderr);assert res.returncode==0,res.stderr
 x=json.loads(res.stdout)['output']
 for k in ['image_png_base64','image_jpg_base64']:
  if x.get(k):(f/'absolute-prior.jpg').write_bytes(base64.b64decode(x.pop(k)))
 x.update(mapping_verified=False,role='catalog_altitude_prior_not_independently_calibrated');(f/'absolute-projection.json').write_text(json.dumps(x,ensure_ascii=False,indent=2));print(item['number'],x['overlay_status'],flush=True)
