from pathlib import Path
import json,requests,base64,subprocess,hashlib,xml.etree.ElementTree as ET
p=Path(__file__).resolve().parent;old=p.parent/'pv-batch10-20260918';catalog=json.loads((p/'project-catalog.json').read_text());root=ET.Element('kml',xmlns='http://www.opengis.net/kml/2.2');doc=ET.SubElement(root,'Document')
for m in catalog['matrices']:
 g=json.loads(m['lngLatAlt']);g=g.get('geometry',g)
 if g.get('type')!='Polygon':continue
 mark=ET.SubElement(doc,'Placemark');ET.SubElement(mark,'name').text=m['id'];poly=ET.SubElement(mark,'Polygon');ET.SubElement(poly,'altitudeMode').text='absolute'
 for i,ring in enumerate(g['coordinates']):
  linear=ET.SubElement(ET.SubElement(poly,'outerBoundaryIs' if i==0 else 'innerBoundaryIs'),'LinearRing');ET.SubElement(linear,'coordinates').text=' '.join(','.join(map(str,v)) for v in ring)
kml=ET.tostring(root,encoding='utf-8');(p/'project-maintained.kml').write_bytes(kml)
for i,r in enumerate(json.loads((old/'selection.json').read_text()),1):
 folder=p/r['id'];channel='W' if i<=5 else 'T';record=json.loads((folder/f'ar-{channel}.json').read_text());logpath=p/(r['taskId']+'.jsonl')
 if not logpath.exists():
  q=requests.get(record['telemetry']['source_detail'],timeout=45);q.raise_for_status();logpath.write_bytes(q.content)
 image=folder/'raw-W.jpg' if i<=5 else old/r['id']/'input.jpg';detail=json.loads((old/r['id']/'targets.json').read_text())['data'];pars=detail[0]['properties']['geolocation'].get('effective_overlay_parameters',{}) if detail else {}
 # No manual correction is inferred; this is an AR prior only.
 data=dict(**record['camera'],shot_time_ms=str(record['image']['shot_time_ms']),debug='true',overlay_params_json=json.dumps(pars))
 files={name:dict(base64=base64.b64encode(b).decode(),mime=mime) for name,b,mime in [('image',image.read_bytes(),'image/jpeg'),('kml',kml,'application/xml'),('log',logpath.read_bytes(),'text/plain')]}
 req=dict(protocol='spatial-replay/v1',operation='ar',input=dict(form=data,files=files));(folder/'ar-replay-request.json').write_text(json.dumps(req));res=subprocess.run(['/Users/wangxiaolin/PycharmProjects/review-annotation-platform/.venv/bin/python','-B','-m','spatial.replay'],cwd='/Users/wangxiaolin/PycharmProjects/drone-diagnostics-integration',input=json.dumps(req),text=True,capture_output=True,timeout=120)
 (folder/'ar-replay-stderr.txt').write_text(res.stderr)
 if res.returncode:print(i,'FAILED',res.stderr[-150:],flush=True);continue
 result=json.loads(res.stdout)['output']
 for key in ('image_png_base64','image_jpg_base64'):
  if result.get(key):(folder/'ar-prior.jpg').write_bytes(base64.b64decode(result.pop(key)))
 result['mapping_verified']=False;result['role']='unverified_AR_prior';(folder/'ar-projection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(i,result.get('overlay_status'),len(result.get('points',[])),flush=True)
