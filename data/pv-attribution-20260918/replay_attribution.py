"""Replay frozen project-scope cases via actual diagnostic MCP and pinned spatial code."""
from pathlib import Path
import json,hashlib,asyncio,sys,subprocess
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
P=Path(__file__).resolve().parent;OLD=P.parent/'pv-batch10-20260918';SP=Path('/Users/wangxiaolin/PycharmProjects/drone-diagnostics-integration');DG=Path('/Users/wangxiaolin/PycharmProjects/vision-diagnostics');PY=str(P.parents[1]/'.venv/bin/python')
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def poly(box):
 a,b,c,d=box;return dict(type='Polygon',coordinates=[[[a,b],[c,b],[c,d],[a,d],[a,b]]])
cat=read(P/'project-catalog.json');catalogsha=sha(P/'project-catalog.json');items=read(P/'review/batch-input.json');selection={x['id']:x for x in read(OLD/'selection.json')}
config=dict(case_root=str(P),case_ids=[x['event_id'] for x in items],output_root=str(P/'mcp-output'),attribution_inputs={},attribution_provider=dict(provider=dict(cwd=str(SP),command=[PY,'-B','-m','spatial.replay'],source_hashes={n:sha(SP/n) for n in ['spatial/replay.py','spatial/pv_attribution.py']},timeout_seconds=60)))
for x in items:
 eid=x['event_id'];folder=P/eid;record=selection[eid];task=read(P/('task-'+record['taskId']+'.json'))['data'];zoneids={q['pointId'] for q in task['prowledObject']['flightPath'] if q.get('pointId')};taskids=[m['id'] for m in cat['matrices'] if m['zoneId'] in zoneids]
 # Catalog coordinate system is explicit; no fake transform into the image.
 matrices=[]
 for m in cat['matrices']:
  g=json.loads(m['lngLatAlt']);g=g.get('geometry',g);components=[]
  for c in m['blocks']:
   try:cg=json.loads(c['lngLatAlt']);cg=cg.get('geometry',cg)
   except (ValueError,TypeError,KeyError):cg=None
   components.append(dict(id=c['id'],code=c.get('code'),row=c.get('blockRow'),column=c.get('blockColumn'),geometry=cg))
  matrices.append(dict(id=m['id'],name=m['code'],geometry=g,components=components))
 req=dict(case_id=eid,image_sha256=x['image_sha256'],catalog_sha256=catalogsha,coordinate_frame='image_pixel_edges',catalog_coordinate_frame='WGS84_longitude_latitude_altitude',catalog_complete=True,matrices=matrices,task_matrix_ids=taskids,detections=[dict(index=d['index'],region=poly(d['bbox_xyxy'])) for d in x['detections']],mapping_report=dict(producer='spatial-provider',status='unverified_AR_prior',image_sha256=x['image_sha256'],catalog_sha256=catalogsha,error_radius=None,evidence_refs=[str(folder/'ar-projection.json')],reason='NO_INDEPENDENT_ROOF_REGISTRATION_OR_NUMBERING_CHECK',coordinate_frame='image_pixel_edges'))
 path=folder/'attribution-input.json';write(path,req);config['attribution_inputs'][eid]=dict(path=str(path),sha256=sha(path))
write(P/'diagnostic-config.json',config)
async def main():
 out=[]
 async with stdio_client(StdioServerParameters(command=sys.executable,args=['-B','-m','vision_diagnostics.cli','mcp','--config',str(P/'diagnostic-config.json')],cwd=str(DG))) as (r,w):
  async with ClientSession(r,w) as s:
   await s.initialize()
   for x in items:
    result=await s.call_tool('assess_attribution',dict(case_id=x['event_id']));assert not result.isError,result
    decision=json.loads(result.content[0].text);write(P/x['event_id']/'attribution-result.json',decision)
    # Exercise actual Java preview against deterministic provider output, without DB.
    cp=str(Path('/private/tmp/pv-attribution-java'))+':'+Path('/private/tmp/pv-attribution-java/classpath.txt').read_text();cmd=['/Library/Java/JavaVirtualMachines/zulu-8.jdk/Contents/Home/bin/java','-cp',cp,'com.fan.framework.inspection.service.support.PhotovoltaicAttributionPreview']
    j=subprocess.run(cmd,input=json.dumps({'targets':decision['targets']}),text=True,capture_output=True,check=True);java=json.loads(j.stdout);write(P/x['event_id']/'java-preview.json',java)
    assert java['description']==decision['description'] and java.get('target_location_id')==decision.get('target_location_id')
    detail=read(OLD/x['event_id']/'detail.json')['data'];out.append(dict(number=x['number'],event_id=x['event_id'],count=len(x['detections']),old_facility_id=detail.get('targetLocationId'),old_facility=detail.get('targetLocation'),old_description=detail.get('taskInfo'),new=decision,old_nearest_fallback='NOT_PROVEN_PER_EVENT_NO_BRANCH_TRACE',task_matrix_ids=read(P/x['event_id']/'attribution-input.json')['task_matrix_ids']))
 write(P/'comparison.json',out);print('MCP and Java replay',len(out),'events',sum(x['count'] for x in out),'targets')
asyncio.run(main())
