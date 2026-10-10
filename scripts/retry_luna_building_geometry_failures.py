"""Retry only transport/execution failures, keeping all first-pass evidence intact."""
import importlib.util,json,os,pathlib,secrets,threading
R=pathlib.Path('/data/workspace/jzheng/luna-building-geometry-20261010')
spec=importlib.util.spec_from_file_location('run',R/'src/run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
def main():
 run.E.mkdir(exist_ok=True);run.TOKEN=secrets.token_urlsafe(32)
 run.CG=pathlib.Path('/sys/fs/cgroup')/next(l.split('::',1)[1] for l in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if '::' in l).lstrip('/')
 assert (run.CG/'memory.max').read_text().strip()==str(4*1024**3) and (run.CG/'memory.swap.max').read_text().strip()=='0'
 assert len(os.sched_getaffinity(0))<=3 and run.resources()['available']>=.15
 hashes=json.loads((R/'regression/package-hashes.json').read_text())
 for p,h in hashes.items():assert run.sha(R/p)==h
 before=run.shared();run.write('runtime-before.json',dict(shared=before,resources=run.resources()))
 threading.Thread(target=run.guard,daemon=True).start();units={u['unit']:u for u in json.loads((R/'regression/units.json').read_text())};jobs=[]
 for p in sorted((R/'regression/results').glob('*/*.json')):
  z=json.loads(p.read_text())
  if z['status']!=200:jobs.append(z)
 run.write('selection.json',[dict(unit=z['unit'],arm=z['arm'],repeat=z['repeat'],original_status=z['status']) for z in jobs]);results=[]
 try:
  for arm in sorted({z['arm'] for z in jobs}):run.launch(arm,'p000-building-only' if arm=='B' else 'p000-building-geometry')
  for z in jobs:
   out=run.request(units[z['unit']],z['arm'],z['repeat']);run.write(f'results/{z["arm"]}-{z["repeat"]}/{z["unit"]}.json',out);results.append(out['status'])
  run.write('run-complete.json',dict(total=len(jobs),success=sum(x==200 for x in results),shared_unchanged=before==run.shared()))
 finally:
  for arm in list(run.CHILDREN):run.stop(arm)
  run.STOP.set();run.write('runtime-after.json',dict(shared=run.shared(),resources=run.resources(),children_stopped=True))
if __name__=='__main__':main()
