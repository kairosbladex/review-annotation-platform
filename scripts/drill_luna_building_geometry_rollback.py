"""Isolated candidate G -> p000 -> G smoke, never touch shared service configuration."""
import importlib.util,json,os,pathlib,secrets,threading,time
R=pathlib.Path('/data/workspace/jzheng/luna-building-geometry-20261010')
spec=importlib.util.spec_from_file_location('run',R/'src/run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
def main():
 run.E.mkdir(exist_ok=True);run.TOKEN=secrets.token_urlsafe(32)
 cg=next(l.split('::',1)[1] for l in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if '::' in l);run.CG=pathlib.Path('/sys/fs/cgroup')/cg.lstrip('/')
 assert (run.CG/'memory.max').read_text().strip()==str(4*1024**3)
 assert (run.CG/'memory.swap.max').read_text().strip()=='0'
 assert len(os.sched_getaffinity(0))<=3 and run.resources()['available']>=.15
 before=run.shared();run.write('runtime-before.json',dict(shared=before,resources=run.resources()))
 threading.Thread(target=run.guard,daemon=True).start()
 unit=next(x for x in json.loads((R/'regression/units.json').read_text()) if x['unit']=='g0903_d645cc066a5bdc4c');steps=[]
 try:
  for version in ['p000-building-geometry','p000','p000-building-geometry']:
   health=run.launch('G',version);assert health['prompt_version']==version
   out=run.request(unit,'G');assert out['status']==200
   steps.append(dict(version=version,health=health,response=out));run.stop('G')
  run.write('drill.json',dict(steps=steps,shared_unchanged=before==run.shared(),scope='isolated candidate configuration only; B is not an approved release target'))
 finally:
  for label in list(run.CHILDREN):run.stop(label)
  run.STOP.set();run.write('runtime-after.json',dict(shared=run.shared(),resources=run.resources(),children_stopped=True))
if __name__=='__main__':main()
