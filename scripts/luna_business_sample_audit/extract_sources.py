"""Extract only small provenance files from immutable Git commits into private storage."""
import argparse,hashlib,json,subprocess
from pathlib import Path
SOURCES=[
 ('autotune.json','yoloe','85d10aea9963042cb62c766c616ff4d0d8760931','data/luna-prompt-autotune-final-20261009/assets/items.json'),
 ('building.json','review','tests/luna-building-release-validation','data/luna-building-release-validation-20261010/execution-units.json'),
 ('crack_hd_labels.json','review','feature/road-crack-labeling','data/road-crack-relabel-hd-20261010/labels-road-crack-hd.json'),
]
def run(review,yoloe,out):
 evidence=[];private=out/'private';private.mkdir(parents=True,exist_ok=True)
 for dest,which,ref,path in SOURCES:
  repo={'review':review,'yoloe':yoloe}[which];commit=subprocess.check_output(['git','-C',str(repo),'rev-parse',ref],text=True).strip()
  content=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+path]);(private/dest).write_bytes(content)
  evidence.append({'repo':repo.name,'commit':commit,'path':path,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
 for name in ['source_manifest.jsonl','input/inference_manifest.jsonl']:
  p=review/'data/ktd-test-real-monitoring-full-20260924'/name;b=p.read_bytes();evidence.append({'repo':review.name,'path':str(p.relative_to(review)),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'version':'local frozen snapshot; untracked raw metadata'})
 (out/'source-manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--review',type=Path,required=True);a.add_argument('--yoloe',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args();run(x.review,x.yoloe,x.out)
