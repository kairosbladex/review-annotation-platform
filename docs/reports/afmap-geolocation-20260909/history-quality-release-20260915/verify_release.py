"""验证指定发布包仍保留本次定位补丁；只读，不启动服务或写数据库。"""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jar',required=True);p.add_argument('--manifest',default=str(Path(__file__).with_name('manifest.json')));a=p.parse_args()
m=json.loads(Path(a.manifest).read_text());failures=[];checked=0
with zipfile.ZipFile(a.jar) as z:
 for jar,mod in m['modules'].items():
  if jar not in z.namelist():failures.append(jar);continue
  with zipfile.ZipFile(io.BytesIO(z.read(jar))) as module:
   for name,expected in mod['classHashes'].items():
    checked+=1
    if name not in module.namelist() or hashlib.sha256(module.read(name)).hexdigest()!=expected:failures.append(jar+':'+name)
print(json.dumps({'passed':not failures,'classes_checked':checked,'mismatched':failures},ensure_ascii=False))
raise SystemExit(1 if failures else 0)
