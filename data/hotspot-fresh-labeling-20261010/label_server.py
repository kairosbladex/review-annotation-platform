"""Local manual box-and-label workbench. Never calls models or business APIs."""
import argparse,csv,datetime,json,math,os,random,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
HERE=Path(__file__).resolve().parent
FIELDS=['key','label','bbox','screenshot_bbox','note','labeled_at'];LABELS={'是光斑','不是光斑','看不清'};LOCK=threading.Lock()
def targets():return json.loads((HERE/'targets.json').read_text())['items']
def load():
 with (HERE/'annotations.csv').open(encoding='utf-8-sig',newline='') as f:return {r['key']:r for r in csv.DictReader(f)}
def inverse(m):
 a,b,c=m[0];d,e,f=m[1];g,h,i=m[2];det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
 if abs(det)<1e-12:raise ValueError('无法对应截图')
 return [[(e*i-f*h)/det,(c*h-b*i)/det,(b*f-c*e)/det],[(f*g-d*i)/det,(a*i-c*g)/det,(c*d-a*f)/det],[(d*h-e*g)/det,(b*g-a*h)/det,(a*e-b*d)/det]]
def validate(body):
 if not isinstance(body,dict):raise ValueError('请求必须是对象')
 row=next((r for r in targets() if r['key']==body.get('key')),None)
 if row is None or body.get('label') not in LABELS:raise ValueError('目标或标签无效')
 box=body.get('bbox');mapped=None
 if box is None:
  if body['label']!='看不清':raise ValueError('请先在原图上框出对应目标')
 else:
  if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (float,int) or not math.isfinite(v) for v in box) or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ValueError('方框范围无效')
  w,h=row['raw_size'];sw,sh=row['screenshot_size'];M=inverse(row['homography']);pts=[]
  for x,y in [(box[0],box[1]),(box[2],box[1]),(box[0],box[3]),(box[2],box[3])]:
   q=[sum(a*b for a,b in zip(line,[x*w,y*h,1])) for line in M]
   if abs(q[2])<1e-12:raise ValueError('无法对应截图')
   pts.append([q[0]/q[2]/sw,q[1]/q[2]/sh])
  mapped=[min(x[0] for x in pts),min(x[1] for x in pts),max(x[0] for x in pts),max(x[1] for x in pts)]
  if not all(math.isfinite(v) and 0<=v<=1 for v in mapped):raise ValueError('方框超出对应截图范围，请缩小或选择看不清并清除方框')
 return {'key':row['key'],'label':body['label'],'bbox':json.dumps(box) if box else '', 'screenshot_bbox':json.dumps(mapped) if mapped else '', 'note':str(body.get('note',''))[:300],'labeled_at':datetime.datetime.now().astimezone().isoformat()}
def save(row):
 with LOCK:
  rows=load();rows[row['key']]=row;p=HERE/'annotations.csv';tmp=p.with_suffix('.tmp')
  with tmp.open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
   for t in targets():w.writerow({k:rows.get(t['key'],{}).get(k,'') for k in FIELDS})
  tmp.replace(p)
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def reply(self,status,body,mime='application/json'):
  data=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
 def local(self):return self.headers.get('Host','').split(':')[0] in ('127.0.0.1','localhost')
 def do_GET(self):
  if not self.local():return self.reply(403,{'error':'仅供本机使用'})
  path=urlparse(self.path).path
  if path=='/':return self.reply(200,(HERE/'index.html').read_bytes(),'text/html; charset=utf-8')
  if path=='/api/rows':
   saved=load();rows=[]
   for r in targets():
    s=saved.get(r['key'],{});rows.append({**r,'label':s.get('label',''),'bbox':json.loads(s['bbox']) if s.get('bbox') else None,'note':s.get('note','')})
   random.Random(20261010).shuffle(rows);return self.reply(200,{'rows':rows})
  if path.startswith('/images/'):
   f=(HERE/path.lstrip('/')).resolve();allowed={(HERE/r[k]).resolve() for r in targets() for k in ('raw_image','screenshot_image')}
   if f in allowed and f.is_file():return self.reply(200,f.read_bytes(),'image/jpeg')
  self.reply(404,{'error':'not found'})
 def do_POST(self):
  origin=self.headers.get('Origin');expected='http://'+self.headers.get('Host','')
  if not self.local() or (origin and origin!=expected):return self.reply(403,{'error':'仅供本机使用'})
  if self.path!='/api/label':return self.reply(404,{'error':'not found'})
  try:
   n=int(self.headers.get('Content-Length','0'))
   if not 0<n<8192 or self.headers.get('Content-Type')!='application/json':raise ValueError('请求格式无效')
   body=json.loads(self.rfile.read(n));row=validate(body);save(row);self.reply(200,{'ok':True})
  except (ValueError,TypeError,KeyError):self.reply(400,{'error':'请框出有效目标范围；找不到时清除方框并选看不清。'})
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8802);p.add_argument('--data',type=Path);a=p.parse_args()
 global HERE
 if a.data:HERE=a.data.resolve()
 print(f'本机光斑标注页 http://127.0.0.1:{a.port}/；共 {len(targets())} 处；仅写 {HERE}/annotations.csv',flush=True)
 ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
