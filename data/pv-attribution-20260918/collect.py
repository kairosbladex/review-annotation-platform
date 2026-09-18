import subprocess,json,hashlib,datetime,pathlib,requests,sys
out=pathlib.Path('/Users/wangxiaolin/PycharmProjects/review-annotation-platform/data/pv-attribution-20260918')
base='https://lxtest.zhuoyilingxi.com:8088/gis_server_a'
s=requests.Session()
def api(path,**kwargs):
 r=s.request(kwargs.pop('method','GET'),base+path,timeout=30,allow_redirects=False,**kwargs)
 if r.status_code!=200:raise RuntimeError('HTTP_STATUS_'+str(r.status_code))
 d=r.json()
 if d.get('code') not in (0,200,'0','200'):raise RuntimeError('BUSINESS_CODE_'+str(d.get('code'))+' PATH '+path+(' MSG '+str(d.get('msg',d.get('message'))) if path=='/screen/warningPage' else ''))
 return d
try:
 public=api('/sm2/public')['data']
 secret=subprocess.run(['security','find-generic-password','-s','codex.afmap.lxtest.zhuoyilingxi.com.8088','-a','adm','-w'],capture_output=True,text=True)
 if secret.returncode:raise RuntimeError('KEYCHAIN_UNAVAILABLE_'+str(secret.returncode))
 mode=public.get('encryptType','').lower()
 headers={}
 if mode=='none' and not public.get('isOpen'):
  username,password='adm',secret.stdout.rstrip('\n')
 elif mode=='sm2':
  js="const fs=require('fs'),sm2=require('/Users/wangxiaolin/PycharmProjects/onegismap_a-test-provenance-20260916/node_modules/sm-crypto').sm2;let p=JSON.parse(fs.readFileSync(0,'utf8')); process.stdout.write(JSON.stringify(p.values.map(v=>'04'+sm2.doEncrypt(v,p.key,1)))));"
  enc=subprocess.run(['node','-e',js],input=json.dumps(dict(key=public['publicKey'],values=['adm',secret.stdout.rstrip('\n')])),capture_output=True,text=True,check=True)
  username,password=json.loads(enc.stdout);headers={'SM2PublicKey':public['publicKey']}
 else:raise RuntimeError('UNSUPPORTED_LOGIN_ENCRYPTION')
 secret=None
 auth=api('/login',method='POST',headers=headers,data=dict(username=username,password=password,vcCode='',vcToken='',clientId='6',loginCount=0,sceneType=''))
 password=None
 token=next(x['token'] for x in auth['data'] if str(x['clientId'])=='6');s.headers['Authorization']=token;auth=None
 s.headers['projectId']='2c95908e958f3b75019593551f520126'
 selected=json.loads((out.parent/'pv-batch10-20260918/selection.json').read_text())
 paths=[('matrices','/panelMatrix/list?withBlock=true')]+[('task-'+tid,'/task/'+tid) for tid in sorted(set(r['taskId'] for r in selected))]
 for name,path in paths:
  try:
   d=api(path);(out/(name+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2));print(name,'OK',flush=True)
  except RuntimeError as exc:print(name,str(exc),flush=True)
except Exception as exc:
 print(str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__);sys.exit(1)
