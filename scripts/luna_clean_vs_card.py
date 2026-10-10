#!/usr/bin/env python3
"""固定 CLI 的 2x2 配对评测；仅访问独立 loopback worker。"""
import argparse, base64, concurrent.futures, hashlib, json, os, random, secrets, socket, subprocess, time, urllib.request
from pathlib import Path
import luna_clean_only_eval as base
from luna_clean_frame_eval import AUTO, health, read, sha, write, promptlib
PORT = 18400
PREFIX = 'luna-clean-vs-card-20261010'
BINARY = Path('/data/workspace/jzheng/codex-luna-review/bin/codex-0.162.0-alpha.17.2/codex')
SEED = 20261010

def payload(root, unit, arm):
    clean = arm in 'CD'
    path = root/'clean'/(unit['unit']+'.jpg') if clean else AUTO/'assets/images'/(unit['unit']+'.jpg')
    assert sha(path) == unit['clean_sha256' if clean else 'sha256']
    candidates = [dict(target_index=i, class_name=c['class_name'], bbox=c['clean_bbox' if clean else 'bbox']) for i,c in enumerate(unit['candidates'])]
    doc = read(root/'baseline/p000.json')
    if arm in 'BD' and all(promptlib.card_key(c['class_name'])=='违建' for c in candidates):
        doc['cards'] = {'违建': read(root/'baseline/p002.json')['cards']['违建']}
    prompt = promptlib.build(doc,candidates)
    return {'image':base64.b64encode(path.read_bytes()).decode(), 'candidates':candidates, 'prompt':prompt}

def run(root):
    e=root/'evidence'; units=read(root/'baseline/included.json')
    assert len(units)==300 and read(e/'prepare-complete.json')['images']==300
    binary_sha=sha(BINARY); assert binary_sha==read(e/'protocol.json')['binary_sha256']
    rng=random.Random(SEED); rng.shuffle(units); jobs=[]
    for u in units:
        conditions=[(a,n) for a in 'ABCD' for n in (1,2)];rng.shuffle(conditions)
        jobs.extend((u,a,n) for a,n in conditions)
    write(e/'schedule.json',[{'unit':u['unit'],'arm':a,'repeat':n} for u,a,n in jobs])
    token=(root/'token').read_text().strip()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def one(u,a,n):
        dest=e/'results'/f'{a}-{n}'/(u['unit']+'.json')
        if dest.exists() and read(dest).get('success'):
            assert read(dest)['binary_sha256']==binary_sha
            return True
        assert sha(BINARY)==binary_sha,'CLI changed'
        state=health(); request=payload(root,u,a); started=time.time(); attempts=read(dest)['attempts'] if dest.exists() else []; result=None; body={}
        for retry in range(2):
            t=time.time()
            try:
                req=urllib.request.Request(f'http://127.0.0.1:{PORT}/review',data=json.dumps(request).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
                with opener.open(req,timeout=240) as response: body=json.load(response)
                by={x['target_index']:x for x in body['results']}
                assert set(by)==set(range(len(u['candidates'])))
                result=[dict(key=c['key'],**by[i]) for i,c in enumerate(u['candidates'])]; status='ok'
            except Exception as err: status=f'http_{err.code}' if hasattr(err,'code') else type(err).__name__
            attempts.append({'at':t,'seconds':time.time()-t,'status':status})
            if result is not None: break
            health();time.sleep(5)
        assert sha(BINARY)==binary_sha,'CLI changed'
        write(e/'results'/f'{a}-{n}'/(u['unit']+'.json'),{'unit':u['unit'],'arm':a,'repeat':n,'success':result is not None,'results':result or [],'attempts':attempts,'started':started,'finished':time.time(),'health':state,'execution_id':body.get('execution_id'),'requested_model':body.get('requested_model'),'binary_sha256':binary_sha,'prompt_sha256':hashlib.sha256(request['prompt'].encode()).hexdigest(),'image_sha256':u['clean_sha256' if a in 'CD' else 'sha256']})
        print(f'{a}-{n} {u["unit"]} {status}',flush=True)
        return result is not None
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        pending=iter(jobs);active=set();consecutive_failures=0
        while True:
            while len(active)<6:
                try:active.add(pool.submit(one,*next(pending)))
                except StopIteration:break
            if not active:break
            done,active=concurrent.futures.wait(active,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                consecutive_failures=0 if f.result() else consecutive_failures+1
                if consecutive_failures>=12:raise RuntimeError('连续12次请求失败，停止本次实验')
    write(e/'run-complete.json',{'at':time.time(),'binary_sha256':sha(BINARY),'health':health()})

def setup(root):
    health();e=root/'evidence';e.mkdir(exist_ok=True)
    write(e/'protocol.json',{'seed':SEED,'arms':{'A':'截图+p000','B':'截图+p000-building-only','C':'干净帧三视图+p000','D':'干净帧三视图+p000-building-only'},'images':300,'repeats':2,'requests':2400,'concurrency':4,'port':PORT,'binary':str(BINARY),'binary_sha256':sha(BINARY),'binary_version':subprocess.check_output([str(BINARY),'--version'],text=True).strip(),'labels':'原冻结标签；54项复核均未填写，无更正清单','prompt_source_commit':'42397238abe9e6fe544af826478f61757fb748ee','hashes':{p.name:sha(p) for p in (root/'baseline').glob('*.json')},'gate':'holdout主判据：D-C违建准确率提升区间下界>0；车辆uncertain差值区间上界<0且人员非目标误放差值区间上界<=0。仅点估计改善不足以推荐。','bootstrap':'按原视频/原照片聚类，重复先平均，2000次百分位95%区间；dev/holdout分别报告；多比较为探索性区间。','started':time.time()})
    base.launch(root,'prepare','2G','100%',['/usr/bin/python3',str(root/'src/luna_clean_vs_card.py'),'prepare','--root',str(root)])

def start(root):
    health(); assert read(root/'evidence/prepare-complete.json')['images']==300
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',PORT))
    os.umask(0o077)
    env=dict(x.split('=',1) for x in (AUTO/'private/worker.env').read_text().splitlines() if '=' in x and not x.startswith('#'))
    token=secrets.token_urlsafe(32);(root/'token').write_text(token);(root/'tmp').mkdir(exist_ok=True)
    private={'CODEX_REVIEW_BINARY':str(BINARY),'CODEX_REVIEW_HOME':env['CODEX_REVIEW_HOME'],'CODEX_REVIEW_WORKER_TOKEN':token,'CODEX_REVIEW_BIND':'127.0.0.1','CODEX_REVIEW_PORT':str(PORT),'CODEX_REVIEW_TMP_DIR':str(root/'tmp'),'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1'}
    (root/'worker.env').write_text(''.join(f'{k}={v}\n' for k,v in private.items()))
    wrapper=root/'src/worker6.py'
    wrapper.write_text('import sys,threading\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent/"luna_prompt_autotune"))\nimport luna_experiment_worker as w\nw.LIMIT=6\nw.SLOTS=threading.BoundedSemaphore(6)\nw.main()\n')
    try:
        base.launch(root,'worker','4G','300%',['/usr/bin/python3',str(wrapper)],['-p','EnvironmentFile='+str(root/'worker.env')])
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(20):
            try:
                with opener.open(f'http://127.0.0.1:{PORT}/health',timeout=2) as response:assert json.load(response)['max_concurrency']==6
                break
            except OSError:time.sleep(1)
        else:raise RuntimeError('worker not ready')
        props=base.properties(PREFIX+'-worker');cg=Path('/sys/fs/cgroup')/props['ControlGroup'].lstrip('/')
        limits={k:(cg/k).read_text().strip() for k in ('memory.max','memory.swap.max')};cpus=sorted(os.sched_getaffinity(int(props['MainPID'])))
        assert limits=={'memory.max':str(4*1024**3),'memory.swap.max':'0'} and len(cpus)<=3
        limit_record={'limits':limits,'allowed_cpus':cpus,'concurrency':6,'at':time.time(),'enforced_before_first_request':True}
        if not (root/'evidence/effective-limits-6.json').exists():write(root/'evidence/effective-limits-6.json',limit_record)
        else:
            with (root/'evidence/resume-limits.jsonl').open('a') as f:f.write(json.dumps(limit_record)+'\n')
        startup={'at':time.time(),'before':4,'after':6,'reason':'首批资源测量后提高至用户允许上限，未查看类别结果；保留已成功结果，固定CLI/输入/随机顺序不变','preserved_successes':sum(read(p).get('success',False) for p in (root/'evidence/results').glob('*/*.json')),'binary_sha256':sha(BINARY),'script_sha256':sha(__file__)}
        if not (root/'evidence/concurrency-amendment.json').exists():write(root/'evidence/concurrency-amendment.json',startup)
        else:
            startup.update(before=6,after=6,reason='仅补跑未成功的调用；保留既有成功结果与失败尝试')
            with (root/'evidence/resumptions.jsonl').open('a') as f:f.write(json.dumps(startup,ensure_ascii=False)+'\n')
        for role,memory,cpu in [('runner','1G','100%'),('watch','256M','100%')]:
            base.launch(root,role,memory,cpu,['/usr/bin/python3',str(root/'src/luna_clean_vs_card.py'),'run' if role=='runner' else 'watch','--root',str(root)])
    except BaseException:
        subprocess.run(['systemctl','--user','stop',PREFIX+'-runner',PREFIX+'-worker'],capture_output=True);raise

base.PORT=PORT;base.PREFIX=PREFIX
original_launch=base.launch
def launch(root,role,memory,cpu,command,extra=()):
    command=[s.replace('/src/luna_clean_only_eval.py','/src/luna_clean_vs_card.py') for s in command]
    return original_launch(root,role,memory,cpu,command,extra)
base.launch=launch
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['setup','prepare','start','run','watch']);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    if a.action=='start':start(a.root)
    elif a.action=='prepare':base.prepare(a.root)
    elif a.action=='watch':base.watch(a.root)
    else:globals()[a.action](a.root)
