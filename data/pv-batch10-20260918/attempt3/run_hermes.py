from pathlib import Path
import os,json,yaml,subprocess,fcntl,sqlite3
root=Path(__file__).resolve().parent
lock=open('/runs/visual-probe.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
db=sqlite3.connect('file:/runs/jobs.sqlite?mode=ro',uri=True)
assert not db.execute("select id from jobs where state in ('QUEUED','RUNNING','CANCEL_REQUESTED')").fetchall(),'ACTIVE_JOB'
source=Path('/runs/8851a77d7f233f2b70e4c6afadeafdab224997b18d0d109ced8c7d150401bb31/attempt-0001/hermes/config.yaml')
c=yaml.safe_load(source.read_text());c['agent']['max_turns']=8
c['mcp_servers']={'diagnostics':dict(timeout=3000,command='python',args=[str(root/'mcp_batch.py')],tools=dict(include=['review_frozen_batch']))}
home=root/'hermes';home.mkdir(exist_ok=True);(home/'config.yaml').write_text(yaml.safe_dump(c))
prompt='仅使用 diagnostics 中的 review_frozen_batch 工具，对已冻结10个测试环境光伏光斑事件执行一次批量复核。必须实际调用该工具，不使用其他工具，不重试。工具运行可能需要数分钟。若schema_valid不为true则该视觉结果无效，不可引用为识别结论。缺少参考和人工辅助是已知约束，不推测阈值过高，不建议放宽验收。1至5是带旧叠加的可见光预警照片，6至10是冻结热成像原图，不能混淆。之后用中文概括各事件的真实结果，区分模型生成、视觉异常、几何配准、目录映射和业务验收。几何候选不是业务确认；无锚点未知，禁止把已有事件描述当真值。给出基于失败证据的改进建议，不修改代码、模型或业务数据。'
(root/'hermes-prompt.txt').write_text(prompt)
env={k:os.environ[k] for k in ('PATH','LANG','SSL_CERT_FILE') if k in os.environ};env.update(HERMES_HOME=str(home),HOME=str(home),OPENAI_API_KEY='local-only',PYTHONUNBUFFERED='1',NO_PROXY='*')
with (root/'hermes.log').open('w') as log:
 r=subprocess.run(['hermes','chat','--query-file',str(root/'hermes-prompt.txt'),'--oneshot','--reasoning','none','--toolsets','diagnostics'],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=3000)
(root/'hermes-exit.json').write_text(json.dumps(dict(returncode=r.returncode)));print('Hermes exit',r.returncode)
