"""Own-process resource guard for the local manual labeling service on macOS."""
import argparse,datetime,json,os,pathlib,resource,signal,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8802);a=p.parse_args()
stop=False
def on_stop(*_):
 global stop
 stop=True
signal.signal(signal.SIGTERM,on_stop);signal.signal(signal.SIGINT,on_stop)
def caps():resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
child=subprocess.Popen([sys.executable,str(HERE/'label_server.py'),'--port',str(a.port)],preexec_fn=caps)
record={'pid':os.getpid(),'child_pid':child.pid,'url':f'http://127.0.0.1:{a.port}/','started_at':datetime.datetime.now().astimezone().isoformat(),'guard':'5-second RSS/CPU sampling; stop own child above 512MiB or 200% CPU; not a kernel memory hard limit','gpu':False,'stop_command':f'kill {os.getpid()}','purpose':'用户手动标注；无模型调用和业务写入'}
(HERE/'local-instance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
try:
 while not stop and child.poll() is None:
  r=subprocess.run(['ps','-o','rss=,pcpu=','-p',str(child.pid)],capture_output=True,text=True)
  fields=r.stdout.split()
  if len(fields)>=2:
   rss,cpu=float(fields[0]),float(fields[1]);record.update(rss_kib=rss,cpu_percent=cpu)
   (HERE/'local-instance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
   if rss>512*1024 or cpu>200:
    print('本机标注服务超过资源阈值，停止本次子进程',flush=True);break
  for _ in range(10):
   if stop:break
   time.sleep(.5)
finally:
 if child.poll() is None:
  child.terminate()
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:child.kill();child.wait()
 record['stopped_at']=datetime.datetime.now().astimezone().isoformat();(HERE/'local-instance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
