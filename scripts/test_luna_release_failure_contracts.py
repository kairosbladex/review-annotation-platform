"""Fault injection into unchanged release worker; no upstream call or business write."""
import base64,importlib.util,io,json,os,pathlib,subprocess,tempfile,threading,unittest,urllib.request,urllib.error
from unittest.mock import patch,MagicMock
from PIL import Image
P=pathlib.Path(os.environ['LUNA_RELEASE_SCRIPT']);spec=importlib.util.spec_from_file_location('release_worker',P);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class FailureContracts(unittest.TestCase):
 def setUp(self):
  b=io.BytesIO();Image.new('RGB',(80,60)).save(b,format='PNG');self.payload={'image':base64.b64encode(b.getvalue()).decode(),'candidates':[{'target_index':0,'class_name':'违建','bbox':[.1,.1,.4,.4]}]}
 def test_http_execution_failure_not_false_label(self):
  server=w.Server(('127.0.0.1',0),w.Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
  try:
   with patch.dict(os.environ,{'CODEX_REVIEW_WORKER_TOKEN':'fixture'}),patch.object(w,'execute',side_effect=RuntimeError('injected CLI failure')):
    req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/review',data=json.dumps(self.payload).encode(),headers={'Authorization':'Bearer fixture'})
    with self.assertRaises(urllib.error.HTTPError) as ctx:urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req)
    self.assertEqual(ctx.exception.code,503);self.assertNotIn('results',json.load(ctx.exception))
  finally:server.shutdown();server.server_close();thread.join()
 def execute_fake(self,events,timeout=False):
  with tempfile.TemporaryDirectory() as td:
   pathlib.Path(td,'auth.json').write_text('{"OPENAI_API_KEY":"fixture"}')
   with patch.dict(os.environ,{'CODEX_REVIEW_HOME':td,'CODEX_REVIEW_BINARY':'/fixture/codex'}),patch.object(w.subprocess,'Popen') as popen,patch.object(w,'terminate') as terminate:
    p=popen.return_value;p.returncode=0
    if timeout:p.communicate.side_effect=subprocess.TimeoutExpired('fixture',215)
    else:p.communicate.return_value=('\n'.join(json.dumps(e) for e in events),'')
    with self.assertRaises((RuntimeError,ValueError,IndexError)):w.execute(self.payload)
    if timeout:terminate.assert_called_once_with(p)
 def test_timeout_terminates_child(self):self.execute_fake([],True)
 def test_tool_call_rejected(self):self.execute_fake([{'type':'item.completed','item':{'type':'command_execution'}}])
 def test_target_mismatch_rejected(self):
  result={'results':[{'target_index':2,'decision':'true','reasoning':'fixture'}]}
  self.execute_fake([{'type':'item.completed','item':{'type':'agent_message','text':json.dumps(result)}}])
 def test_invalid_decision_rejected(self):
  result={'results':[{'target_index':0,'decision':'invalid','reasoning':'fixture'}]}
  self.execute_fake([{'type':'item.completed','item':{'type':'agent_message','text':json.dumps(result)}}])
 def test_no_result_rejected(self):self.execute_fake([])
if __name__=='__main__':unittest.main()
