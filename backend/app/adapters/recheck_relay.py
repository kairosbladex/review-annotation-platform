"""Official CLI transport with response metadata captured by a loopback bridge.

Only the configured relay receives image bytes. No proxy payload/auth logging.
"""
from __future__ import annotations
import base64
import hmac
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import httpx
from .codex_responses import _extract_output_text, _parse_json_text

class RelayFailure(RuntimeError):
    def __init__(self, code, retry_after=None):
        super().__init__(code)
        self.code=code; self.retry_after=retry_after

class RecheckRelay:
    def __init__(self, settings):
        self.settings=settings
        base=settings.auto_annotation_base_url.rstrip('/')
        parsed=urlparse(base)
        if parsed.scheme not in {'https','http'} or not parsed.netloc or parsed.username:
            raise ValueError('invalid configured relay')
        self.base=base if base.endswith('/v1') else base+'/v1'
        self._usage=None; self._usage_at=0; self._lock=threading.Lock()

    def usage(self, force=False):
        with self._lock:
            if not force and self._usage is not None and time.monotonic()-self._usage_at<60:
                return self._usage
            try:
                with httpx.Client(timeout=10, trust_env=False, follow_redirects=False,
                                  verify=self.settings.auto_annotation_verify_tls) as c:
                    r=c.get(self.base+'/usage',headers={'Authorization':'Bearer '+self.settings.auto_annotation_api_key})
                if r.status_code in (401,403): raise RelayFailure('usage_auth_failed')
                if r.status_code!=200:raise RelayFailure('usage_unavailable')
                d=r.json();remaining=d.get('remaining',d.get('quota',{}).get('remaining'))
                if not isinstance(remaining,(int,float)) or isinstance(remaining,bool):raise RelayFailure('usage_invalid')
                import math
                if not math.isfinite(remaining):raise RelayFailure('usage_invalid')
                status=d.get('status');valid=d.get('isValid')
                if valid is not True or status not in {'active','valid','正常',1,'enabled'}:
                    raise RelayFailure('usage_account_inactive')
                if d.get('unit')!='USD':raise RelayFailure('usage_unknown_currency')
                result={'remaining':remaining,'status':status,'unit':d.get('unit'),'isValid':valid}
                self._usage=result;self._usage_at=time.monotonic();return result
            except RelayFailure:raise
            except Exception as e:raise RelayFailure('usage_unavailable') from e

    def request(self, *, model, prompt, images=(), schema=None, idempotency_key=None):
        if not self.settings.auto_annotation_api_key:raise RelayFailure('missing_credential')
        if self.settings.auto_annotation_use_codex_cli:
            return self._cli(model,prompt,images,schema,idempotency_key)
        content=[{'type':'input_text','text':prompt}]+[{'type':'input_image','image_url':'data:image/jpeg;base64,'+base64.b64encode(x).decode(),'detail':'high'} for x in images]
        try:
            with httpx.Client(timeout=self.settings.auto_annotation_timeout_seconds,trust_env=False,
                              verify=self.settings.auto_annotation_verify_tls,follow_redirects=False) as c:
                r=c.post(self.base+'/responses',headers={'Authorization':'Bearer '+self.settings.auto_annotation_api_key,
                        'Idempotency-Key':idempotency_key or secrets.token_hex(16)},
                        json={'model':model,'input':[{'role':'user','content':content}], 'store':False, 'max_output_tokens':1024})
            if r.status_code==429:raise RelayFailure('rate_limited',self._retry_after(r.headers.get('retry-after')))
            if r.status_code in (401,403):raise RelayFailure('auth_failed')
            if r.status_code!=200:raise RelayFailure('request_failed')
            d=r.json();return {'answer':_parse_json_text(_extract_output_text(d)),'actual_model':d.get('model'), 'usage':d.get('usage',{}),'response_id':d.get('id')}
        except RelayFailure:raise
        except (httpx.TimeoutException,httpx.NetworkError) as e:raise RelayFailure('execution_unknown') from e
        except Exception as e:raise RelayFailure('invalid_response') from e

    @staticmethod
    def _retry_after(value):
        try:return max(1,float(value))
        except (ValueError,TypeError):
            from email.utils import parsedate_to_datetime
            from datetime import datetime,timezone
            try:return max(1,(parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds())
            except Exception:return 60

    def _cli(self,model,prompt,images,schema,idempotency_key):
        relay=self;token=secrets.token_hex(24);meta={};t0=time.monotonic()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                if self.path!='/v1/responses' or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                    self.send_error(403);return
                length=int(self.headers.get('Content-Length','0'))
                if length>32_000_000:self.send_error(413);return
                body=self.rfile.read(length)
                headers={k:v for k,v in self.headers.items() if k.lower() not in {'host','connection','content-length','authorization','accept-encoding'}}
                headers['Authorization']='Bearer '+relay.settings.auto_annotation_api_key
                headers['Idempotency-Key']=idempotency_key or secrets.token_hex(16)
                # CLI retries are disabled and the bridge permits one request only.
                if meta.get('sent'):self.send_error(409);return
                meta['sent']=True
                try:
                    with httpx.Client(timeout=relay.settings.auto_annotation_timeout_seconds,trust_env=False,verify=relay.settings.auto_annotation_verify_tls,follow_redirects=False) as c:
                        with c.stream('POST',relay.base+'/responses',headers=headers,content=body) as response:
                            meta['http_status']=response.status_code;meta['retry_after']=response.headers.get('retry-after')
                            self.send_response(response.status_code);self.send_header('Content-Type',response.headers.get('content-type','text/event-stream'));self.send_header('Connection','close');self.end_headers()
                            pending=b''
                            for chunk in response.iter_bytes():
                                self.wfile.write(chunk);self.wfile.flush();pending+=chunk
                                while b'\n' in pending:
                                    line,pending=pending.split(b'\n',1)
                                    if line.startswith(b'data: '):
                                        try:
                                            e=json.loads(line[6:]);r=e.get('response',{})
                                            for k in ('model','usage','id'):
                                                if r.get(k) is not None:meta[k]=r[k]
                                            if e.get('type')=='response.completed':meta['completed']=True
                                        except (ValueError,AttributeError):pass
                            self.close_connection=True
                except Exception:meta['transport_error']=True;self.close_connection=True
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        home=Path(tempfile.mkdtemp(prefix='recheck-cli-'));work=home/'input';work.mkdir()
        try:
            (home/'config.toml').write_text('model_provider = "recheck_relay"\nmodel_reasoning_effort = "low"\n[features]\nshell_tool = false\n[model_providers.recheck_relay]\nname = "Recheck relay"\nbase_url = "http://127.0.0.1:'+str(server.server_port)+'/v1"\nwire_api = "responses"\nenv_key = "RECHECK_LOOPBACK_TOKEN"\nrequest_max_retries = 0\nstream_max_retries = 0\n')
            command=[self.settings.auto_annotation_codex_binary,'exec','--ephemeral','--json','--skip-git-repo-check','--ignore-rules','-C',str(work),'--sandbox','read-only','-m',model,'-c','web_search="disabled"']
            if schema:
                (home/'schema.json').write_text(json.dumps(schema));command+=['--output-schema',str(home/'schema.json')]
            for i,data in enumerate(images):
                path=work/f'{i}.jpg';path.write_bytes(data);command+=['-i',str(path)]
            command+=['-']
            result=subprocess.run(command,input=prompt,text=True,capture_output=True,env={**os.environ,'CODEX_HOME':str(home),'RECHECK_LOOPBACK_TOKEN':token},timeout=self.settings.auto_annotation_timeout_seconds+15)
            code=meta.get('http_status')
            if code==429:raise RelayFailure('rate_limited',self._retry_after(meta.get('retry_after')))
            if code in (401,403):raise RelayFailure('auth_failed')
            if result.returncode or not meta.get('completed'):
                raise RelayFailure('execution_unknown' if meta.get('sent') else 'cli_not_started')
            messages=[]
            for line in result.stdout.splitlines():
                try:
                    event=json.loads(line);item=event.get('item',{})
                    if event.get('type')=='item.completed' and item.get('type')=='agent_message':messages.append(item['text'])
                    if item.get('type') in {'command_execution','web_search','mcp_tool_call','file_change'}:raise RelayFailure('unexpected_tool_use')
                except json.JSONDecodeError:pass
            if not messages:raise RelayFailure('invalid_response')
            answer=_parse_json_text(messages[-1]) if schema else messages[-1]
            return {'answer':answer,'actual_model':meta.get('model'),'usage':meta.get('usage',{}),'response_id':meta.get('id'),'elapsed_seconds':time.monotonic()-t0}
        except subprocess.TimeoutExpired as e:raise RelayFailure('execution_unknown') from e
        except OSError as e:raise RelayFailure('cli_not_started') from e
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2);shutil.rmtree(home)
