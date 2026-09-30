"""Transport protocol checks use an in-memory HTTP endpoint, never a live account."""
from dataclasses import replace
import httpx
import pytest
from app.config import Settings
from app.adapters.recheck_relay import RecheckRelay,RelayFailure
HTTP_CLIENT=httpx.Client

def relay(monkeypatch,handler):
    client=HTTP_CLIENT
    monkeypatch.setattr(httpx,'Client',lambda **kw:client(transport=httpx.MockTransport(handler),**kw))
    return RecheckRelay(replace(Settings(),auto_annotation_base_url='https://configured.invalid',auto_annotation_api_key='isolated-placeholder',auto_annotation_use_codex_cli=False))

def test_usage_cache_60_seconds_and_force_refresh(monkeypatch):
    calls=[]
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200,json={'remaining':30,'status':'active','isValid':True,'unit':'USD'})
    r=relay(monkeypatch,handler)
    assert r.usage()['remaining']==30
    r.usage();assert len(calls)==1
    r.usage(force=True);assert len(calls)==2
    r._usage_at-=61;r.usage();assert len(calls)==3

@pytest.mark.parametrize('status,code',[(401,'usage_auth_failed'),(503,'usage_unavailable')])
def test_usage_http_failure_is_not_valid_quota(monkeypatch,status,code):
    r=relay(monkeypatch,lambda request:httpx.Response(status))
    with pytest.raises(RelayFailure,match=code):r.usage()

@pytest.mark.parametrize('data',[{'remaining':30,'status':'active','isValid':False,'unit':'USD'}, {'remaining':30,'status':'active','isValid':True,'unit':'credits'}])
def test_inactive_account_or_unknown_currency_blocks_usage(monkeypatch,data):
    r=relay(monkeypatch,lambda request:httpx.Response(200,json=data))
    with pytest.raises(RelayFailure):r.usage()

def test_429_wait_time_and_unknown_timeout_are_not_hidden_retries(monkeypatch):
    calls=[]
    def handler(request):calls.append(request);return httpx.Response(429,headers={'Retry-After':'11'})
    r=relay(monkeypatch,handler)
    with pytest.raises(RelayFailure) as e:r.request(model='test',prompt='test')
    assert e.value.code=='rate_limited' and e.value.retry_after==11 and len(calls)==1
    def timeout(request):calls.append(request);raise httpx.ReadTimeout('injected',request=request)
    r=relay(monkeypatch,timeout)
    with pytest.raises(RelayFailure,match='execution_unknown'):r.request(model='test',prompt='test')
    assert len(calls)==2
