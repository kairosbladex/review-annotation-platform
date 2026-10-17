import json
from dataclasses import replace

import httpx
import pytest

from app.adapters.retrieval import RetrievalAdapter, RetrievalServiceError
from app.config import Settings


def test_adapter_uses_fixed_algorithm_paths_and_preserves_sampling_and_threshold():
    requests = []
    def remote(request):
        requests.append(request)
        if request.url.path.endswith('/text-search'):
            return httpx.Response(200, json={'code': 200, 'data': {'job_id': '0123456789abcdef'}})
        if request.url.path.endswith('/stop'):
            return httpx.Response(200, json={'code': 200, 'data': {'status': 'stopping'}})
        return httpx.Response(200, json={'code': 200, 'data': {'status': 'completed', 'results': []}})
    adapter = RetrievalAdapter(replace(Settings(), retrieval_service_base_url='https://algorithm.invalid'),
                               transport=httpx.MockTransport(remote))
    payload = {'text': 'person', 'file_urls': ['https://media.invalid/input.mp4'],
               'confidence': 0.2, 'sample_interval_seconds': 3}
    identifier = adapter.submit(payload)
    assert json.loads(requests[0].content) == payload
    assert requests[0].url.path == '/api/v1/detect/batch/text-search'
    assert adapter.get(identifier)['status'] == 'completed'
    assert adapter.cancel(identifier)['status'] == 'stopping'
    assert [request.method for request in requests] == ['POST', 'GET', 'POST']


@pytest.mark.parametrize('body', [
    {'code': 200, 'success': False, 'data': {'job_id': '0123456789abcdef'}},
    {'code': 500, 'data': {'job_id': '0123456789abcdef'}},
    {'code': 200, 'data': {'job_id': '../unexpected/path'}},
    {'code': 200, 'data': []},
])
def test_http_success_does_not_hide_invalid_or_failed_submission(body):
    adapter = RetrievalAdapter(replace(Settings(), retrieval_service_base_url='https://algorithm.invalid'),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    with pytest.raises(RetrievalServiceError):
        adapter.submit({'text': 'person', 'file_urls': ['https://media.invalid/input.mp4']})


def test_untrusted_job_id_cannot_change_remote_resource():
    requests = []
    adapter = RetrievalAdapter(replace(Settings(), retrieval_service_base_url='https://algorithm.invalid'),
        transport=httpx.MockTransport(lambda request: requests.append(request)))
    with pytest.raises(RetrievalServiceError):
        adapter.get('../other-project')
    assert requests == []
