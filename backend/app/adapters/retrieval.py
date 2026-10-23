from __future__ import annotations

import re

import httpx


class RetrievalServiceError(RuntimeError):
    pass


class RetrievalAdapter:
    """Fixed algorithm API paths; credentials only come from service settings."""
    def __init__(self, settings, *, transport=None):
        self.base_url = settings.retrieval_service_base_url.rstrip('/')
        self.api_key = settings.retrieval_service_api_key
        self.verify_tls = settings.retrieval_service_verify_tls
        self.timeout = settings.retrieval_timeout_seconds
        self.transport = transport

    def _request(self, method, path, **kwargs):
        if not self.base_url:
            raise RetrievalServiceError('service_not_configured')
        headers = {'X-API-Key': self.api_key} if self.api_key else {}
        with httpx.Client(timeout=self.timeout, verify=self.verify_tls, trust_env=False,
                          follow_redirects=False, transport=self.transport) as client:
            response = client.request(method, self.base_url + '/api/v1/detect/batch' + path,
                                      headers=headers, **kwargs)
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError:
            raise RetrievalServiceError('invalid_response') from None
        if not isinstance(body, dict) or body.get('code') != 200 or body.get('success') is False or not isinstance(body.get('data'), dict):
            raise RetrievalServiceError('invalid_response')
        return body['data']

    @staticmethod
    def _job_id(remote_id):
        if not isinstance(remote_id, str) or not re.fullmatch(r'[0-9a-fA-F]{16,64}', remote_id):
            raise RetrievalServiceError('invalid_job_id')
        return remote_id

    def submit(self, payload):
        path = '/visual-search' if payload.get('prompt_image_urls') else '/text-search'
        data = self._request('POST', path, json=payload)
        return self._job_id(data.get('job_id'))

    def get(self, remote_id):
        return self._request('GET', '/jobs/' + self._job_id(remote_id))

    def cancel(self, remote_id):
        data = self._request('POST', '/jobs/' + self._job_id(remote_id) + '/stop')
        if data.get('status') not in {'stopping', 'cancelled'}:
            raise RetrievalServiceError('invalid_response')
        return data
