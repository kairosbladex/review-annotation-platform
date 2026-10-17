import importlib.util
import logging
from pathlib import Path

import pytest


@pytest.fixture
def privacy_logging():
    original = logging.getLogRecordFactory()
    source = Path(__file__).resolve().parents[2] / 'deploy/algorithm-privacy/sitecustomize.py'
    if source.exists():
        spec = importlib.util.spec_from_file_location('vision_privacy_test', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    yield
    logging.setLogRecordFactory(original)


def test_presigned_media_queries_and_url_credentials_never_reach_log_records(privacy_logging):
    logger = logging.getLogger('algorithm-download-test')
    record = logger.makeRecord(logger.name, logging.INFO, 'source.py', 1,
                              'download %s', ('https://identity:synthetic-password@media.invalid/bucket/input.mp4?X-Amz-Signature=synthetic-signature',), None)
    text = logging.Formatter('%(message)s').format(record)
    assert 'X-Amz-Signature' not in text
    assert 'synthetic-password' not in text
    assert 'media.invalid/bucket/input.mp4' in text


def test_exception_and_named_credentials_are_redacted_too(privacy_logging):
    import sys
    logger = logging.getLogger('algorithm-failure-test')
    try:
        raise RuntimeError('failed https://media.invalid/video.mp4?token=synthetic-token')
    except RuntimeError:
        record = logger.makeRecord(logger.name, logging.ERROR, 'source.py', 1,
                                  'API_KEY=synthetic-key Authorization: Bearer synthetic-bearer', (), sys.exc_info())
    text = logging.Formatter('%(message)s').format(record)
    assert all(value not in text for value in ('synthetic-key', 'synthetic-bearer', 'synthetic-token'))
    assert 'RuntimeError' in text


@pytest.mark.parametrize('target', [
    '/api/v1/retrieval/tasks?X-Amz-Signature=synthetic-signature&token=synthetic-token',
    'https://identity:synthetic-password@media.invalid/result.png?X-Amz-Signature=synthetic-signature',
])
def test_uvicorn_access_formatter_keeps_structured_arguments_and_redacts_queries(privacy_logging, target):
    from uvicorn.logging import AccessFormatter

    logger = logging.getLogger('uvicorn.access')
    record = logger.makeRecord(logger.name, logging.INFO, 'source.py', 1,
                              '%s - "%s %s HTTP/%s" %d',
                              ('127.0.0.1:1234', 'GET', target, '1.1', 200), None)
    text = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s',
                           use_colors=False).format(record)
    assert 'GET ' in text and 'HTTP/1.1' in text and '200 OK' in text
    assert all(value not in text for value in (
        'X-Amz-Signature', 'synthetic-signature', 'synthetic-token', 'synthetic-password',
    ))
