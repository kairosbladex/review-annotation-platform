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
                              'download %s', ('https://REDACTED:REDACTED@media.invalid/bucket/input.mp4?X-Amz-Signature=synthetic-signature',), None)
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
