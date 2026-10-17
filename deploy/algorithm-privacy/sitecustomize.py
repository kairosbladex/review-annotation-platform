"""Redact expiring media credentials before the isolated algorithm logs them.

Loaded by Python through the deployment's read-only PYTHONPATH mount. This
does not change model inputs, HTTP requests, detection results or sampling.
"""
import logging
import re
import traceback
from urllib.parse import urlsplit, urlunsplit

_url = re.compile(r'https?://[^\s\x00-\x1f<>\"\']+')
_bearer = re.compile(r'(?i)\bBearer\s+[a-zA-Z0-9_.~+/=-]+')
_credential = re.compile(r'''(?ix)
    \b((?:api|access|secret)[_-]?key|password|token|authorization)\b
    (\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;]+)
''')


def _safe_url(match):
    try:
        parts = urlsplit(match.group(0))
        host = parts.hostname or ''
        if ':' in host:
            host = '[' + host + ']'
        if parts.port:
            host += ':' + str(parts.port)
        return urlunsplit((parts.scheme, host, parts.path, '', ''))
    except ValueError:
        return '[redacted URL]'


def redact(text):
    text = _url.sub(_safe_url, str(text))
    text = _bearer.sub('Bearer [redacted]', text)
    return _credential.sub(r'\1\2[redacted]', text)


_previous_factory = logging.getLogRecordFactory()


def _private_record(*args, **kwargs):
    record = _previous_factory(*args, **kwargs)
    if record.name == 'uvicorn.access' and isinstance(record.args, tuple) and len(record.args) == 5:
        # AccessFormatter reads these five arguments after LogRecord creation.
        # Keep their structure while removing credentials from the request target.
        client, method, target, version, status = record.args
        target = str(target).split('?', 1)[0].split('#', 1)[0]
        record.msg = redact(record.msg)
        record.args = (redact(client), redact(method), redact(target), redact(version), status)
    else:
        try:
            record.msg = redact(record.getMessage())
        except (TypeError, ValueError):
            record.msg = '[unformattable log message]'
        record.args = ()
    if record.exc_info:
        record.exc_text = redact(''.join(traceback.format_exception(*record.exc_info)))
        record.exc_info = None
    if record.stack_info:
        record.stack_info = redact(record.stack_info)
    return record


logging.setLogRecordFactory(_private_record)
