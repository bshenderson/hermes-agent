"""Opt-in synthetic v2 observations; no execution, retry or validation authority.

Uses the selection campaign's ONE aggregate budget. All mutable lifetime flags
share that budget lock so copied provider contexts cannot outlive their request.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
import threading
import time
import uuid
from typing import Any

from agent.selection_capture import SelectionCapture

_call: ContextVar[Any] = ContextVar('response_capture_selection', default=None)
_attempt: ContextVar[Any] = ContextVar('response_capture_attempt', default=None)
_LIMIT = 1048576


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _digest(raw):
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def _text(value):
    if value is not None and (not isinstance(value, str) or len(value) > _LIMIT):
        raise ValueError('unsupported synthetic response shape')
    return value


def _sdk(value, delta=False):
    choices = _field(value, 'choices')
    if not choices:
        return {'choices': 0}
    choice = choices[0]
    msg = _field(choice, 'delta' if delta else 'message')
    data: dict[str, Any] = {k: _text(_field(msg, k)) for k in ('content', 'reasoning', 'reasoning_content')}
    data['finish_reason'] = _text(_field(choice, 'finish_reason'))
    details = _field(msg, 'reasoning_details')
    if details is not None:
        if not isinstance(details, list) or len(details) > 128:
            raise ValueError('unsupported synthetic details')
        data['reasoning_details'] = []
        for item in details:
            if not isinstance(item, dict):
                raise ValueError('unsupported synthetic detail')
            data['reasoning_details'].append({k: _text(item[k]) for k in ('summary', 'content', 'text') if k in item})
    return data


class ResponseCapture(SelectionCapture):
    def __init__(self, root, expires_at, *, identity, campaign_id):
        super().__init__(root, expires_at)
        self.identity = dict(identity)
        self.campaign_id = campaign_id
        self.closed_requests = 0
        self._lock = threading.RLock()

    def _status(self):
        # Bounded metadata, never model strings or exception text.
        payload = json.dumps({**self.status(), 'closed_requests': self.closed_requests,
            'attribution': 'incomplete' if self.omitted or self.errors else 'unadjudicated',
            'native_settlement': 'unknown', 'authorizes_execution': False}).encode()
        fd = os.open(self.root / 'status.json', os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def payload(self, request, call, attempt, kind, data):
        return (json.dumps(dict(classification='private-synthetic-response-observation',
            campaign_id=self.campaign_id, identity=self.identity, request_id=request.id,
            selection_call_id=call.id, source=call.source,
            provider_attempt_id=attempt.id if attempt else None,
            transport_ordinal=attempt.ordinal if attempt else None,
            kind=kind, data=data, authorizes_execution=False), ensure_ascii=True) + '\n').encode()

    def emit(self, request, call, attempt, kind, build, *, charged=0):
        with self._lock:
            self.events += 1
            if kind == 'transport_request' and attempt is not None:
                attempt.ordinal += 1
            try:
                if (self.events > 128 or time.monotonic() >= self.deadline
                        or not request.active or not call.active
                        or (attempt and not attempt.active and kind in ('transport_request', 'sdk_delta', 'sdk_delta_batch'))):
                    self.omitted += 1
                    return
                payload = self.payload(request, call, attempt, kind, build())
                if charged and len(payload) != charged:
                    raise ValueError('delta batch reservation mismatch')
                if self.bytes + (0 if charged else len(payload)) > _LIMIT:
                    self.omitted += 1
                    return
                self.bytes += 0 if charged else len(payload)  # Nonrefundable, including buffered deltas.
                fd = os.open(self.root / f'event-{self.events:04d}.json',
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                self.errors += 1
            finally:
                try:
                    self._status()
                except Exception:
                    self.errors += 1

    @contextmanager
    def request_scope(self):
        try:
            request = RequestHandle(self)
        except Exception:
            with self._lock:
                self.errors += 1
            yield None
            return
        try:
            yield request
        finally:
            with self._lock:
                request.active = False
                self.closed_requests += 1
                try:
                    self._status()
                except Exception:
                    self.errors += 1


class RequestHandle:
    def __init__(self, campaign: ResponseCapture):
        self.campaign = campaign
        self.id, self.active = uuid.uuid4().hex, True

    def record(self, source, response):
        call = _call.get()
        if call is None or call.request is not self:
            return
        def build():
            if _digest(source.encode()) != call.source:
                raise ValueError('selection observation chunk mismatch')
            result = {'response': _text(response), **_digest(source.encode())}
            try:
                value = json.loads(response)
                excerpts = value.get('excerpts') if isinstance(value, dict) else None
                if isinstance(excerpts, list) and len(excerpts) <= 128:
                    result['occurrences'] = [source.count(q) if isinstance(q, str) and q else None for q in excerpts]
            except (ValueError, TypeError, RecursionError):
                pass
            return result
        self.campaign.emit(self, call, call.last_attempt, 'selection', build)


class SelectionCall:
    def __init__(self, request, source):
        self.request, self.id, self.active = request, uuid.uuid4().hex, True
        self.source = _digest(source.encode())
        self.last_attempt = None


@contextmanager
def selection_scope(source):
    from agent.selection_capture import _current
    request = _current.get()
    if not isinstance(request, RequestHandle):
        yield
        return
    try:
        call = SelectionCall(request, source)
    except Exception:
        with request.campaign._lock:
            request.campaign.errors += 1
        token = _call.set(None)
        try:
            yield
        finally:
            _call.reset(token)
        return
    token = _call.set(call)
    try:
        yield
    finally:
        with request.campaign._lock:
            call.active = False
        _call.reset(token)


def _emit(kind, build):
    call = _call.get()
    if call is not None and (kind != 'sdk_delta' or _attempt.get() is not None):
        call.request.campaign.emit(call.request, call, _attempt.get() or call.last_attempt, kind, build)


def observe_sdk(value, *, delta=False):
    if delta:
        attempt = _attempt.get()
        if attempt is not None:
            attempt.add_delta(value)
        return
    _emit('sdk_delta' if delta else 'summary_response', lambda: _sdk(value, delta))


def observe_error(error):
    _emit('extraction_error', lambda: {'error_class': type(error).__name__})


def extraction_observer():
    if _call.get() is None:
        return None
    def observe(selected, branch, truncated=False):
        _emit('extraction', lambda: dict(selected=_text(selected), branch=branch, truncated=truncated))
    return observe


class ProviderAttempt:
    def __init__(self, call):
        self.call, self.id, self.active, self.ordinal = call, uuid.uuid4().hex, True, 0
        self.deltas, self.delta_charge = [], 0

    def flush(self):
        c = self.call.request.campaign
        with c._lock:
            if self.deltas:
                data, charge = self.deltas, self.delta_charge
                self.deltas, self.delta_charge = [], 0
                c.emit(self.call.request, self.call, self, 'sdk_delta_batch',
                       lambda: {'deltas': data}, charged=charge)

    def add_delta(self, value):
        c, call = self.call.request.campaign, self.call
        with c._lock:
            try:
                if (not self.active or not call.active or not call.request.active
                        or time.monotonic() >= c.deadline or c.events >= 128):
                    c.omitted += 1
                    return
                data = _sdk(value, delta=True)
                size = len(json.dumps(data, ensure_ascii=True).encode())
                if self.deltas and self.delta_charge + size + 2 > 16384:
                    self.flush()
                overhead = (2 if self.deltas else len(c.payload(
                    call.request, call, self, 'sdk_delta_batch', {'deltas': []})))
                # Reserve before retaining raw fields; never refund failed/expired batches.
                if c.bytes + size + overhead > _LIMIT:
                    c.omitted += 1
                    return
                c.bytes += size + overhead
                self.delta_charge += size + overhead
                self.deltas.append(data)
                if len(self.deltas) >= 256 or self.delta_charge >= 16384:
                    self.flush()
            except Exception:
                c.errors += 1

    def hook(self, base_url):
        expected = (base_url.scheme, base_url.host, base_url.port, base_url.path.rstrip('/') + '/chat/completions')
        def observe(request):
            try:
                self.flush()  # A retry/fallback request gets its own transport ordinal.
                def build():
                    if (request.url.scheme, request.url.host, request.url.port, request.url.path) != expected:
                        raise ValueError('unsupported observation destination')
                    raw = request.content  # Already-buffered only. Never read()/consume a stream.
                    if len(raw) > 8388608:
                        raise ValueError('unsupported observation body size')
                    value = json.loads(raw)
                    messages = value.get('messages')
                    if not isinstance(messages, list) or len(messages) > 128:
                        raise ValueError('unsupported message shape')
                    result = {**_digest(raw), 'messages': []}
                    for msg in messages:
                        role, content = msg.get('role'), msg.get('content')
                        if role not in ('user', 'assistant', 'system', 'tool') or not isinstance(content, str):
                            raise ValueError('unsupported message content')
                        result['messages'].append(dict(role=role, **_digest(content.encode())))
                    for k in ('model', 'max_tokens', 'max_completion_tokens', 'stream'):
                        v = value.get(k)
                        if v is None or type(v) in (int, bool) or isinstance(v, str) and len(v) <= 256:
                            result[k] = v
                    fmt = value.get('response_format')
                    result['response_format'] = _digest(json.dumps(fmt, sort_keys=True).encode())
                    think = (value.get('chat_template_kwargs') or {}).get('enable_thinking')
                    if type(think) is bool:
                        result['enable_thinking'] = think
                    return result
                c = self.call
                c.request.campaign.emit(c.request, c, self, 'transport_request', build)
            except Exception:
                # Defensive: observation must not become an SDK dispatch failure.
                with self.call.request.campaign._lock:
                    self.call.request.campaign.errors += 1
        return observe


@contextmanager
def provider_scope(task, client):
    call = _call.get()
    if call is None or task != 'compression':
        token = _attempt.set(None)
        try:
            yield None
        finally:
            _attempt.reset(token)
        return
    try:
        attempt = ProviderAttempt(call)
        hook = attempt.hook(client.base_url)
    except Exception:
        with call.request.campaign._lock:
            call.request.campaign.errors += 1
        yield None
        return
    with call.request.campaign._lock:
        call.last_attempt = attempt
    token = _attempt.set(attempt)
    try:
        yield hook
    finally:
        # Keep the attempt's identity for the summary/selection return leg, but
        # an escaped HTTP hook must not emit after this attempt has unwound.
        with call.request.campaign._lock:
            attempt.flush()
            attempt.active = False
        _attempt.reset(token)


def unsupported_transport(task):
    if task == 'compression':
        _emit('unsupported_transport', lambda: {'attribution': 'incomplete'})
