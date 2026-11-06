"""Operator-armed, synthetic-only selector observation. Never dispatches or retries.

The fixed manifest is local administrative authority, not a client field. A
separate finite supervisor owns request budgets and exact opt-in rollback.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import threading
import time

from agent.owned_chat_history import OwnedChatHistory
from hermes_constants import get_hermes_home

_LOG = logging.getLogger(__name__)
_current = ContextVar('selection_capture', default=None)
_lock = threading.Lock()
_campaigns = {}


class SelectionCapture:
    """One process-incarnation, nonrefundable retention budget per campaign."""

    def __init__(self, root, expires_at):
        self.root = root
        root.mkdir(mode=0o700, exist_ok=False)
        self.deadline = time.monotonic() + max(0, expires_at - time.time())
        self.events = self.bytes = self.omitted = self.errors = 0
        self._lock = threading.Lock()

    def record(self, source, response):
        with self._lock:
            self.events += 1
            try:
                if self.events > 128 or time.monotonic() >= self.deadline:
                    self.omitted += 1
                    return
                if not isinstance(source, str) or not isinstance(response, str) or len(response) > 1048576:
                    self.omitted += 1
                    return
                occurrences = None
                try:
                    value = json.loads(response)
                    excerpts = value.get('excerpts') if isinstance(value, dict) else None
                    if isinstance(excerpts, list) and len(excerpts) <= 128:
                        occurrences = [source.count(q) if isinstance(q, str) and q else None for q in excerpts]
                except (ValueError, TypeError, RecursionError):
                    pass
                raw_source = source.encode()
                payload = (json.dumps({
                    'classification': 'private-synthetic-selector-observation',
                    'source_sha256': hashlib.sha256(raw_source).hexdigest(),
                    'source_bytes': len(raw_source), 'response': response,
                    'occurrences': occurrences, 'authorizes_execution': False,
                }, ensure_ascii=True) + '\n').encode()
                if self.bytes + len(payload) > 1048576:
                    self.omitted += 1
                    return
                self.bytes += len(payload)
                fd = os.open(self.root / f'selection-{self.events:04d}.json',
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                self.errors += 1
                _LOG.warning('Synthetic selection capture persistence failed', exc_info=True)
            finally:
                # Bounded metadata is separate from the 1MiB raw-response budget.
                try:
                    payload = json.dumps(self.status()).encode()
                    fd = os.open(self.root / 'status.json',
                                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                except Exception:
                    self.errors += 1
                    _LOG.warning('Synthetic selection capture status unavailable', exc_info=True)

    def status(self):
        return dict(events=self.events, bytes=self.bytes, omitted=self.omitted, errors=self.errors)


def _manifest(root):
    path = root / 'selection-capture.json'
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_uid != os.getuid() or before.st_nlink != 1 or before.st_size > 4096):
            raise ValueError('unsafe selection capture manifest')
        raw = stream.read(4097)
        after = os.fstat(stream.fileno())
        if len(raw) > 4096 or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('selection capture manifest changed')
    value = json.loads(raw)
    fields = {'campaign_id', 'profile', 'subject', 'chat', 'root_session_id',
              'armed_at', 'expires_at', 'synthetic_only'}
    v2 = (isinstance(value, dict) and set(value) == fields | {'version', 'capture_kind'}
          and type(value['version']) is int and value['version'] == 2
          and value['capture_kind'] == 'response-boundary-v1')
    if not isinstance(value, dict) or (set(value) != fields and not v2) or value['synthetic_only'] is not True:
        raise ValueError('invalid selection capture manifest')
    if not all(isinstance(value[k], str) and value[k] for k in fields - {'armed_at', 'expires_at', 'synthetic_only'}):
        raise ValueError('invalid selection capture identity')
    if re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', value['campaign_id']) is None:
        raise ValueError('invalid selection capture campaign')
    now = time.time()
    if not (type(value['armed_at']) in (float, int) and type(value['expires_at']) in (float, int)
            and value['armed_at'] <= now < value['expires_at'] <= value['armed_at'] + 1800):
        return None
    return value


def for_owned_request(profile, owned_history):
    """Resolve only server-minted signed identity; ignore all client diagnostics."""
    if not isinstance(owned_history, OwnedChatHistory) or profile != owned_history.profile:
        return None
    root = get_hermes_home() / 'runtime'
    try:
        value = _manifest(root)
        if value is None or any(value[key] != actual for key, actual in (
                ('profile', profile), ('subject', owned_history.subject),
                ('chat', owned_history.chat), ('root_session_id', owned_history.root_session_id))):
            return None
        identity = (str(root), json.dumps(value, sort_keys=True))
        with _lock:
            if identity not in _campaigns:
                # Do not reopen a previous incarnation's directory or reset its budget.
                parent = root / 'selection-captures'
                parent.mkdir(mode=0o700, exist_ok=True)
                info = parent.lstat()
                if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
                        or info.st_uid != os.getuid()):
                    raise ValueError('unsafe selection capture root')
                if value.get('version') == 2:
                    from agent.response_capture import ResponseCapture
                    _campaigns[identity] = ResponseCapture(parent / value['campaign_id'], value['expires_at'],
                        identity={k: value[k] for k in ('profile', 'subject', 'chat', 'root_session_id')},
                        campaign_id=value['campaign_id'])
                else:
                    _campaigns[identity] = SelectionCapture(parent / value['campaign_id'], value['expires_at'])
            return _campaigns[identity]
    except FileNotFoundError:
        return None
    except Exception:
        _LOG.warning('Synthetic selection capture disabled: invalid or unavailable local campaign', exc_info=True)
        return None


@contextmanager
def capture_scope(capture):
    if hasattr(capture, 'request_scope'):
        with capture.request_scope() as request:
            token = _current.set(request)
            try:
                yield
            finally:
                _current.reset(token)
        return
    token = _current.set(capture)
    try:
        yield
    finally:
        _current.reset(token)


def observe_selection(source, response):
    capture = _current.get()
    if capture is not None:
        try:
            capture.record(source, response)
        except Exception:
            # The original validator remains the sole result/exception authority.
            _LOG.warning('Synthetic selection observer failed', exc_info=True)
