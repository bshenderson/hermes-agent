"""Body-bound domain-action provenance, kept out of model-controlled arguments.

The router signs the final body after ordinary chat signing. Ingress validates
and removes this envelope before validating the enclosed chat context. Callers
must explicitly bind the accepted value across executor-thread boundaries.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
import hmac
import json
import math
import re
import time
import uuid

FIELD = 'hermes_domain_action_context'
SCHEMA = 'hermes.domain-action-turn.v1'
TTL = 900
_turn = ContextVar('domain_action_turn', default=None)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(body):
    return hashlib.sha256(_canonical({k: v for k, v in body.items() if k != FIELD})).hexdigest()


def _validate(value, *, now):
    fields = {'schema', 'subject', 'groups', 'profile', 'chat_id', 'message_id',
              'user_message_id', 'issued_at', 'expires_at', 'body_sha256'}
    if (not isinstance(value, dict) or set(value) not in (fields, fields | {'approved_send_draft'})
            or value['schema'] != SCHEMA):
        raise ValueError('invalid_domain_action_context')
    if 'approved_send_draft' in value:
        draft = value['approved_send_draft']
        if not isinstance(draft, str) or str(uuid.UUID(draft)) != draft:
            raise ValueError('invalid_approved_send_draft')
    for name in ('subject', 'profile'):
        if not isinstance(value[name], str) or not 0 < len(value[name]) <= 128:
            raise ValueError('invalid_domain_action_principal')
    groups = value['groups']
    if (not isinstance(groups, list) or len(groups) > 64
            or any(not isinstance(g, str) or not 0 < len(g) <= 128 for g in groups)
            or groups != sorted(set(groups))):
        raise ValueError('invalid_domain_action_groups')
    for name in ('chat_id', 'message_id', 'user_message_id'):
        raw = value[name]
        if not isinstance(raw, str) or str(uuid.UUID(raw)) != raw:
            raise ValueError('invalid_domain_action_origin')
    issued = value['issued_at']
    if (type(issued) not in (int, float) or not math.isfinite(issued)
            or value['expires_at'] != issued + TTL or not issued <= now < value['expires_at']):
        raise ValueError('expired_domain_action_context')
    digest = value['body_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid_domain_action_body_digest')


def explicit_send_draft(body):
    """Only a literal final authenticated user command can approve a draft."""
    messages = body.get('messages') if isinstance(body, dict) else None
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return None
    message = messages[-1]
    content = message.get('content')
    if message.get('role') != 'user' or not isinstance(content, str):
        return None
    match = re.fullmatch(r'/govcon-send ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', content)
    return match[1] if match else None


def sign(body, *, subject, groups, profile, key, now=None):
    """Only the trusted front-door identity resolver calls this signer."""
    now = time.time() if now is None else now
    if not isinstance(body, dict) or FIELD in body or not key:
        raise ValueError('caller_domain_action_authority_denied')
    native = body.get('hermes_chat_context')
    if not isinstance(native, dict) or not subject or native.get('subject_id') != subject:
        raise ValueError('domain_action_subject_mismatch')
    if not isinstance(groups, (list, tuple, set)) or any(not isinstance(g, str) for g in groups):
        raise ValueError('invalid_domain_action_groups')
    value = {'schema': SCHEMA, 'subject': subject, 'groups': sorted(set(groups)),
             'profile': profile, **{k: native.get(k) for k in ('chat_id', 'message_id', 'user_message_id')},
             'issued_at': now, 'expires_at': now + TTL, 'body_sha256': _digest(body)}
    if draft := explicit_send_draft(body):
        value['approved_send_draft'] = draft
    _validate(value, now=now)
    signature = hmac.new(key.encode(), _canonical(value), hashlib.sha256).hexdigest()
    return {**body, FIELD: {**value, 'signature': signature}}


def accept(body, *, profile, key, now=None):
    """Validate before agent construction; unsigned calls acquire no authority."""
    now = time.time() if now is None else now
    envelope = body.get(FIELD) if isinstance(body, dict) else None
    if not isinstance(envelope, dict) or not key or len(_canonical(envelope)) > 16384:
        raise ValueError('authenticated_domain_action_context_required')
    value = {k: v for k, v in envelope.items() if k != 'signature'}
    expected = hmac.new(key.encode(), _canonical(value), hashlib.sha256).hexdigest()
    if not isinstance(envelope.get('signature'), str) or not hmac.compare_digest(expected, envelope['signature']):
        raise ValueError('invalid_domain_action_signature')
    _validate(value, now=now)
    native = body.get('hermes_chat_context') or {}
    if value.get('approved_send_draft') != explicit_send_draft(body):
        raise ValueError('domain_action_send_approval_mismatch')
    if (value['profile'] != profile or value['body_sha256'] != _digest(body)
            or native.get('subject_id') != value['subject']
            or any(native.get(k) != value[k] for k in ('chat_id', 'message_id', 'user_message_id'))):
        raise ValueError('domain_action_origin_mismatch')
    return copy.deepcopy(value)


@contextmanager
def bound(value):
    """Bind only an ingress-validated value; reset on every termination path."""
    token = bind(value)
    try:
        yield
    finally:
        reset(token)


def bind(value):
    return _turn.set(copy.deepcopy(value))


def reset(token):
    _turn.reset(token)


def optional_current(*, now=None):
    """Absent means ordinary authority; expired/malformed never means absent."""
    return None if _turn.get() is None else current(now=now)


def current(*, now=None):
    now = time.time() if now is None else now
    value = _turn.get()
    _validate(value, now=now)
    return copy.deepcopy(value)
