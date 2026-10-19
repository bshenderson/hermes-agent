"""Request-local Media provenance, never a model/requester identity override."""
from contextvars import ContextVar
import copy
import time

_turn = ContextVar('homelab_media_file_turn', default=None)


def bind(turn):
    return _turn.set(copy.deepcopy(turn))


def reset(token):
    _turn.reset(token)


def current():
    value = _turn.get()
    if not isinstance(value, dict) or value.get('profile') != 'media' or time.time() >= value.get('expires_at', 0):
        raise ValueError('authenticated_fresh_media_turn_required')
    return copy.deepcopy(value)
