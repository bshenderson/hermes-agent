"""Runtime-owned automatic-retention policy, scoped to one admitted turn.

This is not authorization and is never derived from text or caller author metadata.
Explicit memory tools and conversation history are unaffected.
"""
from contextlib import contextmanager
from contextvars import ContextVar


_automatic_input = ContextVar("memory_automatic_input", default=False)


def automatic_memory_input() -> bool:
    return _automatic_input.get()


@contextmanager
def memory_input_scope(*, automatic: bool):
    token = _automatic_input.set(automatic)
    try:
        yield
    finally:
        _automatic_input.reset(token)
