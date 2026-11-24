"""One monotonic budget inherited by source workers and subprocess groups."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import time

_END: ContextVar[float | None] = ContextVar('secret_source_deadline', default=None)


def bounded() -> bool:
    from hermes_cli.startup_budget import active
    return _END.get() is not None or active()


def remaining(cap: float) -> float:
    from hermes_cli.startup_budget import remaining as startup_remaining
    cap = startup_remaining(cap)
    end = _END.get()
    value = cap if end is None else min(cap, end - time.monotonic())
    if value <= 0:
        raise RuntimeError('secret source deadline exhausted; no retry')
    return value


@contextmanager
def secret_budget(seconds: float):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('secret budget must be finite and positive')
    parent = _END.get()
    end = time.monotonic() + seconds
    token = _END.set(min(end, parent) if parent is not None else end)
    try:
        yield
    finally:
        _END.reset(token)
