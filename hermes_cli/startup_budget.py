"""Finite POSIX client startup, from configuration through gateway attachment.

Only a fresh interactive client arms this budget. It never restarts or stops the
shared gateway. SIGALRM unwinds Python finally blocks so owned subprocess calls
can reap children; the existing gateway watchdog remains its own lifecycle owner.
"""
import math
import signal
import sys
import threading
import time

_end: float | None = None
_started = 0.0
_stage = 'configuration'
_previous = None
_attached = False
_STAGES = frozenset({'configuration', 'config', 'secrets', 'plugins', 'gateway', 'model', 'memory', 'mcp'})


def _expired(_signal, _frame):
    if _attached:
        ready()
        return
    elapsed = time.monotonic() - _started
    print(f'Hermes startup refused: reason=deadline stage={_stage} elapsed={elapsed:.2f}s; '
          'use the model-free local recovery runbook; no provider fallback.', file=sys.stderr)
    raise SystemExit(78)


def arm(seconds: float = 60.0) -> bool:
    global _end, _started, _previous, _attached
    if not math.isfinite(seconds) or not 0 < seconds <= 300:
        raise ValueError('startup budget must be finite, positive and at most 300 seconds')
    if _end is not None:
        return True
    if not hasattr(signal, 'setitimer') or threading.current_thread() is not threading.main_thread():
        return False
    # Never steal another timer's ownership.
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('startup timer already owned')
    _started = time.monotonic()
    _attached = False
    _end = _started + seconds
    _previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return True


def stage(name: str) -> None:
    global _stage
    if name not in _STAGES:
        raise ValueError('unknown startup stage')
    _stage = name


def remaining(cap: float) -> float:
    if _end is None or _attached:
        return cap
    return max(0.0, min(cap, _end - time.monotonic()))


def active() -> bool:
    return _end is not None and not _attached


def attached() -> None:
    global _attached
    _attached = True
    if threading.current_thread() is threading.main_thread():
        ready()


def ready() -> None:
    global _end
    if _end is not None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, _previous)
        _end = None


def arm_for_argv(argv: list[str]) -> bool:
    # Profile options were consumed by the real profile parser. Maintenance,
    # server and info commands keep their existing lifecycle/deadline owners.
    if '--help' in argv or '-h' in argv or '--version' in argv:
        return False
    from hermes_cli._parser import command_argv
    command = command_argv(argv)
    if not command or command[0] == 'chat':
        return arm()
    return False
