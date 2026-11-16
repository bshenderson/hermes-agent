"""Notify bound gateway loops after process-local work ownership changes.

Producers never own gateway instances or status files. Weak subscriptions cannot
keep a stopped runner alive, and queued notifications cannot revive its loop.
"""
import logging
import threading
import weakref

_lock = threading.RLock()
_owners = weakref.WeakKeyDictionary()
logger = logging.getLogger(__name__)


def bind_owner(runner, loop, context):
    with _lock:
        _owners[runner] = (loop, context)


def _publish(reference, loop):
    runner = reference()
    if runner is None or not runner._running:
        return
    with _lock:
        binding = _owners.get(runner)
    if binding is None or binding[0] is not loop or not loop.is_running():
        return
    try:
        runner._persist_active_agents()
    except Exception:
        # Publication is diagnostic; failure must not undo authoritative retirement.
        logger.warning('Work ownership publication failed', exc_info=True)


def notify_work_changed():
    with _lock:
        owners = [(weakref.ref(runner), *binding) for runner, binding in _owners.items()]
    for reference, loop, context in owners:
        if not loop.is_running():
            continue
        try:
            loop.call_soon_threadsafe(_publish, reference, loop, context=context.copy())
        except RuntimeError:
            # Closing between observation and dispatch is ordinary shutdown.
            continue
