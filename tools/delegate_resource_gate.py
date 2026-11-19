"""Bounded per-process worker launch pacing, subordinate to inference admission.

Used by the native delegate runner before any child conversation/inference execution.
It is NOT a server reservation, cross-host lock, or permission to replay 429s.
"""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class Limit:
    resource: str
    slots: int
    pending: int
    wait_seconds: float

    def __post_init__(self):
        if not self.resource or self.slots < 1 or self.pending < 0 or not 0 < self.wait_seconds <= 3600:
            raise ValueError('invalid worker resource limit')


class Deferred(RuntimeError):
    def __init__(self, reason, waited=0.0):
        self.reason, self.waited = reason, waited
        super().__init__(reason)


class Gate:
    def __init__(self):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.active = {}
        self.queues = {}

    @contextmanager
    def acquire(self, limit, cancelled=lambda: False, on_wait=lambda: None):
        started = time.monotonic()
        ticket = object()
        acquired = False
        with self.condition:
            queue = self.queues.setdefault(limit.resource, deque())
            # Slots remain local pacing only: the backend's admission is final.
            if self.active.get(limit.resource, 0) >= limit.slots or queue:
                if len(queue) >= limit.pending:
                    raise Deferred('capacity_pending_full')
            queue.append(ticket)
            try:
                notify_wait = True
                while True:
                    elapsed = time.monotonic() - started
                    if cancelled():
                        raise Deferred('capacity_cancelled', elapsed)
                    if elapsed >= limit.wait_seconds:
                        raise Deferred('capacity_deadline', elapsed)
                    if queue[0] is ticket and self.active.get(limit.resource, 0) < limit.slots:
                        queue.popleft()
                        self.active[limit.resource] = self.active.get(limit.resource, 0) + 1
                        acquired = True
                        break
                    if not notify_wait:
                        self.condition.wait(min(0.1, limit.wait_seconds - elapsed))
                        notify_wait = True
                        continue
                    self.lock.release()
                    try:
                        on_wait()
                    finally:
                        self.lock.acquire()
                    # Re-evaluate predicates after callback I/O before sleeping.
                    notify_wait = False
            finally:
                if not acquired:
                    queue.remove(ticket)
                    self.condition.notify_all()
        try:
            yield time.monotonic() - started
        finally:
            with self.condition:
                self.active[limit.resource] -= 1
                self.condition.notify_all()


GATE = Gate()


def resolve_limit(config, *, model, base_url):
    """Exact resolved model+endpoint match; never classify by prompt/provider label."""
    rules = config.get('resource_limits', [])
    by_resource = {}
    for rule in rules:
        shape = (rule['slots'], rule['max_pending'], rule['wait_seconds'])
        previous = by_resource.setdefault(rule['resource'], shape)
        if previous != shape:
            raise ValueError('conflicting limits for shared worker resource')
    matches = [r for r in rules if r.get('model') == model and r.get('base_url') == base_url]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError('ambiguous worker resource limits')
    r = matches[0]
    return Limit(str(r['resource']), int(r['slots']), int(r['max_pending']), float(r['wait_seconds']))


def install_child_gate(child, config):
    """Wrap the actual conversation, not its possibly timed-out observer.

    A future still running after parent timeout keeps the slot until it settles.
    The wrapper also covers output-schema correction calls on the same child.
    """
    if getattr(child, '_capacity_gate_installed', False):
        return
    limit = resolve_limit(config, model=getattr(child, 'model', None),
                          base_url=str(getattr(child, 'base_url', '')))
    if limit is None:
        return
    original = child.run_conversation
    child._capacity_gate_installed = True

    def wrapped(*args, **kwargs):
        child._delegate_capacity = {'resource': limit.resource, 'state': 'pending',
                                    'reservation': False, 'scope': 'process_local_pacing'}
        def on_wait():
            touch = getattr(child, '_touch_activity', None)
            if callable(touch):
                touch('waiting for delegated inference capacity')
        try:
            with GATE.acquire(limit, lambda: getattr(child, '_interrupt_requested', False) is True, on_wait) as waited:
                child._delegate_capacity.update(state='running', queue_seconds=round(waited, 3))
                result = original(*args, **kwargs)
                child._delegate_capacity['state'] = 'settled'
                return result
        except Deferred as exc:
            child._delegate_capacity.update(state='deferred', reason=exc.reason,
                                            queue_seconds=round(exc.waited, 3))
            return {'failed': True, 'completed': False,
                    'error': f'capacity_deferred:{exc.reason}', 'final_response': ''}
        except BaseException:
            child._delegate_capacity['state'] = 'error'
            raise
    child.run_conversation = wrapped
