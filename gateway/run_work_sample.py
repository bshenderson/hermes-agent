"""Bounded publication provenance; observations never grant lifecycle authority."""
from datetime import UTC, datetime
import os
import math
import asyncio
from contextvars import copy_context
from functools import partial
import threading
import time


_lock = threading.RLock()
_sequence = 0


def bind_work_sample_owner(runner):
    """Bind at startup, before adapters or callbacks can publish."""
    runner._work_sample_loop = asyncio.get_running_loop()
    runner._work_sample_context = copy_context()


def _best_effort_count(read):
    try:
        return read(), True
    except Exception:  # health: allow BLE001 -- preserve legacy fallback; expose availability, never private exception text
        # Preserve the legacy aggregate fallback while marking it unobserved.
        return 0, False


def project_work_sample(value, *, pid, start_time, total):
    """Allowlisted diagnostic projection; malformed provenance is unavailable."""
    if not isinstance(value, dict):
        return None
    integers = ('schema', 'pid', 'sequence', 'total')
    if any(type(value.get(k)) is not int or value[k] < 0 for k in integers):
        return None
    if value['schema'] != 1 or value['pid'] != pid or value['total'] != total or value['sequence'] < 1:
        return None
    if type(start_time) not in (int, float) or not math.isfinite(start_time) or start_time <= 0:
        return None
    if type(value.get('start_time')) not in (int, float) or value['start_time'] != start_time:
        return None
    kinds = ('chat', 'managed', 'cron', 'api', 'deferred')
    components = value.get('components')
    if not isinstance(components, dict) or set(components) != set(kinds):
        return None
    if any(type(components[k]) is not int or components[k] < 0 for k in kinds):
        return None
    if sum(components.values()) != total:
        return None
    availability = value.get('availability')
    if not isinstance(availability, dict) or set(availability) != set(kinds):
        return None
    if any(type(availability[k]) is not bool for k in kinds):
        return None
    for key in ('monotonic_start', 'monotonic_end'):
        number = value.get(key)
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            return None
    if value['monotonic_end'] < value['monotonic_start']:
        return None
    stamp = value.get('observed_at')
    if not isinstance(stamp, str) or len(stamp) > 64:
        return None
    try:
        if datetime.fromisoformat(stamp).tzinfo is None:
            return None
    except ValueError:
        return None
    keys = (*integers, 'start_time', 'observed_at', 'monotonic_start', 'monotonic_end')
    return {**{k: value[k] for k in keys}, 'components': {k: components[k] for k in kinds},
            'availability': {k: availability[k] for k in kinds}}


def publish_work_sample(runner, **fields):
    """Publish one interval's components and their sum, without a second recount."""
    from gateway.run import _write_runtime_status_quiet
    from gateway.run_runtime import managed_turn_count
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    owner_loop = getattr(runner, '_work_sample_loop', None)
    if owner_loop is None and current_loop is not None:
        bind_work_sample_owner(runner)
        owner_loop = current_loop
    if owner_loop is not None and current_loop is not owner_loop:
        if not owner_loop.is_running():
            # Teardown must not label an off-loop recount as an owner observation.
            _write_runtime_status_quiet(**fields, active_agents=runner._active_work_count())
            return
        try:
            owner_loop.call_soon_threadsafe(
                partial(publish_work_sample, runner, **fields),
                context=runner._work_sample_context.copy())
        except RuntimeError:
            # The owner can close between the liveness read and dispatch.
            _write_runtime_status_quiet(**fields, active_agents=runner._active_work_count())
        return
    from gateway.status import _build_pid_record

    global _sequence
    with _lock:
        started = time.monotonic()
        cron, cron_available = _best_effort_count(runner._running_cron_job_count)
        api, api_available = _best_effort_count(partial(runner._api_server_hook, 'active_agent_work_count'))
        components = {
            'chat': runner._running_agent_count(),
            'managed': managed_turn_count(runner),
            'cron': cron,
            'api': api,
            'deferred': runner._active_deferred_agent_worker_count(),
        }
        total = sum(components.values())
        _sequence += 1
        sample = {
            'schema': 1, 'pid': os.getpid(),
            'start_time': _build_pid_record()['start_time'],
            'sequence': _sequence,
            'observed_at': datetime.now(UTC).isoformat(),
            'monotonic_start': started, 'monotonic_end': time.monotonic(),
            'components': components, 'total': total,
            'availability': dict(chat=True, managed=True, cron=cron_available,
                                 api=api_available, deferred=True),
        }
        _write_runtime_status_quiet(active_agents=total, work_sample=sample, **fields)
