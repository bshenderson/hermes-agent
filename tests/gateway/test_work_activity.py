"""Notification lifetime and gateway profile-context ownership."""
import asyncio
from contextvars import ContextVar
import gc
import weakref
import pytest
from gateway import status
from gateway.run_work_sample import bind_work_sample_owner
from gateway.work_activity import notify_work_changed
from tests.gateway.restart_test_helpers import make_restart_runner
from tests.gateway.test_retirement_publication import barrier


@pytest.mark.asyncio
async def test_stopped_owner_cannot_republish(tmp_path, monkeypatch):
    target = tmp_path / 'status.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    bind_work_sample_owner(runner)
    runner._running = False
    notify_work_changed()
    await barrier()
    assert not target.exists()


@pytest.mark.asyncio
async def test_retirement_dispatch_uses_bound_context_and_loop(monkeypatch):
    scope = ContextVar('fixture_scope', default='foreign')
    seen = []
    runner, _ = make_restart_runner()
    token = scope.set('owner')
    bind_work_sample_owner(runner)
    scope.reset(token)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(runner, '_persist_active_agents', lambda: seen.append((scope.get(), asyncio.get_running_loop())))
    await asyncio.to_thread(notify_work_changed)
    await barrier()
    assert seen == [('owner', loop)]


@pytest.mark.asyncio
async def test_subscription_does_not_retain_runner():
    runner, _ = make_restart_runner()
    bind_work_sample_owner(runner)
    reference = weakref.ref(runner)
    del runner, _
    gc.collect()
    assert reference() is None
    notify_work_changed()
    await barrier()
