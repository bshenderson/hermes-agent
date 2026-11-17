"""GXTD-755: idle observation freshness must not depend on new work."""
import asyncio
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from gateway import run_work_sample as samples, status
from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_startup_refreshes_idle_sample_without_new_work(tmp_path, monkeypatch):
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    samples.bind_work_sample_owner(runner)
    now = datetime.now(UTC)
    clock = SimpleNamespace(now=lambda tz: now, fromisoformat=datetime.fromisoformat)
    monkeypatch.setattr(samples, 'datetime', clock)
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    old = status.read_runtime_status(target)['work_sample']
    now += timedelta(seconds=42)

    ticked = asyncio.Event()
    parked = asyncio.Event()
    ticks = []

    async def tick(interval):
        ticks.append(interval)
        if len(ticks) > 1:
            ticked.set()
            await parked.wait()

    monkeypatch.setattr(samples, 'asyncio', SimpleNamespace(
        get_running_loop=asyncio.get_running_loop, sleep=tick))
    # Other background services are not part of the observation producer.
    spawn = runner._spawn_supervised
    monkeypatch.setattr(runner, '_spawn_supervised', lambda factory, name, **kw:
                        None if name in {'runtime_downgrade_watcher', 'drain_control_watcher'}
                        else spawn(factory, name, **kw))
    runner._PRE_RECONNECT_WATCHERS = ()
    runner._POST_RECONNECT_WATCHERS = ()
    runner._failed_platforms = {}
    runner._spawn_reconnect_watcher = lambda: None
    runner._scale_to_zero_should_arm = lambda: False
    runner._log_scale_to_zero_not_armed_reason = lambda: None
    GatewayRunner._start_spawn_background_watchers(runner)
    waiter = asyncio.create_task(ticked.wait())
    try:
        await asyncio.wait({waiter}, timeout=2)
        assert status.flush_runtime_status()
        fresh = status.read_runtime_status(target)['work_sample']
        age = (now - datetime.fromisoformat(fresh['observed_at'])).total_seconds()
        assert 0 <= age <= 20, 'idle owner observation expired without periodic recount'
        assert fresh['sequence'] > old['sequence']
        assert fresh['total'] == 0
        assert all(fresh['availability'].values())
        assert fresh['pid'] == old['pid'] and fresh['start_time'] == old['start_time']
        assert ticks and all(0 < interval < 20 for interval in ticks)
    finally:
        runner._running = False
        tasks = list(runner._background_tasks) + [waiter]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('owner', ['chat', 'managed', 'cron', 'api', 'deferred', 'unavailable'])
async def test_refresh_recounts_in_owner_context(tmp_path, monkeypatch, owner):
    from gateway import run_runtime
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    scope = ContextVar('publication_owner', default='foreign')
    token = scope.set('launch')
    samples.bind_work_sample_owner(runner)
    scope.reset(token)
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    old = status.read_runtime_status(target)['work_sample']
    observed_scopes = []

    def count(kind):
        observed_scopes.append(scope.get())
        if owner == 'unavailable' and kind == 'api':
            raise RuntimeError('unavailable fixture')
        return int(owner == kind)

    monkeypatch.setattr(runner, '_running_agent_count', lambda: count('chat'))
    monkeypatch.setattr(run_runtime, 'managed_turn_count', lambda r: count('managed'))
    monkeypatch.setattr(runner, '_running_cron_job_count', lambda: count('cron'))
    monkeypatch.setattr(runner, '_api_server_hook', lambda name: count('api'))
    monkeypatch.setattr(runner, '_active_deferred_agent_worker_count', lambda: count('deferred'))
    calls = 0

    async def tick(interval):
        nonlocal calls
        calls += 1
        if calls == 2:
            runner._running = False

    monkeypatch.setattr(samples, 'asyncio', SimpleNamespace(
        get_running_loop=asyncio.get_running_loop, sleep=tick))
    task = runner._spawn_supervised(lambda: samples.refresh_work_samples(runner), 'fixture_sample')
    await asyncio.wait_for(task, 2)
    assert status.flush_runtime_status()
    fresh = status.read_runtime_status(target)['work_sample']
    assert fresh['sequence'] > old['sequence']
    assert observed_scopes == ['launch'] * 5
    assert fresh['total'] == (0 if owner == 'unavailable' else 1)
    if owner == 'unavailable':
        assert fresh['availability']['api'] is False
    else:
        assert fresh['components'][owner] == 1
    assert task not in runner._background_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize('retirement', ['stopped', 'rebound', 'cancelled'])
async def test_refresh_cannot_publish_after_owner_retirement(tmp_path, monkeypatch, retirement):
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    samples.bind_work_sample_owner(runner)
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    before = target.read_bytes()

    async def tick(interval):
        if retirement == 'stopped':
            runner._running = False
        elif retirement == 'rebound':
            runner._work_sample_loop = object()
        else:
            raise asyncio.CancelledError

    monkeypatch.setattr(samples, 'asyncio', SimpleNamespace(
        get_running_loop=asyncio.get_running_loop, sleep=tick))
    task = runner._spawn_supervised(lambda: samples.refresh_work_samples(runner), 'fixture_sample')
    await asyncio.gather(task, return_exceptions=True)
    assert status.flush_runtime_status()
    assert target.read_bytes() == before
    assert task not in runner._background_tasks
