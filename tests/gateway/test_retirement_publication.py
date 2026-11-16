"""GXTD-755: owner retirement updates durable counts without a heartbeat."""
import asyncio
from types import SimpleNamespace
import pytest
from cron import scheduler
from gateway import status
from gateway.run_work_sample import bind_work_sample_owner
from gateway.platforms.api_server_work import reserve_pending_api_work
from tests.gateway.restart_test_helpers import make_restart_runner


async def barrier():
    event = asyncio.Event()
    asyncio.get_running_loop().call_soon(event.set)
    await event.wait()
    assert status.flush_runtime_status()


@pytest.mark.asyncio
async def test_cron_worker_retirement_preserves_competing_api(tmp_path, monkeypatch):
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    monkeypatch.setattr(runner, '_api_server_hook', lambda name: 1)
    bind_work_sample_owner(runner)
    home = scheduler._get_hermes_home()
    owner = object()
    assert scheduler.try_register_running_job('retirement-fixture', owner=owner)
    runner._persist_active_agents()
    await barrier()
    assert status.read_runtime_status(target)['active_agents'] == 2
    await asyncio.to_thread(scheduler.release_running_job, 'retirement-fixture', home, owner=owner)
    await barrier()
    state = status.read_runtime_status(target)
    assert state['active_agents'] == 1
    assert state['work_sample']['components']['cron'] == 0
    assert state['work_sample']['components']['api'] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'refusal', 'cancel'])
async def test_api_reservation_retirement_publishes(tmp_path, monkeypatch, outcome):
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    adapter = SimpleNamespace(_pending_agent_requests=0)
    monkeypatch.setattr(runner, '_api_server_hook', lambda name: adapter._pending_agent_requests)
    bind_work_sample_owner(runner)
    try:
        with reserve_pending_api_work(adapter):
            runner._persist_active_agents()
            await barrier()
            assert status.read_runtime_status(target)['active_agents'] == 1
            if outcome == 'refusal':
                raise ValueError('fixture refusal')
            if outcome == 'cancel':
                raise asyncio.CancelledError()
    except (ValueError, asyncio.CancelledError):
        pass
    await barrier()
    assert adapter._pending_agent_requests == 0
    assert status.read_runtime_status(target)['active_agents'] == 0


@pytest.mark.asyncio
async def test_scheduler_dispatch_finally_publishes(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from cron import jobs
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    monkeypatch.setattr(runner, '_api_server_hook', lambda name: 1)
    bind_work_sample_owner(runner)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    def body(job):
        runner._persist_active_agents()
        loop.call_soon_threadsafe(started.set)
        if not release.wait(10):
            raise RuntimeError('fixture release deadline')
        return True
    job = jobs.create_job('offline fixture', 'every 15m', deliver='local')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = scheduler._submit_with_guard(job, pool, body)
        try:
            await asyncio.wait_for(started.wait(), 5)
            await barrier()
            assert status.read_runtime_status(target)['active_agents'] == 2
        finally:
            release.set()
        assert await asyncio.wrap_future(future)
    await barrier()
    assert not scheduler.is_job_running(job['id'])
    assert status.read_runtime_status(target)['active_agents'] == 1
