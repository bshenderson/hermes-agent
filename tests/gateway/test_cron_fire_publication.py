"""Direct cron fire lifetime uses the same publication contract."""
import asyncio
import pytest
from cron import jobs, scheduler
from gateway import status
from gateway.run_work_sample import bind_work_sample_owner
from tests.gateway.restart_test_helpers import make_restart_runner
from tests.gateway.test_retirement_publication import barrier


@pytest.mark.asyncio
async def test_direct_fire_retirement_publishes(tmp_path, monkeypatch):
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    bind_work_sample_owner(runner)
    loop = asyncio.get_running_loop()
    def body(*args, **kwargs):
        runner._persist_active_agents()
        asyncio.run_coroutine_threadsafe(barrier(), loop).result(5)
        assert status.read_runtime_status(target)['active_agents'] == 1
        return True
    monkeypatch.setattr(scheduler, '_launch_external_cron_worker', lambda job: False)
    monkeypatch.setattr(scheduler, '_run_one_job_body', body)
    job = jobs.create_job('offline fixture', 'every 15m', deliver='local')
    assert await asyncio.to_thread(scheduler.run_one_job, job)
    await barrier()
    assert not scheduler.get_running_job_ids()
    assert status.read_runtime_status(target)['active_agents'] == 0


@pytest.mark.asyncio
async def test_stale_registration_sweep_publishes(tmp_path, monkeypatch):
    from concurrent.futures import Future
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    bind_work_sample_owner(runner)
    job = jobs.create_job('offline stale fixture', 'every 15m', deliver='local')
    future = Future()
    future.set_result(True)
    assert scheduler.try_register_running_job(job['id'], owner=future)
    key = scheduler._inflight_key(job['id'])
    with scheduler._running_lock:
        scheduler._running_since[key] -= 100_000_000
    runner._persist_active_agents()
    await barrier()
    assert status.read_runtime_status(target)['active_agents'] == 1
    assert await asyncio.to_thread(scheduler.sweep_stale_inflight, [job]) == [job['id']]
    await barrier()
    assert status.read_runtime_status(target)['active_agents'] == 0
