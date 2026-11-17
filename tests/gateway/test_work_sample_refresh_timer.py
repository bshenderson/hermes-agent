"""Exercise the actual periodic timer and durable writer without live services."""
import asyncio
from datetime import UTC, datetime

import pytest

from gateway import run_work_sample as samples, status
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_real_timer_recounts_twice_and_cancellation_retires(tmp_path, monkeypatch):
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    samples.bind_work_sample_owner(runner)
    published = asyncio.Event()
    observations = []
    publish = samples.publish_work_sample

    def observe(owner):
        publish(owner)
        assert status.flush_runtime_status()
        state = status.read_runtime_status(target)
        assert state is not None
        observations.append(state['work_sample'])
        if len(observations) == 2:
            published.set()

    monkeypatch.setattr(samples, 'publish_work_sample', observe)
    task = runner._spawn_supervised(lambda: samples.refresh_work_samples(runner), 'fixture_timer')
    try:
        await asyncio.wait_for(published.wait(), 20)
        first, second = observations
        assert second['sequence'] > first['sequence']
        assert second['monotonic_start'] > first['monotonic_end']
        age = (datetime.now(UTC) - datetime.fromisoformat(second['observed_at'])).total_seconds()
        assert 0 <= age <= 20
        assert first['total'] == second['total'] == 0
        assert all(second['availability'].values())
    finally:
        runner._running = False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert task not in runner._background_tasks
