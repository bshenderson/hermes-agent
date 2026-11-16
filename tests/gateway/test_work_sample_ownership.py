"""Owner binding and unavailable-counter provenance for GXTD-755."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from gateway import status
from gateway.run_startup import GatewayStartupMixin
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_start_binds_before_startup_can_publish(monkeypatch):
    runner, _ = make_restart_runner()
    async def startup():
        assert getattr(runner, '_work_sample_loop', None) is asyncio.get_running_loop()
        return False
    monkeypatch.setattr(runner, '_start_impl', startup)
    monkeypatch.setattr(runner, '_start_flush_runtime_status', AsyncMock())
    assert await GatewayStartupMixin.start(runner) is False


@pytest.mark.parametrize('family', ['cron', 'api'])
def test_unavailable_counter_is_not_observed_idle(tmp_path, monkeypatch, family):
    target = tmp_path / 'status.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    def fail(*args):
        raise RuntimeError('PRIVATE_SENTINEL')
    if family == 'cron':
        monkeypatch.setattr(runner, '_running_cron_job_count', fail)
    else:
        monkeypatch.setattr(runner, '_api_server_hook', fail)
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    payload = status.read_runtime_status(target)
    sample = payload['work_sample']
    assert sample['availability'][family] is False
    assert sample['components'][family] == 0
    assert sample['total'] == payload['active_agents']
    assert 'PRIVATE_SENTINEL' not in target.read_text()


def test_components_are_sampled_once_without_a_second_aggregate(tmp_path, monkeypatch):
    from gateway import run_runtime
    target = tmp_path / 'status.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    calls = []
    def read(kind, count):
        def capture(*args):
            calls.append(kind)
            return count
        return capture
    monkeypatch.setattr(runner, '_running_agent_count', read('chat', 1))
    monkeypatch.setattr(run_runtime, 'managed_turn_count', read('managed', 2))
    monkeypatch.setattr(runner, '_running_cron_job_count', read('cron', 3))
    monkeypatch.setattr(runner, '_api_server_hook', read('api', 4))
    monkeypatch.setattr(runner, '_active_deferred_agent_worker_count', read('deferred', 5))
    def forbidden():
        raise AssertionError('second aggregate read')
    monkeypatch.setattr(runner, '_active_work_count', forbidden)
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    payload = status.read_runtime_status(target)
    assert payload is not None
    sample = payload['work_sample']
    assert sorted(calls) == sorted(sample['components'])
    assert sample['components'] == dict(chat=1, managed=2, cron=3, api=4, deferred=5)
    assert payload['active_agents'] == sample['total'] == sum(sample['components'].values())
