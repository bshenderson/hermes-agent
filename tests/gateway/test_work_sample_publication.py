"""GXTD-755: published work provenance survives platform heartbeats."""
from gateway import status
import json
import asyncio
import threading
import pytest
from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner


def test_work_sample_and_total_share_publication(tmp_path, monkeypatch):
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    first = status.read_runtime_status(target)
    assert first is not None
    assert 'work_sample' in first
    sample = first['work_sample']
    assert sample['total'] == first['active_agents'] == sum(sample['components'].values())
    assert set(sample['components']) == {'chat', 'managed', 'cron', 'api', 'deferred'}
    status.publish_runtime_status(platform='api_server', platform_state='connected')
    assert status.flush_runtime_status()
    refreshed = status.read_runtime_status(target)
    assert refreshed is not None
    assert refreshed['work_sample'] == sample


def test_lifecycle_publication_has_matching_sample(tmp_path, monkeypatch):
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    GatewayRunner._update_runtime_status(runner, 'running')
    assert status.flush_runtime_status()
    result = status.read_runtime_status(target)
    assert result is not None
    assert result.get('work_sample') is not None
    assert result['work_sample']['total'] == result['active_agents']


@pytest.mark.asyncio
async def test_health_exposes_same_sanitized_sample(api, tmp_path, monkeypatch):
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    monkeypatch.setattr('gateway.run._resolve_gateway_model', lambda: 'fixture')
    runner, _ = make_restart_runner()
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    runtime = status.read_runtime_status(target)
    assert runtime is not None
    expected = runtime['work_sample'].copy()
    runtime['work_sample']['prompt'] = 'PRIVATE_SENTINEL'
    monkeypatch.setattr(status, 'read_runtime_status', lambda: runtime)
    response = await api._handle_health_detailed(None)
    data = json.loads(response.text)
    assert data.get('work_sample') == expected
    assert 'PRIVATE_SENTINEL' not in response.text


@pytest.mark.asyncio
async def test_off_loop_publication_reads_on_bound_owner_loop(tmp_path, monkeypatch):
    target = tmp_path / 'gateway_state.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    threads = []
    monkeypatch.setattr(runner, '_running_agent_count', lambda: threads.append(threading.get_ident()) or 0)
    runner._persist_active_agents()
    await asyncio.to_thread(runner._persist_active_agents)
    event = asyncio.Event()
    asyncio.get_running_loop().call_soon(event.set)
    await event.wait()
    assert threads == [threading.get_ident(), threading.get_ident()]
    assert status.flush_runtime_status()
