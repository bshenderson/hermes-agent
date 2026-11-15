"""GXTD-755: persisted work accounting follows deferred-worker retirement."""

import asyncio

import pytest

from gateway import status
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancelled"])
@pytest.mark.parametrize("other_active", [False, True])
async def test_deferred_retirement_publishes_remaining_work(tmp_path, monkeypatch, outcome, other_active):
    target = tmp_path / "gateway_state.json"
    monkeypatch.setattr(status, "_get_runtime_status_path", lambda: target)
    runner, _adapter = make_restart_runner()
    loop = asyncio.get_running_loop()
    worker = loop.create_future()
    other = loop.create_future() if other_active else None
    runner._track_deferred_agent_worker(worker, object())
    if other is not None:
        runner._track_deferred_agent_worker(other, object())
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    assert status.read_runtime_status(target)["active_agents"] == 1 + int(other_active)
    settled = asyncio.Event()
    worker.add_done_callback(lambda _: settled.set())
    try:
        if outcome == "cancelled":
            worker.cancel()
        elif outcome == "error":
            worker.set_exception(RuntimeError("fixture worker failure"))
        else:
            worker.set_result(None)
        await settled.wait()
        assert worker not in runner._deferred_agent_workers
        assert runner._active_work_count() == int(other_active)
        assert status.flush_runtime_status()
        assert status.read_runtime_status(target)["active_agents"] == int(other_active)
        if other is not None:
            assert not other.done()
            assert other in runner._deferred_agent_workers
    finally:
        if other is not None:
            retired = asyncio.Event()
            other.add_done_callback(lambda _: retired.set())
            other.set_result(None)
            await retired.wait()
        assert status.flush_runtime_status()
