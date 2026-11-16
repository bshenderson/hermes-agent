"""Diagnostic dispatch must not turn a loop-close race into a turn failure."""
from contextvars import copy_context
from types import SimpleNamespace

from gateway import status
from tests.gateway.restart_test_helpers import make_restart_runner


def test_loop_closes_between_liveness_read_and_dispatch(tmp_path, monkeypatch):
    path = tmp_path / 'status.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: path)
    runner, _ = make_restart_runner()
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    def closed(*args, **kwargs):
        raise RuntimeError('Event loop is closed')
    runner._work_sample_loop = SimpleNamespace(is_running=lambda: True, call_soon_threadsafe=closed)
    runner._work_sample_context = copy_context()
    monkeypatch.setattr(runner, '_active_work_count', lambda: 2)
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    payload = status.read_runtime_status(path)
    assert payload is not None
    assert payload['active_agents'] == 2
    assert payload['work_sample'] is None
