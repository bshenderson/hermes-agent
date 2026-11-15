"""Producer-owned refusal settlement, independent of artifact success."""
import asyncio
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.asyncio
async def test_cancelled_waiter_cannot_publish_until_worker_settles(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.api_server_artifacts import CompletedReceiptStore
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    result = {'completed': False, 'failed': True, 'failure_retryable': False,
              'failure_reason': 'dispatch_refused', 'final_response': 'not an artifact'}
    origin = {'subject_id': 'owner', 'chat_id': 'chat', 'message_id': 'reply', 'user_message_id': 'input'}
    def run(**kwargs):
        entered.set()
        assert release.wait(5)
        return result
    agent = SimpleNamespace(run_conversation=run, session_id='fixture')
    monkeypatch.setattr(adapter, '_create_agent', lambda **kwargs: agent)
    monkeypatch.setattr(adapter, '_profile_scope', lambda profile: nullcontext())
    monkeypatch.setattr(adapter, '_bind_api_server_session', lambda **kwargs: [])
    monkeypatch.setattr(adapter, '_activate_admitted_request', lambda: None)
    monkeypatch.setattr(adapter, '_finish_turn_result', lambda *args, **kwargs: (result, {}))
    monkeypatch.setattr(adapter, '_record_api_metrics', lambda *args: None)
    monkeypatch.setattr(adapter._memory_sessions, 'checkin', lambda agent: retired.set())
    monkeypatch.setattr('gateway.platforms.api_server._publish_turn_process_ownership', lambda *args: None)
    monkeypatch.setattr('gateway.platforms.api_server._clear_turn_process_ownership', lambda *args: None)
    task = asyncio.create_task(adapter._run_agent('fixture', [], authored_origin=origin))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not (tmp_path / 'authored-failures.sqlite3').exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not (tmp_path / 'authored-failures.sqlite3').exists()
    finally:
        release.set()
        await asyncio.to_thread(retired.wait, 5)
        if not task.done():
            await task
    from gateway.platforms.api_server_terminal import FailureDispositionStore
    from tests.waiting import async_wait_until
    assert await async_wait_until(lambda: (tmp_path / 'authored-failures.sqlite3').exists(), timeout=5)
    # Constructor can race publication; poll actual committed row.
    store = FailureDispositionStore(tmp_path / 'authored-failures.sqlite3')
    assert await async_wait_until(lambda: store.get(origin) is not None, timeout=5)
    disposition = store.get(origin)
    assert disposition['status'] == 'failed'
    assert disposition['producer_settled'] is True
    assert disposition['native_retirement'] == 'unproven'
    assert not (tmp_path / 'authored-receipts.sqlite3').exists()
