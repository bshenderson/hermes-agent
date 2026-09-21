"""Exercise the real native executor paths without models or production state."""
import asyncio
from contextlib import nullcontext
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_DELEGATED_CHILD_CONTEXT', raising=False)
    from gateway.config import PlatformConfig
    from gateway.platforms import api_server, api_server_runs, api_server_lifecycle_metrics
    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True))
    adapter._api_lifecycle_metrics = api_server_lifecycle_metrics.Collector()
    monkeypatch.setattr(adapter, '_profile_scope', lambda *_: nullcontext())
    monkeypatch.setattr(adapter, '_bind_api_server_session', lambda **_: {})
    monkeypatch.setattr(api_server, '_publish_turn_process_ownership', lambda *_: None)
    monkeypatch.setattr(api_server, '_clear_turn_process_ownership', lambda *_: None)
    yield adapter, api_server, api_server_runs
    adapter._run_idempotency_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('family', ['request', 'durable_run'])
async def test_native_cancelled_waiter_does_not_end_executor(native, monkeypatch, family):
    adapter, api_server, runs = native
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    callbacks = {}
    def conversation(**kwargs):
        entered.set()
        try:
            assert release.wait(10)
            return {'interrupted': True, 'completed': False}
        finally:
            exited.set()
    agent = SimpleNamespace(run_conversation=conversation)
    def create(**kwargs):
        callbacks.update(kwargs)
        return agent
    monkeypatch.setattr(adapter, '_create_agent', create)
    if family == 'request':
        task = asyncio.create_task(adapter._run_agent('PRIVATE-PROMPT', [], stream_delta_callback=lambda *_: None))
    else:
        queue = asyncio.Queue()
        adapter._run_streams['PRIVATE-RUN'] = queue
        run = SimpleNamespace(
            run_id='PRIVATE-RUN', session_id='PRIVATE-SESSION', request_profile=None,
            approval_session_key='PRIVATE-APPROVAL', browser_control_principal=None,
            browser_control_transport_family=None, agent_kwargs={'room_dispatch': None},
            declared_selected=False, user_message='PRIVATE-PROMPT', conversation_history=[],
            queue=queue, put_event=queue.put_nowait, turn_author=None, session_history_delivery=False,
        )
        task = asyncio.create_task(runs._execute_run(adapter, run, _api_server=api_server))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        text = adapter._api_lifecycle_metrics.render()
        assert f'hermes_api_executors_active{{family="{family}"}} 1' in text
        assert f'hermes_api_progress_unobserved{{family="{family}"}} 1' in text
        callbacks['stream_delta_callback']('PRIVATE-TOKEN')
        assert f'hermes_api_progress_unobserved{{family="{family}"}} 0' in adapter._api_lifecycle_metrics.render()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert f'hermes_api_executors_active{{family="{family}"}} 1' in adapter._api_lifecycle_metrics.render()
    finally:
        release.set()
        assert await asyncio.to_thread(exited.wait, 5)
        if not task.done():
            await task
    # Let the executor's instrumentation finally run after the callback returned.
    for _ in range(100):
        text = adapter._api_lifecycle_metrics.render()
        if f'hermes_api_executors_active{{family="{family}"}} 0' in text:
            break
        await asyncio.sleep(.01)
    assert f'hermes_api_executor_outcomes_total{{family="{family}",outcome="interrupted"}} 1' in text
    assert 'PRIVATE-' not in text
