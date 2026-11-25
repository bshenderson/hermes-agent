"""Canonical admission/retirement diagnostic characterization, GXTD-755.
No inference; real authority, API admission, and runtime status writer.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch
import pytest

async def case(home, outcome, other):
    from gateway import status
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.api_server_run_queue import admit_request
    from gateway.session_results import execution_result
    from persistence_assembly import open_persistence, profile_scope
    from tests.gateway.restart_test_helpers import make_restart_runner

    db = open_persistence(profile_scope(home=home))
    store = SessionStore(config=GatewayConfig(), sessions_dir=home / 'sessions')
    store._db, store._persistence_beside = db, lambda pinned: pinned
    runner, _ = make_restart_runner()
    runner.session_store = store
    authority = SessionAuthority(runner, profile_id='default', instance_id='probe', persistence=db,
                                 epoch=db.runtime.begin_epoch(instance_id='probe'))
    runner.session_authority = authority
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    adapter._session_db = db
    runner.adapters = {Platform.API_SERVER: adapter}
    runner._adapter_for_source = lambda source: adapter
    adapter._check_auth = lambda request: None
    adapter._draining_response = lambda: None
    adapter._max_concurrent_runs = 0
    adapter._max_pending_runs = 0
    target = home / 'gateway_state.json'
    loop = asyncio.get_running_loop()
    remaining = loop.create_future() if other else None
    if remaining is not None:
        runner._track_deferred_agent_worker(remaining, object())
    entered = asyncio.Event()
    release = asyncio.Event()
    snapshots = []

    async def handle(event):
        entered.set()
        await release.wait()
        execution_result.get()['result'] = {'final_response': 'fixture completion'}
        return 'fixture completion'
    runner._handle_message = handle

    async def handler(a, request):
        runner._persist_active_agents()
        assert status.flush_runtime_status()
        snapshots.append(status.read_runtime_status(target)['active_agents'])
        if outcome == 'refusal':
            return 'refused-before-turn'
        if outcome == 'exception':
            raise ValueError('fixture handler failure')
        return await a._run_agent('fixture', [], session_id='probe-conversation')

    try:
        with patch.object(status, '_get_runtime_status_path', return_value=target), \
             patch('gateway.platforms.api_server_runs._uses_room_run_auth', return_value=False):
            task = asyncio.create_task(admit_request(handler)(adapter, SimpleNamespace()))
            if outcome in ('success', 'cancelled'):
                waiter = asyncio.create_task(entered.wait())
                done, _ = await asyncio.wait([task, waiter], timeout=10, return_when=asyncio.FIRST_COMPLETED)
                if task in done:
                    await task  # expose setup errors instead of hanging on an event
                assert waiter in done, 'canonical handler did not start'
                if outcome == 'cancelled':
                    task.cancel()
                else:
                    release.set()
            try:
                await asyncio.wait_for(task, 10)
            except asyncio.CancelledError:
                assert outcome == 'cancelled'
            except ValueError:
                assert outcome == 'exception'
            # Cancellation retires the observer, not the shielded canonical producer.
            mid = runner._active_work_count()
            release.set()
            drains = [live.task for live in authority.sessions.values() if live.task is not None]
            if drains:
                await asyncio.wait_for(asyncio.gather(*drains), 10)
            from gateway.session_runtime_workers import mutation_tasks
            mutations = mutation_tasks(authority)
            if mutations:
                await asyncio.wait_for(asyncio.gather(*mutations), 10)
            # Run queued completion callbacks without introducing a timing sleep.
            barrier = loop.create_future()
            loop.call_soon(barrier.set_result, None)
            await barrier
            live = runner._active_work_count()
            assert status.flush_runtime_status()
            payload = status.read_runtime_status(target)
            saved = payload['active_agents']
            assert payload['work_sample']['total'] == saved
            assert all(payload['work_sample']['availability'].values())
            runner._persist_active_agents()
            assert status.flush_runtime_status()
            control = status.read_runtime_status(target)['active_agents']
            if live != int(other):
                print('LIVE_DIAGNOSTIC', runner._describe_active_work(), flush=True)
                print('AUTHORITY_TASKS', [(str(t), t.done()) for t in getattr(authority, '_mutation_tasks', ())], flush=True)
            assert live == int(other), (outcome, live)
            assert control == int(other)
            assert saved == int(other)
            assert adapter._pending_agent_requests == 0
            if remaining is not None:
                assert not remaining.done()
            return dict(outcome=outcome, other_owner=other, admitted=snapshots,
                        after_observer=mid, live=live, persisted=saved, control=control)
    finally:
        release.set()
        if remaining is not None:
            remaining.set_result(None)
            retired = loop.create_future()
            loop.call_soon(retired.set_result, None)
            await retired
        status.flush_runtime_status()
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'refusal', 'exception', 'cancelled'])
@pytest.mark.parametrize('other', [False, True])
async def test_canonical_publication(tmp_path, monkeypatch, outcome, other):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(tmp_path / 'runtime'))
    await case(tmp_path, outcome, other)
