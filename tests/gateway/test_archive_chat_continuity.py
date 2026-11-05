"""GXTD-814: signed UI continuation uses owned native originals, not raw replay."""
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from agent.chat_location import native_context, sign_context
from agent.usage_anchor import capture_usage_anchor, restore_usage_anchor
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from hermes_state import SessionDB

PROFILE = 'homelab-coordinator'
KEY = 'fixture-key'
CHAT = '11111111-1111-4111-8111-111111111111'


def signed(messages, *, subject='owner-a', chat=CHAT, profile=PROFILE, stream=False):
    body = {'model': 'fixture', 'messages': messages, 'stream': stream,
            'hermes_artifact_transport': 'inline-v1'}
    context = native_context(user_id=subject, metadata={
        'chat_id': chat, 'message_id': str(uuid.uuid4()), 'user_message_id': str(uuid.uuid4())})
    body['hermes_chat_context'] = sign_context(
        context, body=body, user_id=subject, profile=profile, key=KEY)
    return body


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from agent import conversation_compression
    monkeypatch.setattr(conversation_compression, '_archive_receipts_enabled', lambda: True)
    db = SessionDB(db_path=tmp_path / 'state.db')
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': KEY}))
    monkeypatch.setattr(adapter, '_expected_api_key', lambda: KEY)
    monkeypatch.setattr(adapter, '_ensure_session_db_async', AsyncMock(return_value=db))
    calls = []

    async def run(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        sid = kwargs['session_id']
        if not kwargs['conversation_history']:
            # Controlled provider observation, never a native model dispatch.
            db.create_session(sid, 'api_server', profile_name=PROFILE)
            user = {'role': 'user', 'content': kwargs['user_message']}
            db.append_message(sid, 'user', user['content'])
            db.append_message(sid, 'assistant', 'READY')
            db.patch_session_model_config(sid, {'_usage_anchor': capture_usage_anchor(5000, 10, [user])})
        return {'final_response': 'READY', 'completed': True, 'session_id': sid}, {
            'input_tokens': 5000, 'output_tokens': 10, 'total_tokens': 5010}

    monkeypatch.setattr(adapter, '_run_agent', run)

    async def post(body, *, headers=None, profile=PROFILE):
        request = make_mocked_request('POST', '/v1/chat/completions',
                                      headers={'Authorization': 'Bearer ' + KEY, **(headers or {})})
        monkeypatch.setattr(request, 'json', AsyncMock(return_value=body))
        token = _api_request_profile.set(profile)
        try:
            return await adapter._handle_chat_completions(request)
        finally:
            _api_request_profile.reset(token)

    yield db, adapter, calls, post
    db.close()


@pytest.mark.asyncio
async def test_real_handler_restores_native_prefix_and_real_anchor_shape(harness):
    db, _, calls, post = harness
    first = [{'role': 'user', 'content': 'Original raw UI text'}]
    assert (await post(signed(first))).status == 200
    sid = calls[0]['session_id']
    native = db.get_messages_as_conversation(sid)
    assert native[0]['content'] != first[0]['content']  # server delivery guidance
    second = first + [{'role': 'assistant', 'content': 'READY'}, {'role': 'user', 'content': 'Next'}]
    assert (await post(signed(second))).status == 200
    assert calls[1]['session_id'] == sid
    assert calls[1]['conversation_history'] == native
    agent = SimpleNamespace(_usage_anchor=None, _persist_disabled=False, session_id=sid, _session_db=db)
    restore_usage_anchor(agent, calls[1]['conversation_history'])
    assert agent._usage_anchor is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('other_subject,other_chat', [('owner-b', CHAT), ('owner-a', '22222222-2222-4222-8222-222222222222')])
async def test_owned_idempotency_isolates_subject_and_chat(harness, other_subject, other_chat):
    _, _, calls, post = harness
    messages = [{'role': 'user', 'content': 'Identical greeting'}]
    headers = {'Idempotency-Key': 'owned-' + str(uuid.uuid4())}
    first = signed(messages)
    assert (await post(first, headers=headers)).status == 200
    assert (await post(first, headers=headers)).status == 200
    assert len(calls) == 1  # same signed request really used the cache
    assert (await post(signed(messages, subject=other_subject, chat=other_chat), headers=headers)).status == 200
    assert len(calls) == 2  # another owner/chat must execute its own turn
    assert calls[0]['session_id'] != calls[1]['session_id']


@pytest.mark.asyncio
async def test_real_http_ingress_auth_and_owned_cache_isolation(harness):
    import os
    import tempfile
    from aiohttp import ClientSession, UnixConnector
    _, adapter, calls, _ = harness
    app = web.Application()

    @web.middleware
    async def selected_profile(request, handler):
        token = _api_request_profile.set(PROFILE)
        try:
            return await handler(request)
        finally:
            _api_request_profile.reset(token)

    app.middlewares.append(selected_profile)
    handler = next(handler for method, path, handler in adapter._http_route_table()
                   if method == 'POST' and path == '/v1/chat/completions')
    app.router.add_post('/v1/chat/completions', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        with tempfile.TemporaryDirectory(prefix='a814-', dir=os.environ['TMPDIR']) as directory:
            path = directory + '/s'
            await web.UnixSite(runner, path).start()
            async with ClientSession(connector=UnixConnector(path=path)) as client:
                body = signed([{'role': 'user', 'content': 'HTTP greeting'}])
                for authorization in ('', 'Bearer wrong-key'):
                    async with client.post('http://localhost/v1/chat/completions', json=body,
                                           headers={'Authorization': authorization}) as response:
                        assert response.status == 401
                assert calls == []
                headers = {'Authorization': 'Bearer ' + KEY, 'Idempotency-Key': str(uuid.uuid4())}
                for subject in ('owner-a', 'owner-a', 'owner-b'):
                    async with client.post('http://localhost/v1/chat/completions',
                                           json=signed([{'role': 'user', 'content': 'HTTP greeting'}], subject=subject),
                                           headers=headers) as response:
                        assert response.status == 200
                        await response.json()
                assert len(calls) == 2
                assert calls[0]['session_id'] != calls[1]['session_id']
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_different_subject_and_chat_do_not_share_greeting(harness):
    _, _, calls, post = harness
    messages = [{'role': 'user', 'content': 'Hello'}]
    for subject, chat in [('a', CHAT), ('b', CHAT), ('a', str(uuid.uuid4()))]:
        assert (await post(signed(messages, subject=subject, chat=chat))).status == 200
    assert len({c['session_id'] for c in calls}) == 3


@pytest.mark.asyncio
async def test_import_only_history_refuses_before_dispatch(harness):
    _, _, calls, post = harness
    response = await post(signed([{'role': 'user', 'content': 'Imported'},
                                 {'role': 'assistant', 'content': 'Old'},
                                 {'role': 'user', 'content': 'Next'}]))
    assert response.status == 409
    assert calls == []


@pytest.mark.asyncio
async def test_arbitrary_continuation_header_refuses(harness):
    _, _, calls, post = harness
    response = await post(signed([{'role': 'user', 'content': 'Hello'}]),
                          headers={'X-Hermes-Session-Id': 'foreign'})
    assert response.status == 400
    assert calls == []


@pytest.mark.asyncio
async def test_signature_tampering_refuses_before_history(harness):
    _, _, calls, post = harness
    body = signed([{'role': 'user', 'content': 'Hello'}])
    body['messages'][0]['content'] = 'Changed after signature'
    assert (await post(body)).status == 400
    assert calls == []


@pytest.mark.asyncio
async def test_store_fault_never_falls_back_to_client_history(harness, monkeypatch):
    _, adapter, calls, post = harness
    monkeypatch.setattr(adapter, '_ensure_session_db_async', AsyncMock(return_value=None))
    response = await post(signed([{'role': 'user', 'content': 'Hello'}]))
    assert response.status == 503
    assert calls == []


@pytest.mark.asyncio
async def test_owned_chat_remains_owned_when_publication_opt_out(harness, monkeypatch):
    db, _, calls, post = harness
    first = [{'role': 'user', 'content': 'Original'}]
    assert (await post(signed(first))).status == 200
    native = db.get_messages_as_conversation(calls[0]['session_id'])
    monkeypatch.setattr('agent.conversation_compression._archive_receipts_enabled', lambda: False)
    assert (await post(signed(first + [{'role': 'assistant', 'content': 'READY'},
                                    {'role': 'user', 'content': 'Next'}]))).status == 200
    assert calls[1]['conversation_history'] == native


@pytest.mark.asyncio
async def test_unclaimed_disabled_chat_keeps_legacy_replay(harness, monkeypatch):
    from gateway.platforms.api_server import _derive_chat_session_id
    db, _, calls, post = harness
    monkeypatch.setattr('agent.conversation_compression._archive_receipts_enabled', lambda: False)
    history = [{'role': 'user', 'content': 'Original'}, {'role': 'assistant', 'content': 'Old'}]
    assert (await post(signed(history + [{'role': 'user', 'content': 'Next'}]))).status == 200
    assert calls[0]['conversation_history'] == history
    assert calls[0]['session_id'] == _derive_chat_session_id(None, 'Original', PROFILE)
    with db._read_ctx() as conn:
        assert conn.execute("SELECT count(*) FROM sessions WHERE id LIKE 'owui-archive-%'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_unsigned_request_keeps_legacy_behavior(harness):
    _, _, calls, post = harness
    assert (await post({'messages': [{'role': 'user', 'content': 'Unsigned'}]})).status == 200
    assert not calls[0]['session_id'].startswith('owui-archive-')


@pytest.mark.asyncio
async def test_bad_api_key_refuses_before_dispatch(harness):
    _, _, calls, post = harness
    response = await post(signed([{'role': 'user', 'content': 'Hello'}]),
                          headers={'Authorization': 'Bearer wrong-fixture-key'})
    assert response.status == 401
    assert calls == []


@pytest.mark.asyncio
async def test_wrong_profile_signature_refuses(harness):
    _, _, calls, post = harness
    assert (await post(signed([{'role': 'user', 'content': 'Hello'}]), profile='research')).status == 400
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [('profile_name', 'research'), ('user_id', 'foreign'),
                                        ('chat_id', 'different-chat')])
async def test_corrupted_owner_never_rebinds(harness, field, value):
    db, _, calls, post = harness
    first = [{'role': 'user', 'content': 'Original'}]
    assert (await post(signed(first))).status == 200
    sid = calls[0]['session_id']
    db._execute_write(lambda conn: conn.execute(f'UPDATE sessions SET {field}=? WHERE id=?', (value, sid)))
    assert (await post(signed(first + [{'role': 'assistant', 'content': 'READY'},
                                    {'role': 'user', 'content': 'Next'}]))).status == 409
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_rotated_busy_owned_turn_refuses_without_dispatch(harness):
    db, _, calls, post = harness
    first = [{'role': 'user', 'content': 'Original'}]
    assert (await post(signed(first))).status == 200
    parent = calls[0]['session_id']
    child = 'busy-rotated-child'
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source='api_server',
        messages=[{'role': 'assistant', 'content': 'Owned rotated summary'}],
        profile_name=PROFILE, require_compression_lease=False)
    holder = 'rotated-busy-holder'
    assert db.acquire_session_turn_lease(child, holder, wait_seconds=0)
    try:
        assert (await post(signed(first + [{'role': 'assistant', 'content': 'READY'},
                                         {'role': 'user', 'content': 'Next'}]))).status == 409
        assert len(calls) == 1
    finally:
        db.release_session_turn_lease(child, holder)


@pytest.mark.asyncio
async def test_busy_owned_turn_refuses_without_dispatch(harness):
    db, _, calls, post = harness
    first = [{'role': 'user', 'content': 'Original'}]
    assert (await post(signed(first))).status == 200
    sid = calls[0]['session_id']
    assert db.acquire_session_turn_lease(sid, 'offline-test-holder', wait_seconds=0)
    try:
        assert (await post(signed(first + [{'role': 'assistant', 'content': 'READY'},
                                        {'role': 'user', 'content': 'Next'}]))).status == 409
        assert len(calls) == 1
    finally:
        db.release_session_turn_lease(sid, 'offline-test-holder')


@pytest.mark.asyncio
async def test_client_history_cannot_regrow_native_compaction(harness):
    from agent.compression_archive import request_archive_mint, CARRIER_KEY
    db, _, calls, post = harness
    first = [{'role': 'user', 'content': 'Original KESTREL request'}]
    assert (await post(signed(first))).status == 200
    sid = calls[0]['session_id']
    summary = {'role': 'assistant', 'content': 'Owned compacted summary.', '_compressed_summary': True}
    request_archive_mint([summary])
    db.archive_and_compact(sid, [summary], watermark=db.get_active_message_watermark(sid))
    native = db.get_messages_as_conversation(sid)
    assert len(native) == 1 and CARRIER_KEY in native[0]['display_metadata']
    assert (await post(signed(first + [{'role': 'assistant', 'content': 'Stale raw browser answer'},
                                    {'role': 'user', 'content': 'Next'}]))).status == 200
    assert calls[-1]['conversation_history'] == native


@pytest.mark.asyncio
async def test_real_stream_spawn_gets_same_owned_native_history(harness, monkeypatch):
    db, adapter, calls, post = harness
    first = [{'role': 'user', 'content': 'Original'}]
    assert (await post(signed(first))).status == 200
    native = db.get_messages_as_conversation(calls[0]['session_id'])
    seen = []

    async def writer(request, completion_id, model, created, queue, task, agent_ref, **kwargs):
        await task  # Real spawn/task handoff; controlled model, no wire socket.
        seen.append(kwargs['session_id'])
        return web.Response(text='controlled stream completed')

    monkeypatch.setattr(adapter, '_write_sse_chat_completion', writer)
    response = await post(signed(first + [{'role': 'assistant', 'content': 'READY'},
                                        {'role': 'user', 'content': 'Next'}], stream=True))
    assert response.status == 200
    assert seen == [calls[0]['session_id']]
    assert calls[1]['conversation_history'] == native


@pytest.mark.asyncio
async def test_published_rotation_keeps_owned_native_tip(harness):
    db, _, calls, post = harness
    first = [{'role': 'user', 'content': 'Original'}]
    assert (await post(signed(first))).status == 200
    parent = calls[0]['session_id']
    child = 'offline-compression-child'
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source='api_server',
        messages=[{'role': 'assistant', 'content': 'Owned rotated summary'}],
        profile_name=PROFILE, require_compression_lease=False)
    native = db.get_messages_as_conversation(child)
    assert (await post(signed(first + [{'role': 'assistant', 'content': 'READY'},
                                    {'role': 'user', 'content': 'Next'}]))).status == 200
    assert calls[-1]['session_id'] == child
    assert calls[-1]['conversation_history'] == native


@pytest.mark.asyncio
async def test_rotation_resolution_fault_refuses_before_dispatch(harness, monkeypatch):
    db, _, calls, post = harness
    first = [{'role': 'user', 'content': 'Original'}]
    assert (await post(signed(first))).status == 200

    def unavailable(_session_id):
        raise RuntimeError('offline lineage lookup unavailable')

    monkeypatch.setattr(db, 'get_compression_tip', unavailable)
    response = await post(signed(first + [{'role': 'assistant', 'content': 'READY'},
                                        {'role': 'user', 'content': 'Next'}]))
    assert response.status == 503
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_completed_writer_between_history_read_and_real_admission_is_not_lost(harness, monkeypatch):
    import threading
    from agent.turn_facade_lease import admit_durable_turn_lease

    db, adapter, _calls, post = harness
    first = [{'role': 'user', 'content': 'Original'}]
    assert (await post(signed(first))).status == 200
    observed = []

    async def admitted_run(**kwargs):
        sid = kwargs['session_id']
        # A real competing writer completes after route history loading but
        # before this consumer acquires the ordinary durable turn lease.
        assert db.acquire_session_turn_lease(sid, 'completed-writer', wait_seconds=0)
        try:
            db.append_message(sid, 'user', 'Completed intervening user')
            db.append_message(sid, 'assistant', 'Completed intervening answer')
        finally:
            db.release_session_turn_lease(sid, 'completed-writer')
        agent = SimpleNamespace(
            _session_db=db, _persist_disabled=False, session_id=sid,
            _owned_chat_history=kwargs.get('owned_chat_history'),
            _liveness_activity_lock=lambda: threading.RLock())
        admission = admit_durable_turn_lease(
            agent, session_id=sid, relay_turn_id='offline-admitted-reader',
            task_context={'platform': 'api_server'},
            conversation_history=kwargs['conversation_history'])
        assert admission.lease is not None
        try:
            assert admission.conversation_history is not None
            observed.extend(admission.conversation_history)
        finally:
            admission.lease.release()
        return {'final_response': 'READY', 'completed': True, 'session_id': sid}, {}

    monkeypatch.setattr(adapter, '_run_agent', admitted_run)
    assert (await post(signed(first + [{'role': 'assistant', 'content': 'READY'},
                                    {'role': 'user', 'content': 'Next'}]))).status == 200
    assert any(row['content'] == 'Completed intervening answer' for row in observed)


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['owner', 'rotation', 'store'])
async def test_post_read_fault_refuses_and_releases_real_lease(harness, monkeypatch, fault):
    import threading
    from agent.turn_facade_lease import admit_durable_turn_lease

    db, adapter, calls, post = harness
    first = [{'role': 'user', 'content': 'Original'}]
    assert (await post(signed(first))).status == 200
    sid = calls[0]['session_id']

    async def refusing_run(**kwargs):
        if fault == 'owner':
            db._execute_write(lambda conn: conn.execute(
                'UPDATE sessions SET user_id=? WHERE id=?', ('foreign', sid)))
        elif fault == 'rotation':
            db.publish_compression_child(
                parent_session_id=sid, child_session_id='new-live-tip', source='api_server',
                messages=[{'role': 'assistant', 'content': 'New summary'}],
                profile_name=PROFILE, require_compression_lease=False)
        else:
            def unavailable(_sid):
                raise RuntimeError('post-read lineage unavailable')
            monkeypatch.setattr(db, 'get_compression_tip', unavailable)
        agent = SimpleNamespace(
            _session_db=db, _persist_disabled=False, session_id=sid,
            _owned_chat_history=kwargs['owned_chat_history'],
            _liveness_activity_lock=lambda: threading.RLock())
        with pytest.raises((ValueError, RuntimeError)):
            admit_durable_turn_lease(
                agent, session_id=sid, relay_turn_id='refusing-reader',
                task_context={'platform': 'api_server'},
                conversation_history=kwargs['conversation_history'])
        assert agent._active_session_turn_lease_holder is None
        assert db.acquire_session_turn_lease(sid, 'cleanup-probe', wait_seconds=0)
        db.release_session_turn_lease(sid, 'cleanup-probe')
        return {'final_response': 'refused', 'completed': False, 'session_id': sid}, {}

    monkeypatch.setattr(adapter, '_run_agent', refusing_run)
    await post(signed(first + [{'role': 'assistant', 'content': 'READY'},
                              {'role': 'user', 'content': 'Next'}]))


@pytest.mark.asyncio
async def test_ordinary_agent_restores_exact_priced_anchor_after_admitted_reload(harness, tmp_path, monkeypatch):
    import json
    from contextlib import nullcontext
    from run_agent import AIAgent
    from agent import conversation_loop

    db, adapter, calls, post = harness
    first = [{'role': 'user', 'content': 'Original raw UI input'}]
    assert (await post(signed(first))).status == 200
    sid = calls[0]['session_id']
    home = tmp_path / 'ordinary-home'
    home.mkdir()
    (home / 'config.yaml').write_text(json.dumps({'auxiliary': {'compression': {'archive_receipts': True}}}))
    monkeypatch.setenv('HERMES_HOME', str(home))
    observed = []
    expected = capture_usage_anchor(5000, 10, db.get_messages_as_conversation(sid)[:1])

    class HydrationReached(RuntimeError):
        pass

    original_build = conversation_loop.build_turn_context

    def observed_build(agent, *args, **kwargs):
        original_build(agent, *args, **kwargs)
        assert agent._usage_anchor == expected
        assert agent._active_session_turn_lease_holder is not None
        observed.append(agent._usage_anchor)
        raise HydrationReached('production-hydration-reached')

    monkeypatch.setattr(conversation_loop, 'build_turn_context', observed_build)

    async def ordinary_run(**kwargs):
        agent = AIAgent(
            api_key='fixture', provider='custom', model='fixture', base_url='http://127.0.0.1:9/v1',
            quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=1,
            enabled_toolsets=['session_search'], session_id=sid, session_db=db)

        monkeypatch.setattr(adapter, '_create_agent', lambda **_kwargs: agent)
        monkeypatch.setattr(adapter, '_profile_scope', lambda _profile: nullcontext())
        with pytest.raises(HydrationReached, match='production-hydration-reached'):
            await APIServerAdapter._run_agent(adapter, **kwargs)
        assert agent._active_session_turn_lease_holder is None
        return {'final_response': 'READY', 'completed': True, 'session_id': sid}, {}

    monkeypatch.setattr(adapter, '_run_agent', ordinary_run)
    assert (await post(signed(first + [{'role': 'assistant', 'content': 'READY'},
                                    {'role': 'user', 'content': 'Next'}]))).status == 200
    assert len(observed) == 1
