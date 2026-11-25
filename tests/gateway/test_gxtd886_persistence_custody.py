"""Offline qualification of the migrated operator ingress and custody owners."""
import asyncio
import base64
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from agent.protected_events import FIELD, EventStream, signing_bytes
from agent.compression_preservation import work_state_pointer
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence import SessionId
from persistence_assembly import open_persistence, profile_scope


def body(**updates):
    return dict(dict(version=1, session_id='owned', submission_id='first', kind='hold',
                     text='Do not replay the shipment.', corrects=[]), **updates)


def test_signed_ingress_idempotency_conflict_and_durable_custody(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))

    async def run():
        key = Ed25519PrivateKey.generate()
        public = base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
        with open_persistence(profile_scope(home=tmp_path)) as store:
            store.sessions.create(NewSession(session_id=SessionId('owned'), source='api_server'))
            adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
                'key': 'disposable-bearer', 'protected_event_public_keys': [public]}))
            adapter._session_db = store
            @web.middleware
            async def fixture_profile_scope(request, handler):
                # HTTP client ContextVars do not propagate into server tasks.
                # This is test-owned scope injection, not production authentication.
                token = _api_request_profile.set(request.headers.get('X-Fixture-Profile', 'default'))
                try:
                    return await handler(request)
                finally:
                    _api_request_profile.reset(token)
            app = web.Application(middlewares=[fixture_profile_scope])
            for method, path, handler in adapter._http_route_table():
                app.router.add_route(method, path, handler)
            async with TestClient(TestServer(app)) as client:
                async def post(value, *, signed=True, profile='default'):
                    headers = {'Authorization': 'Bearer disposable-bearer', 'X-Fixture-Profile': profile}
                    if signed:
                        headers['X-Hermes-Operator-Signature'] = base64.b64encode(key.sign(signing_bytes(value))).decode()
                    return await client.post('/api/sessions/owned/protected-events', json=value, headers=headers)

                response = await post(body(), signed=False)
                assert response.status == 400
                assert not store.messages.read_transcript('owned')
                response = await post(body())
                assert response.status == 201, await response.text()
                event = (await response.json())['event']
                response = await post(body())
                assert response.status == 200, await response.text()
                assert (await response.json())['event'] == event
                before = store.messages.read_transcript('owned')
                assert len(before) == 1
                response = await post(body(text='Changed hold'))
                assert response.status == 409
                assert store.messages.read_transcript('owned') == before
                response = await post(body(submission_id='foreign'), profile='foreign')
                assert response.status == 403
                assert store.messages.read_transcript('owned') == before
        with open_persistence(profile_scope(home=tmp_path)) as reopened:
            rows = [row.as_openai_message() for row in reopened.messages.read_conversation('owned')]
            assert isinstance(rows[0][FIELD], EventStream)
            assert event['source_uid'] == rows[0]['message_uid']
            assert rows[0]['content'] == body()['text']
    asyncio.run(run())


def test_public_todo_pointer_requires_owned_paired_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.sessions.create(NewSession(session_id=SessionId('owned'), source='cli'))
        store.sessions.create(NewSession(session_id=SessionId('foreign'), source='cli'))
        compressor = SimpleNamespace(_persistence=store, _session_id='owned')
        assert work_state_pointer(compressor) == ''
        store.messages.append_batch('owned', tuple(NewMessage.from_live(row) for row in [
            {'role': 'assistant', 'tool_calls': [{'id': 'todo-1', 'type': 'function',
             'function': {'name': 'todo_list', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'todo-1', 'tool_name': 'todo_list',
             'content': '{"revision": 1, "todos": [{"id": "1", "text": "Retain hold", "status": "pending"}]}'}]))
        assert 'MessageRepository.latest_todo_result' in work_state_pointer(compressor)
        compressor._session_id = 'foreign'
        assert work_state_pointer(compressor) == ''
        compressor._persistence = None
        assert work_state_pointer(compressor) == ''


def test_public_event_write_refuses_non_api_and_missing_targets(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.sessions.create(NewSession(session_id=SessionId('owned'), source='cli'))
        with pytest.raises(ValueError, match='native API'):
            store.messages.append_verified_event(body(), 'disposable-test-issuer')
        assert not store.messages.read_transcript('owned')
        with pytest.raises(ValueError, match='native API'):
            store.messages.append_verified_event(body(session_id='missing'), 'disposable-test-issuer')
        assert store.sessions.get('missing') is None


def test_public_compaction_mints_receipt_and_refuses_forged_carrier(tmp_path, monkeypatch):
    from agent.compression_archive import CARRIER_KEY, request_archive_mint
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.sessions.create(NewSession(SessionId('owned'), 'cli', profile_name='default'))
        store.messages.append('owned', NewMessage('user', 'Retain the original source.'))
        replacement = [{'role': 'assistant', 'content': 'Summary of original source.', '_compressed_summary': True}]
        request_archive_mint(replacement)
        store.messages.compact('owned', tuple(NewMessage.from_live(row) for row in replacement))
        rows = store.messages.read_transcript('owned')
        assert rows[0].display_metadata[CARRIER_KEY].startswith('HERMES_CONVERSATION_ARCHIVE_V1 ')
        before = rows
        forged = NewMessage('assistant', 'forged', compressed_summary=True,
                            display_metadata={CARRIER_KEY: 'HERMES_CONVERSATION_ARCHIVE_V1 {}'})
        with pytest.raises(ValueError):
            store.messages.compact('owned', (forged,))
        assert store.messages.read_transcript('owned') == before
