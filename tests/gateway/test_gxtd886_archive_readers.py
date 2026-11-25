"""Offline public-persistence archive/citation consumer qualification."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, profile_scope
from tests.gateway.test_gxtd886_persistence_custody import body
from tests.agent.test_gxtd811_best_effort_compression import best_effort_compressor, history


def test_archive_public_read_reopen_recall_and_foreign_refusals(tmp_path, monkeypatch):
    from agent.compression_archive import ArchiveRecall, CARRIER_KEY, request_archive_mint, resolve_presentation, resolve
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.sessions.create(NewSession(SessionId('archive'), 'cli', profile_name='default'))
        store.messages.append('archive', NewMessage('user', 'KESTREL exact original source.'))
        replacement = [{'role': 'assistant', 'content': 'Summary.', '_compressed_summary': True}]
        request_archive_mint(replacement)
        store.messages.compact('archive', tuple(NewMessage.from_live(row) for row in replacement))
        reference = store.messages.read_transcript('archive')[0].display_metadata[CARRIER_KEY]
    with open_persistence(profile_scope(home=tmp_path)) as store:
        assert resolve_presentation(store, 'archive', 'default') == reference
        assert resolve(store, 'archive', 'default', reference)[0]['content'] == 'KESTREL exact original source.'
        recall = ArchiveRecall(store, 'archive', 'default', allowed_tools=['session_search'])
        result = json.loads(recall.invoke(reference, {'query': 'KESTREL'}))
        assert result['success'], result
        assert any(row['content'] == 'KESTREL exact original source.' for row in result['messages'])
        with pytest.raises(ValueError, match='owner'):
            resolve(store, 'archive', 'foreign', reference)
        with pytest.raises(ValueError, match='owner'):
            recall.invoke(reference, {'query': 'KESTREL', 'session_id': 'foreign'})
        with pytest.raises(ValueError, match='grant'):
            ArchiveRecall(store, 'archive', 'default', allowed_tools=[]).invoke(reference, {'query': 'KESTREL'})


def test_citation_registered_http_public_store_and_stale_checkpoint(tmp_path, monkeypatch, best_effort_compressor):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))

    async def run():
        with open_persistence(profile_scope(home=tmp_path)) as store:
            store.sessions.create(NewSession(SessionId('owned'), 'api_server', profile_name='default'))
            import base64
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
            from agent.protected_events import signing_bytes
            from gateway.platforms.api_server_protected_events import verified_issuer
            key = Ed25519PrivateKey.generate()
            public = base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
            signature = base64.b64encode(key.sign(signing_bytes(body()))).decode()
            event, _ = store.messages.append_verified_event(body(), verified_issuer([public], body(), signature))
            adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
                'key': 'disposable-bearer', 'source_citations_enabled': True}))
            adapter._session_db = store
            app = web.Application()
            for method, path, handler in adapter._http_route_table():
                app.router.add_route(method, path, handler)
            async with TestClient(TestServer(app)) as client:
                url = '/api/sessions/owned/source-citations'
                headers = {'Authorization': 'Bearer disposable-bearer'}
                response = await client.get(url, headers=headers)
                assert response.status == 200, await response.text()
                checkpoint = (await response.json())['checkpoint']
                payload = {'checkpoint': checkpoint, 'citations': [event['text']]}
                response = await client.post(url, headers=headers, json=payload)
                assert response.status == 200, await response.text()
                result = await response.json()
                assert result['citations'][0]['sources'][0]['original_rows'][0]['message_uid'] == event['source_uid']
                store.messages.append('owned', NewMessage('assistant', 'Additional durable response.'))
                response = await client.post(url, headers=headers, json=payload)
                assert response.status == 409
                with pytest.raises(ValueError, match='owner mismatch'):
                    store.messages.citation_snapshot('owned', 'foreign')
            # Actual compressor and public publication; only its summary-model
            # response is the explicitly offline fixture from best_effort_compressor.
            from persistence.conversation.messages import ConversationRequest
            from agent.protected_events import collect_events, decode_events
            store.messages.append_batch('owned', tuple(NewMessage.from_live(row) for row in history()[1:]))
            original = [row.as_openai_message() for row in store.messages.read_conversation(
                'owned', ConversationRequest(repair_alternation=True))]
            best_effort_compressor.bind_session_state(persistence=store, session_id='owned')
            compressed = best_effort_compressor.compress(original, force=True)
            assert compressed != original, best_effort_compressor._last_summary_error
            assert collect_events(best_effort_compressor, compressed) == [event]
            assert any(decode_events(row.get('content', '')) == [event] for row in compressed)
            store.messages.compact('owned', tuple(NewMessage.from_live(row) for row in compressed))
        with open_persistence(profile_scope(home=tmp_path)) as reopened:
            restored = [row.as_openai_message() for row in reopened.messages.read_conversation(
                'owned', ConversationRequest(repair_alternation=True))]
            assert collect_events(best_effort_compressor, restored) == [event]
            assert any(decode_events(row.get('content', '')) == [event] for row in restored)
    asyncio.run(run())


def test_tool_archive_uses_bound_persistence_and_refuses_foreign_reference(tmp_path, monkeypatch):
    from agent.tool_source_custody import archive_current_tools, resolve_reference
    from agent.compression_budget import CompressionBudgetError
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.sessions.create(NewSession(SessionId('tools'), 'cli', profile_name='default'))
        message = {'role': 'tool', 'tool_call_id': 'call-one', 'content': 'Exact tool output.'}
        store.messages.append('tools', NewMessage.from_live(message))
        compressor = SimpleNamespace(_persistence=store, _session_id='tools',
                                     _is_context_summary_message=lambda row: False)
        reference = archive_current_tools(compressor, [message], [])[0]
        assert resolve_reference(store, 'tools', reference)[0]['content'] == message['content']
        with pytest.raises(CompressionBudgetError):
            resolve_reference(store, 'foreign', reference)


def test_worker_owner_history_json_restores_custody_and_refuses_foreign(tmp_path, monkeypatch):
    import copy
    from agent.protected_events import FIELD, EventStream, collect_events
    from agent.runtime_session_compression import RuntimeSessionCompressionMixin
    from agent.compression_budget import CompressionBudgetError
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.sessions.create(NewSession(SessionId('owned'), 'api_server', profile_name='default'))
        event, _ = store.messages.append_verified_event(body(), 'ed25519:disposable-storage-fixture')
        rows = [row.as_openai_message() for row in store.messages.read_conversation('owned')]
        wire = json.loads(json.dumps(rows))
    assert not isinstance(wire[0][FIELD], EventStream)

    class Worker(RuntimeSessionCompressionMixin):
        def _apply(self, operation, payload):
            assert operation == 'compression.history'
            return {'messages': copy.deepcopy(wire)}

    restored = Worker().get_messages_as_conversation('owned')
    assert isinstance(restored[0][FIELD], EventStream)
    assert collect_events(SimpleNamespace(_session_id='owned'), restored) == [event]
    with pytest.raises(CompressionBudgetError, match='session mismatch'):
        Worker().get_messages_as_conversation('foreign')
    # A JSON-shaped model dictionary remains untrusted outside this owner seam.
    assert collect_events(SimpleNamespace(_session_id='owned'), wire) == []
