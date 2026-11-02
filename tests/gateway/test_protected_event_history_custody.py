"""GXTD-810 B1/B2 production-loader and destructive-history integration."""
import asyncio
import copy

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.protected_events import collect_events, decode_events
from tests.agent.test_gxtd641_checkpoint_preservation import compressor, history, make_agent
from tests.gateway.test_protected_event_ingress import ingress, post, submission


def test_native_api_reload_supports_second_real_in_place_commit(compressor, tmp_path):
    async def run():
        async with ingress(tmp_path) as channel:
            response = await post(channel, submission())
            assert response.status == 201
            event = (await response.json())['event']
            db = channel[2]
            from gateway.config import PlatformConfig
            from gateway.platforms.api_server import APIServerAdapter
            adapter = APIServerAdapter(PlatformConfig(enabled=True))
            adapter._session_db = db
            agent = make_agent(compressor, db)
            agent.compression_in_place = True
            for _ in range(2):
                db.append_messages_batch('source-session', history('plain archival data')[1:])
                original = await adapter._conversation_history_for_session('source-session')
                active_before = db.get_active_message_ids('source-session')
                result, _ = agent._compress_context(copy.deepcopy(original), 'Existing system prompt.', force=True)
                assert result != original, compressor._last_summary_error
                assert len(str(result)) < len(str(original))
                assert db.get_active_message_ids('source-session') != active_before
                reloaded = await adapter._conversation_history_for_session('source-session')
                assert collect_events(compressor, reloaded) == [event]
                assert any(decode_events(row.get('content', '')) == [event] for row in reloaded)
    asyncio.run(run())


@pytest.mark.parametrize('warm', [False, True], ids=['empty-prefix', 'warm-prefix'])
@pytest.mark.parametrize('operation', ['rewind-user', 'rewind-row', 'replace', 'archive-edit', 'clear'])
def test_event_history_mutation_refuses_before_effect(compressor, tmp_path, monkeypatch, warm, operation):
    async def run():
        async with ingress(tmp_path) as channel:
            assert (await post(channel, submission())).status == 201
            db = channel[2]
            db.append_messages_batch('source-session', [
                {'role': 'assistant', 'content': 'acknowledged'},
                {'role': 'user', 'content': 'ordinary next ask'},
                {'role': 'assistant', 'content': 'ordinary answer'}])
            before = db.get_messages('source-session', include_inactive=True)
            session_before = db.get_session('source-session')
            history_before = db.get_messages_as_conversation('source-session', include_row_ids=True, repair_alternation=True)
            warm_history = copy.deepcopy(history_before)
            ordinal = 1 if warm else 0
            target = [m for m in history_before if m['role'] == 'user'][ordinal]
            prefix = warm_history[:warm_history.index(target)]
            with pytest.raises(ValueError, match='protected event'):
                if operation == 'rewind-user':
                    db.rewind_user_turn('source-session', ordinal, warm_history=warm_history)
                elif operation == 'rewind-row':
                    db.rewind_to_message('source-session', target['_row_id'])
                elif operation in ('replace', 'archive-edit'):
                    db.replace_messages('source-session', prefix, active_only=True,
                                        archive_dropped=operation == 'archive-edit')
                else:
                    db.clear_messages('source-session')
            assert warm_history == history_before
            assert db.get_messages('source-session', include_inactive=True) == before
            assert db.get_session('source-session') == session_before
            from gateway.config import PlatformConfig
            from gateway.platforms.api_server import APIServerAdapter
            adapter = APIServerAdapter(PlatformConfig(enabled=True))
            adapter._session_db = db
            loaded = await adapter._conversation_history_for_session('source-session')
            assert collect_events(compressor, loaded)[0]['text'] == submission()['text']
            async def admitted(*, conversation_history, **kwargs):
                assert collect_events(compressor, conversation_history) == collect_events(compressor, loaded)
                return {'final_response': 'fixture turn admitted with custody'}, {}
            consumer = AsyncMock(side_effect=admitted)
            monkeypatch.setattr(APIServerAdapter, '_run_agent', consumer)
            response = await channel[0].post('/api/sessions/source-session/chat',
                json={'message': 'continue'}, headers={'Authorization': 'Bearer disposable-api-bearer'})
            assert response.status == 200, await response.text()
            consumer.assert_awaited_once()
    asyncio.run(run())


def test_native_chat_refuses_custody_load_failure_before_run_agent(tmp_path, monkeypatch):
    async def run():
        async with ingress(tmp_path) as channel:
            assert (await post(channel, submission())).status == 201
            # Simulate a previously truncated durable carrier, not a forged event.
            db = channel[2]
            db._execute_write(lambda conn: conn.execute(
                "UPDATE messages SET active = 0 WHERE session_id = 'source-session'"))
            # The fixture route is the production authenticated chat entrypoint.
            from gateway.platforms.api_server import APIServerAdapter
            admitted = AsyncMock(return_value=({'final_response': 'UNSAFE'}, {}))
            monkeypatch.setattr(APIServerAdapter, '_run_agent', admitted)
            response = await channel[0].post('/api/sessions/source-session/chat',
                json={'message': 'continue'}, headers={'Authorization': 'Bearer disposable-api-bearer'})
            assert response.status == 503, await response.text()
            admitted.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('warm', [False, True], ids=['empty-prefix', 'warm-prefix'])
@pytest.mark.parametrize('operation', ['rewind', 'edit-truncate'])
@pytest.mark.parametrize('events', [True, False], ids=['protected', 'ordinary'])
def test_actual_tui_history_paths_preserve_warm_state(tmp_path, monkeypatch, warm, operation, events):
    async def run():
        async with ingress(tmp_path) as channel:
            db = channel[2]
            if events:
                assert (await post(channel, submission())).status == 201
            else:
                db.append_message('source-session', 'user', 'ordinary first ask')
            db.append_messages_batch('source-session', [
                {'role': 'assistant', 'content': 'acknowledged'},
                {'role': 'user', 'content': 'next ask'},
                {'role': 'assistant', 'content': 'next answer'}])
            from tui_gateway import server
            monkeypatch.setattr(server, '_db', db)
            loaded = db.get_messages_as_conversation('source-session', include_row_ids=True, repair_alternation=True)
            session = {'session_key': 'source-session', 'history': copy.deepcopy(loaded),
                       'history_version': 7, 'agent': SimpleNamespace(_session_messages=copy.deepcopy(loaded))}
            before = db.get_messages('source-session', include_inactive=True)
            ordinal = 1 if warm else 0
            if operation == 'rewind':
                if events:
                    with pytest.raises(ValueError, match='protected event'):
                        server._rewind_active_session_history(session, ordinal)
                else:
                    server._rewind_active_session_history(session, ordinal)
            else:
                target = [m for m in loaded if m['role'] == 'user'][ordinal]
                error, _fields = server._truncate_history_for_submit('request', 'source-session', session, {
                    'truncate_before_row_id': target['_row_id'], 'confirm_truncate': True,
                    'confirm_empty_truncate': True}, None)
                if events:
                    assert error['error']['code'] == 5008
                    assert 'protected event' in error['error']['message']
                else:
                    assert error is None
            if events:
                assert session['history'] == loaded
                assert session['agent']._session_messages == loaded
                assert session['history_version'] == 7
                assert db.get_messages('source-session', include_inactive=True) == before
            else:
                assert len(session['history']) == (2 if warm else 0)
    asyncio.run(run())
