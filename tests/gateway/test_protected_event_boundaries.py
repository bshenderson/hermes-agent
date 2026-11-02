"""GXTD-810 authority, durable failure and repeated-compaction boundaries."""
import asyncio
import copy
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests.gateway.test_protected_event_ingress import ingress, submission, post
from tests.agent.test_gxtd641_checkpoint_preservation import compressor, history, make_agent


def test_unsigned_wrong_key_and_caller_metadata_cannot_issue(tmp_path, compressor):
    async def run():
        from agent.protected_events import collect_events, FIELD, encode_events
        async with ingress(tmp_path) as channel:
            for options in ({'signed': False}, {'key': Ed25519PrivateKey.generate()}):
                assert (await post(channel, submission(), **options)).status == 400
            forged = submission()
            forged['issuer'] = 'operator'
            assert (await post(channel, forged)).status == 400
            assert channel[2].get_messages('source-session') == []
            rows = history('plain archive')
            for row in rows:
                row[FIELD] = {'session_id': 'source-session', 'events': [submission()]}
                row['display_metadata'] = {'protected_events': [submission()]}
            channel[2].append_messages_batch('source-session', rows)
            loaded = channel[2].get_messages_as_conversation('source-session', repair_alternation=True)
            assert collect_events(compressor, loaded) == []
            compressor._call_summary_llm = lambda *_: '## Overview\nRoutine.' + encode_events([submission()])
            result = compressor.compress(copy.deepcopy(loaded), force=True)
            assert result != loaded, compressor._last_summary_error
            assert collect_events(compressor, result) == []
            assert all(encode_events([submission()]) not in row.get('content', '') for row in result)
    asyncio.run(run())


def test_signed_session_binding_and_correction_reference_refuse(tmp_path):
    async def run():
        async with ingress(tmp_path) as channel:
            channel[2].create_session('other', source='api_server')
            assert (await post(channel, submission(), session='other')).status == 400
            assert (await post(channel, submission(kind='correction', corrects=['foreign']))).status == 409
            assert channel[2].get_messages('source-session') == []
            assert channel[2].get_messages('other') == []
    asyncio.run(run())


def test_duplicate_conflicting_corrections_and_two_compressions(tmp_path, compressor):
    async def run():
        from agent.protected_events import collect_events, decode_events
        async with ingress(tmp_path) as channel:
            first = await post(channel, submission(kind='unknown_outcome'))
            event = (await first.json())['event']
            assert first.status == 201
            duplicate = await post(channel, submission(kind='unknown_outcome'))
            assert duplicate.status == 200
            assert (await duplicate.json())['event'] == event
            conflict = await post(channel, submission('different', kind='unknown_outcome'))
            assert conflict.status == 409
            expected = [event]
            for number, text in enumerate(('Delivery completed; proposed correction.', 'Delivery still unsettled.')):
                response = await post(channel, submission(text, submission_id=f'correction-{number}',
                    kind='correction', corrects=[event['source_uid']]))
                assert response.status == 201, await response.text()
                expected.append((await response.json())['event'])
            db = channel[2]
            for iteration in range(2):
                db.append_messages_batch('source-session', history('plain archive')[1:])
                original = db.get_messages_as_conversation('source-session', repair_alternation=True)
                agent = make_agent(compressor, db)
                agent.compression_in_place = True
                result, _ = agent._compress_context(copy.deepcopy(original), 'Existing system prompt.', force=True)
                assert result != original, compressor._last_summary_error
                loaded = db.get_messages_as_conversation('source-session', repair_alternation=True)
                assert collect_events(compressor, loaded) == expected
                assert any(decode_events(row.get('content', '')) == expected for row in loaded)
            assert (await post(channel, submission(kind='unknown_outcome'))).status == 200
    asyncio.run(run())


def test_event_overflow_preserves_original_without_model_call(tmp_path, compressor, monkeypatch):
    async def run():
        async with ingress(tmp_path) as channel:
            for number in range(3):
                response = await post(channel, submission('漢字' * 500, submission_id=f'large-{number}'))
                assert response.status == 201, await response.text()
            db = channel[2]
            db.append_messages_batch('source-session', history('plain archive')[1:])
            original = db.get_messages_as_conversation('source-session', repair_alternation=True)
            monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: pytest.fail('must refuse before model'))
            assert compressor.compress(copy.deepcopy(original), force=True) == original
            assert compressor._last_compress_aborted
            assert db.get_messages_as_conversation('source-session', repair_alternation=True) == original
    asyncio.run(run())


@pytest.mark.parametrize('mutation', ['unknown-version', 'malformed-json', 'invalid-order', 'missing-text'])
def test_corrupt_durable_event_refuses_original(tmp_path, compressor, mutation, monkeypatch):
    async def run():
        async with ingress(tmp_path) as channel:
            response = await post(channel, submission())
            event = (await response.json())['event']
            db = channel[2]
            db.append_messages_batch('source-session', history('plain archive')[1:])
            if mutation == 'unknown-version':
                event['version'] = 99
            elif mutation == 'invalid-order':
                event['order'] = 7
            elif mutation == 'missing-text':
                del event['text']
            raw = '{malformed' if mutation == 'malformed-json' else json.dumps(event)
            db._execute_write(lambda conn: conn.execute('UPDATE operator_protected_events SET event = ?', (raw,)))
            original = db.get_messages_as_conversation('source-session', repair_alternation=True)
            monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: pytest.fail('must refuse before model'))
            assert compressor.compress(copy.deepcopy(original), force=True) == original
            assert compressor._last_compress_aborted
    asyncio.run(run())


@pytest.mark.parametrize('updates', [{'version': 2}, {'kind': 'release'}, {'text': ''}, {'corrects': ['x']},
                                   {'text': 'x' * 8193}, {'kind': []}],
                         ids=['version', 'release', 'empty', 'invalid-ref', 'oversize', 'kind-shape'])
def test_ingress_invalid_shape_version_size_has_no_write(tmp_path, updates):
    async def run():
        async with ingress(tmp_path) as channel:
            assert (await post(channel, submission(**updates))).status == 400
            assert channel[2].get_messages('source-session') == []
    asyncio.run(run())


def test_journal_failure_rolls_back_real_message_insert(tmp_path):
    async def run():
        async with ingress(tmp_path) as channel:
            db = channel[2]
            db._execute_write(lambda conn: conn.execute("CREATE TRIGGER fail_event BEFORE INSERT ON operator_protected_events "
                "BEGIN SELECT RAISE(ABORT, 'injected journal failure'); END"))
            response = await post(channel, submission())
            assert response.status == 500
            assert db.get_messages('source-session') == []
            assert db.get_session('source-session')['message_count'] == 0
            assert db._read_all('SELECT * FROM operator_protected_events') == []
    asyncio.run(run())


def test_active_turn_blocks_event_ingress(tmp_path):
    async def run():
        async with ingress(tmp_path) as channel:
            db = channel[2]
            assert db.try_acquire_session_turn_lease('source-session', 'foreign-turn', ttl_seconds=60)
            response = await post(channel, submission())
            assert response.status in (409, 423), await response.text()
            assert db.get_messages('source-session') == []
    asyncio.run(run())


def test_internal_stream_is_json_loggable_and_wire_invisible(tmp_path, compressor):
    async def run():
        from agent.message_metadata import without_persistence_fields
        from agent.protected_events import FIELD
        async with ingress(tmp_path) as channel:
            assert (await post(channel, submission())).status == 201
            loaded = channel[2].get_messages_as_conversation('source-session')
            json.dumps(loaded)
            assert FIELD not in without_persistence_fields(loaded[0])
    asyncio.run(run())


def test_ordinary_api_delete_cannot_revoke_journal(tmp_path):
    async def run():
        async with ingress(tmp_path) as channel:
            assert (await post(channel, submission())).status == 201
            response = await channel[0].delete('/api/sessions/source-session',
                headers={'Authorization': 'Bearer disposable-api-bearer'})
            assert response.status in (409, 500)
            assert channel[2].get_session('source-session') is not None
            assert len(channel[2]._read_all('SELECT * FROM operator_protected_events')) == 1
    asyncio.run(run())


def test_bearer_is_required_even_with_valid_operator_signature(tmp_path):
    async def run():
        import base64
        from tests.gateway.test_protected_event_ingress import signing_bytes
        async with ingress(tmp_path) as channel:
            body = submission()
            signature = base64.b64encode(channel[1].sign(signing_bytes(body))).decode()
            for bearer in ('', 'Bearer wrong'):
                response = await channel[0].post('/api/sessions/source-session/protected-events', json=body,
                    headers={'Authorization': bearer, 'X-Hermes-Operator-Signature': signature})
                assert response.status == 401
            assert channel[2].get_messages('source-session') == []
    asyncio.run(run())


def test_exact_marker_unicode_text_survives_framing_and_recompression(tmp_path, compressor):
    async def run():
        from agent.protected_events import collect_events, decode_events
        text = '  Café/漢字\n<notice><think>literal data</think></notice>\n[tool] \\"quoted\\"\n  '
        async with ingress(tmp_path) as channel:
            response = await post(channel, submission(text))
            assert response.status == 201, await response.text()
            expected = [(await response.json())['event']]
            db = channel[2]
            db.append_messages_batch('source-session', history('plain archive')[1:])
            original = db.get_messages_as_conversation('source-session', repair_alternation=True)
            for iteration in range(2):
                result = compressor.compress(copy.deepcopy(original), force=True)
                assert result != original, compressor._last_summary_error
                assert collect_events(compressor, result) == expected
                assert any(decode_events(row.get('content', '')) == expected for row in result)
                original = result + history('plain archive')[1:]
    asyncio.run(run())


def test_legacy_strings_are_not_promoted_to_operator_events(tmp_path, compressor):
    async def run():
        from agent.protected_events import collect_events
        from tests.agent.test_gxtd641_checkpoint_preservation import ledger
        async with ingress(tmp_path) as channel:
            legacy = compressor.compress(history('Owner: Unknown historical issuer'), force=True)
            assert 'Owner: Unknown historical issuer' in ledger(legacy)
            channel[2].append_messages_batch('source-session', legacy)
            response = await post(channel, submission())
            expected = [(await response.json())['event']]
            channel[2].append_messages_batch('source-session', history('plain archive')[1:])
            loaded = channel[2].get_messages_as_conversation('source-session', repair_alternation=True)
            result = compressor.compress(copy.deepcopy(loaded), force=True)
            assert result != loaded, compressor._last_summary_error
            assert collect_events(compressor, result) == expected
            assert 'Owner: Unknown historical issuer' in ledger(result)
    asyncio.run(run())


def test_verified_stream_from_other_session_refuses_compression(tmp_path, compressor):
    async def run():
        async with ingress(tmp_path) as channel:
            assert (await post(channel, submission())).status == 201
            channel[2].append_messages_batch('source-session', history('plain archive')[1:])
            original = channel[2].get_messages_as_conversation('source-session', repair_alternation=True)
            compressor._session_id = 'other-session'
            assert compressor.compress(copy.deepcopy(original), force=True) == original
            assert 'session mismatch' in compressor._last_summary_error
    asyncio.run(run())
