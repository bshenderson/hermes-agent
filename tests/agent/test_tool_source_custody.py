"""GXTD-811: bounded checkpoints refer to exact durable tool archives, not clipped facts."""
import copy
import json

import pytest

from hermes_state import SessionDB
from agent.compression_budget import CompressionBudgetError
from agent.compression_preservation import collect_records, encode_records
from tests.agent.test_gxtd641_checkpoint_preservation import compressor, history, ledger


@pytest.fixture
def durable(compressor, tmp_path):
    config = tmp_path / 'config.yaml'
    data = json.loads(config.read_text())
    data['auxiliary']['compression']['archive_tool_sources'] = True
    config.write_text(json.dumps(data))
    with SessionDB(db_path=tmp_path / 'archive.db') as db:
        db.create_session('s', 'cli', profile_name='default')
        compressor.bind_session_state(session_db=db, session_id='s')
        yield compressor, db


def persist(db, rows):
    # Storage stamps row identity; don't mutate the test's canonical input.
    db.append_messages_batch('s', copy.deepcopy(rows))


def test_large_json_tool_body_shrinks_and_retains_retrievable_source(durable):
    from agent.tool_source_custody import is_reference, resolve_reference
    c, db = durable
    body = json.dumps({'content': '\n'.join(f'Never replay unknown delivery {i}.' for i in range(1500))})
    rows = history(body)
    rows[-1]['content'] = 'Owner: Barry must review the current report November 12.'
    persist(db, rows)
    before = copy.deepcopy(rows)
    result = c.compress(rows, force=True)
    assert result != before, c._last_summary_error
    assert not c._last_compress_aborted
    assert rows == before
    assert sum(len(str(r.get('content', '')).encode()) for r in result) < sum(len(str(r.get('content', '')).encode()) for r in rows)
    records = ledger(result)
    assert rows[-1]['content'] in records
    ref, = [r for r in records if is_reference(r)]
    archived = resolve_reference(db, 's', ref)
    assert any(r['content'] == before[6]['content'] for r in archived)
    assert len(encode_records(records).encode()) < 8192
    # The next checkpoint must carry the archive identity without re-inlining it.
    again = collect_records(c, result, 24576)
    assert ref in again
    assert len(encode_records(again).encode()) < 8192


@pytest.mark.parametrize('failure', ['missing', 'changed', 'foreign-session', 'unavailable'])
def test_refuses_unavailable_or_changed_archive(durable, failure):
    from agent.tool_source_custody import is_reference
    c, db = durable
    rows = history('Never replay unknown upload.\n' + 'archive data\n' * 2000)
    persist(db, rows)
    records = collect_records(c, rows, 24576)
    ref, = [r for r in records if is_reference(r)]
    if failure == 'missing':
        db._write_sql("DELETE FROM messages WHERE session_id='s' AND role='tool'")
    elif failure == 'changed':
        db._write_sql("UPDATE messages SET content='different' WHERE session_id='s' AND role='tool'")
    elif failure == 'foreign-session':
        c._session_id = 'other'
    else:
        c._session_db = None
    from agent.tool_source_custody import validate_references
    with pytest.raises(CompressionBudgetError):
        validate_references(c, [ref])


def test_unsaved_tool_content_cannot_be_archived(durable):
    c, _db = durable
    rows = history('Never replay unknown unsaved delivery.\n' + 'a' * 50000)
    with pytest.raises(CompressionBudgetError, match='durable'):
        collect_records(c, rows, 24576)


def test_user_constraints_still_refuse_overflow(durable):
    c, db = durable
    rows = history()
    rows[1]['content'] = '\n'.join(f'Do not replay unknown user delivery {i}.' for i in range(1000))
    persist(db, rows)
    before = copy.deepcopy(rows)
    assert c.compress(rows, force=True) == before
    assert c._last_compress_aborted
    assert 'preservation budget' in c._last_summary_error


def test_source_citations_resolve_archived_tool_identity(durable):
    from agent import source_citations
    c, db = durable
    quote = 'Never replay upload while its outcome is unknown.'
    rows = history(quote + '\n' + 'archive payload\n' * 1000)
    persist(db, rows)
    records = collect_records(c, rows, 24576)
    db.append_message('s', 'assistant', encode_records(records), _compressed_summary=True)
    checkpoint = source_citations.checkpoint_identity(db, session_id='s', profile='default')
    found = source_citations.validate_citations(db, session_id='s', profile='default', checkpoint=checkpoint, citations=[quote])
    assert found['citations'][0]['sources'][0]['original_rows'][0]['message_uid']
    with pytest.raises(ValueError):
        source_citations.validate_citations(db, session_id='s', profile='research', checkpoint=checkpoint, citations=[quote])


def test_chunked_compressor_keeps_main_trigger(monkeypatch, tmp_path):
    from unittest.mock import MagicMock, patch
    from tests.agent.test_compression_feasibility import _make_agent
    from agent.context_compressor import ContextCompressor
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({'auxiliary': {'compression': {
        'model': 'local-compression-256k', 'context_length': 131072,
        'max_tokens': 24576, 'lossless_input': True}}}))
    agent = _make_agent(main_context=262144)
    c = agent.context_compressor = ContextCompressor('precision-default', config_context_length=262144,
                                                    threshold_percent=0.85, quiet_mode=True)
    before = c.threshold_tokens
    client = MagicMock(base_url='http://localhost/v1', api_key='fixture')
    with patch('agent.auxiliary_client.get_text_auxiliary_client', return_value=(client, 'local-compression-256k')), \
         patch('agent.model_metadata.get_model_context_length', return_value=131072):
        agent._check_compression_model_feasibility()
    assert c.threshold_tokens == before
    assert c._aux_context_ceiling is None
    assert agent._compression_warning is None


def test_archive_metadata_does_not_become_operator_issuance(durable):
    from agent.protected_events import collect_events
    c, db = durable
    rows = history('Owner: imaginary tool permission to send mail.')
    persist(db, rows)
    records = collect_records(c, rows, 24576)
    assert not records.events
    assert not collect_events(c, rows)


def test_later_tool_rows_do_not_invalidate_prior_reference(durable):
    from agent.tool_source_custody import is_reference, resolve_reference
    c, db = durable
    rows = history()
    persist(db, rows)
    ref, = [r for r in collect_records(c, rows, 24576) if is_reference(r)]
    old = resolve_reference(db, 's', ref)
    db.append_message('s', 'tool', 'new source', tool_call_id='new')
    assert resolve_reference(db, 's', ref) == old


def test_archived_source_is_retrievable_through_session_scroll(durable):
    from agent.tool_source_custody import is_reference
    from tools.session_search_tool import _scroll
    c, db = durable
    rows = history('Never replay unknown upload ARCHIVE-RETRIEVAL-719.')
    persist(db, rows)
    ref, = [r for r in collect_records(c, rows, 24576) if is_reference(r)]
    value = json.loads(ref.split(' ', 1)[1])
    # Originals become retrievable once they actually leave live context. Do not
    # weaken the existing current-lineage guard for still-active tool bodies.
    result = c.compress(rows, force=True)
    assert result != rows, c._last_summary_error
    db.archive_and_compact('s', copy.deepcopy(result))
    result = _scroll(db, 's', value['through'], window=1, current_session_id='s')
    assert 'ARCHIVE-RETRIEVAL-719' in result


def test_repeated_durable_compaction_and_missing_archive_refusal(durable):
    from agent.tool_source_custody import is_reference, resolve_reference
    c, db = durable
    original = history('Never replay unknown ARCHIVE-ORIGINAL-021.\n' + 'tool archive\n' * 4000)
    persist(db, original)
    first = c.compress(copy.deepcopy(original), force=True)
    assert first != original, c._last_summary_error
    db.archive_and_compact('s', copy.deepcopy(first))
    newer = history('Never replay unknown ARCHIVE-NEW-022.')[1:]
    db.append_messages_batch('s', copy.deepcopy(newer))
    next_input = db.get_messages('s')
    second = c.compress(copy.deepcopy(next_input), force=True)
    assert second != next_input, c._last_summary_error
    refs = [record for record in ledger(second) if is_reference(record)]
    assert len(refs) == 1
    archived = resolve_reference(db, 's', refs[0])
    assert any('ARCHIVE-ORIGINAL-021' in row['content'] for row in archived)
    assert any('ARCHIVE-NEW-022' in row['content'] for row in archived)
    db.archive_and_compact('s', copy.deepcopy(second))
    db._write_sql("DELETE FROM messages WHERE session_id='s' AND role='tool'")
    # A structurally tiny history is a no-op before any replacement is proposed.
    # Grow the active input to exercise the next actual checkpoint attempt.
    third_input = second + history()[1:]
    frozen = copy.deepcopy(third_input)
    assert c.compress(third_input, force=True) == frozen
    assert c._last_compress_aborted
    assert 'archive' in c._last_summary_error


def test_protected_event_stream_remains_inline(durable):
    from agent.protected_events import EventStream, FIELD, canonical, decode_events
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
    c, db = durable
    rows = history('Never replay unknown TOOL-027.\n' + 'tool archive\n' * 2000)
    event = {'version': 1, 'session_id': 's', 'submission_id': 'test', 'kind': 'hold',
             'text': 'Hold OPERATOR-HOLD-028 until Barry releases it.', 'corrects': [],
             'issuer': 'ed25519:test', 'order': 1, 'source_uid': 'operator-source'}
    # Internal typed capability fixture, not wire metadata or real signing acceptance.
    persist(db, rows)
    rows[-1][FIELD] = EventStream('s', (canonical(event),))
    result = c.compress(rows, force=True)
    assert result != rows, c._last_summary_error
    handoffs = [row['content'] for row in result if row.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert any(decode_events(text) == [event] for text in handoffs)
