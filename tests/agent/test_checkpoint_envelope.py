"""GXTD-843: final envelope saturation, controlled replies (no native calls)."""
import copy

import pytest

from agent.context_compressor import (
    ContextCompressor, HISTORICAL_TASK_HEADING, _NO_USER_TASK_SENTINEL,
    _handoff_only_content, _content_text_for_contains, is_compaction_summary_message,
    _extract_pruned_skill_names,
)
from agent.context_compressor_tail import SKILL_SAFETY_RULE
from tests.agent.test_gxtd811_best_effort_compression import _write_config


def history():
    rows = [{'role': 'system', 'content': 'Synthetic envelope fixture.'}]
    for i in range(16):
        rows.extend([{'role': 'user', 'content': f'Round {i}.\n' + 'archive detail\n' * 5},
                     {'role': 'assistant', 'content': f'Result {i}.\n' + 'completed routine work\n' * 80}])
    return rows


@pytest.fixture
def compressor(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, 'best_effort')
    c = ContextCompressor(model='fixture', config_context_length=100000, quiet_mode=True,
                          protect_first_n=1, protect_last_n=2, abort_on_summary_failure=True)
    c.tail_token_budget = 10
    return c


def carrier(rows):
    return '\n'.join(_content_text_for_contains(_handoff_only_content(m.get('content')))
                     for m in rows if is_compaction_summary_message(m))


@pytest.mark.parametrize('marker_at', ['none', 'summary', 'tail'])
def test_full_compression_saturated_optional_prose(compressor, monkeypatch, marker_at):
    rows = history()
    marker = "[SKILL_PRUNED: fixture; reload with skill_view(name='fixture')]"
    reply = f'{HISTORICAL_TASK_HEADING}\n{_NO_USER_TASK_SENTINEL}\n\n## Overview\n'
    reply += 'x' * 30000 + (('\n' + marker) if marker_at == 'summary' else '')
    if marker_at == 'tail':
        rows[-1]['content'] += '\n' + marker
    before = copy.deepcopy(rows)
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: reply)
    result = compressor.compress(rows, force=True)
    assert not compressor._last_compress_aborted, compressor._last_summary_error
    assert result != before
    assert rows == before
    text = carrier(result)
    assert len(text.encode()) <= 24576
    assert 'Optional summary excerpt' in text
    if marker_at != 'none':
        assert SKILL_SAFETY_RULE in text
        assert any('fixture' in _extract_pruned_skill_names(str(m.get('content'))) for m in result)


def test_unknown_generation_failure_is_content_free(compressor, monkeypatch):
    import json
    def fail(*args, **kwargs):
        raise ValueError('PRIVATE_TRANSCRIPT_AND_SECRET')
    monkeypatch.setattr(compressor, '_call_summary_llm', fail)
    rows = history()
    assert compressor.compress(copy.deepcopy(rows), force=True) == rows
    telemetry = compressor._last_compression_telemetry
    assert telemetry['checkpoint_stage'] == 'generation'
    assert telemetry['summary_error_code'] == 'summary_failure'
    assert 'PRIVATE_TRANSCRIPT_AND_SECRET' not in json.dumps(telemetry)


def test_native_terminal_accounting_survives_truncation(compressor, monkeypatch):
    from types import SimpleNamespace as NS
    import agent.context_compressor as module
    response = NS(choices=[NS(message=NS(content='partial'), finish_reason='length')],
                  usage=NS(prompt_tokens=100, completion_tokens=50))
    monkeypatch.setattr(module, 'call_llm', lambda **kwargs: response)
    rows = history()
    assert compressor.compress(copy.deepcopy(rows), force=True) == rows
    t = compressor._last_compression_telemetry
    assert t['aux_call_count'] == 1
    assert t['aux_finish_reason'] == 'length'
    assert t['aux_usage_prompt_tokens'] == 100
    assert t['aux_usage_completion_tokens'] == 50
    assert t['summary_error_code'] == 'native_output_length'


def test_final_validator_records_size_and_reason(compressor):
    from agent.compression_preservation import validate_checkpoint
    from agent.compression_budget import CompressionBudgetError
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
    compressor._begin_compress_attempt(None, True)
    candidate = [{'role': 'assistant', 'content': compressor._with_summary_prefix('x' * 30000),
                  COMPRESSED_SUMMARY_METADATA_KEY: True}]
    with pytest.raises(CompressionBudgetError):
        validate_checkpoint(compressor, [], candidate, 24576)
    t = compressor._last_compression_telemetry
    assert t['checkpoint_bytes'] > t['checkpoint_limit_bytes'] == 24576
    assert t['checkpoint_stage'] == 'final_validation'


def test_map_reduce_counts_real_chunks(compressor, monkeypatch):
    from types import SimpleNamespace as NS
    import agent.context_compressor as module
    seen = []
    def reply(**kwargs):
        seen.append(kwargs['messages'])
        return NS(choices=[NS(message=NS(content='No pending actions.'), finish_reason='stop')],
                  usage=NS(prompt_tokens=100, completion_tokens=10))
    monkeypatch.setattr(module, 'call_llm', reply)
    rows = history()
    rows[2]['content'] = 'ordinary archived text\n' * 24000
    compressor.compress(rows, force=True)
    t = compressor._last_compression_telemetry
    assert len(seen) > 1
    assert t['chunking'] is True
    assert t['chunk_count'] == len(seen) - 1
    assert t['aux_call_count'] == len(seen)
    assert t['aux_usage_completion_tokens'] == 10 * len(seen)


@pytest.mark.parametrize('delta', [-1, 0, 1])
@pytest.mark.parametrize('glyph', ['x', '界'])
@pytest.mark.parametrize('pointer', [False, True])
def test_exact_boundary_with_real_task_pointer(compressor, tmp_path, delta, glyph, pointer):
    from hermes_state import SessionDB
    from agent.compression_preservation import frame_best_effort, WORK_STATE_POINTER_HEADING
    from agent.protected_events import SourceRecords
    from agent.context_compressor import _SUMMARY_END_MARKER
    from tests.agent.test_gxtd811_best_effort_compression import _append_todo_pair
    with SessionDB(db_path=tmp_path / 'boundary.db') as db:
        db.create_session('boundary', 'cli', profile_name='default')
        compressor.bind_session_state(session_db=db, session_id='boundary')
        if pointer:
            _append_todo_pair(db, 'boundary', 'bounded')
        prose = glyph * 1000
        wide = frame_best_effort(compressor, prose, SourceRecords([], []), 24576)
        limit = len((wide + '\n\n' + _SUMMARY_END_MARKER).encode()) + delta
        actual = frame_best_effort(compressor, prose, SourceRecords([], []), limit)
        text = actual + '\n\n' + _SUMMARY_END_MARKER
        assert len(text.encode()) <= limit
        assert (WORK_STATE_POINTER_HEADING in actual) == pointer
        assert ('Optional summary excerpt' in actual) == (delta < 0)
        assert '\ufffd' not in actual


def test_two_saturated_compactions_install_and_reopen(compressor, monkeypatch, tmp_path):
    from hermes_state import SessionDB
    from tests.agent.test_gxtd811_best_effort_compression import _append_todo_pair
    from agent.compression_preservation import WORK_STATE_POINTER_HEADING
    path = tmp_path / 'repeat.db'
    with SessionDB(db_path=path) as db:
        db.create_session('repeat', 'cli', profile_name='default')
        compressor.bind_session_state(session_db=db, session_id='repeat')
        db.append_messages_batch('repeat', copy.deepcopy(history()))
        _append_todo_pair(db, 'repeat', 'one')
        for turn in range(2):
            if turn:
                db.append_messages_batch('repeat', copy.deepcopy(history()[1:]))
            with db._read_ctx() as conn:
                originals = [(r['id'], r['role'], r['content']) for r in conn.execute(
                    "SELECT id,role,content FROM messages WHERE session_id='repeat' ORDER BY id")]
            rows = db.get_messages_as_conversation('repeat', repair_alternation=True)
            reply = '## Overview\nNext action: review fixture.\n' + '界' * 12000
            monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: reply)
            result = compressor.compress(copy.deepcopy(rows), force=True)
            assert not compressor._last_compress_aborted, compressor._last_summary_error
            assert result != rows
            assert len(carrier(result).encode()) <= 24576
            assert WORK_STATE_POINTER_HEADING in carrier(result)
            db.archive_and_compact('repeat', copy.deepcopy(result))
            with db._read_ctx() as conn:
                after = {r['id']: (r['role'], r['content']) for r in conn.execute(
                    "SELECT id,role,content FROM messages WHERE session_id='repeat'")}
            assert all(after[i] == (role, text) for i, role, text in originals)
    with SessionDB(db_path=path) as db:
        rows = db.get_messages_as_conversation('repeat', repair_alternation=True)
        assert 'Next action: review fixture.' in carrier(rows)
        assert WORK_STATE_POINTER_HEADING in carrier(rows)
        assert db.get_latest_todo_result('repeat')


@pytest.mark.parametrize('tail_role', ['user', 'assistant'])
def test_merged_and_standalone_final_carriers(compressor, tail_role):
    from types import SimpleNamespace
    from agent.compression_preservation import frame_best_effort, validate_checkpoint
    from agent.protected_events import SourceRecords
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
    rows = [{'role': 'system', 'content': 'fixture'}, {'role': 'assistant', 'content': 'old'},
            {'role': tail_role, 'content': 'fresh retained text'}]
    scan = SimpleNamespace(tail_start=2, summary_indices=set())
    compressor._summary_has_user_turn = True
    summary = frame_best_effort(compressor, 'x' * 30000, SourceRecords([], []), 24576)
    actual = compressor._assemble_compressed(rows, 1, 2, scan, summary)
    validate_checkpoint(compressor, rows, actual, 24576)
    carrying = [m for m in actual if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert len(carrying) == 1
    assert ('fresh retained text' in str(carrying[0]['content'])) == (tail_role == 'user')


@pytest.mark.parametrize('oversized', [False, True])
def test_trusted_events_survive_or_refuse(compressor, monkeypatch, oversized):
    from agent.protected_events import EventStream, FIELD, canonical, decode_events
    event = {'version': 1, 'session_id': 'fixture-session', 'submission_id': 'test',
             'kind': 'unknown_outcome', 'text': 'Unsettled action.' + ('界' * 12000 if oversized else ''),
             'corrects': [], 'issuer': 'ed25519:test', 'order': 1, 'source_uid': 'src-1'}
    rows = history()
    rows[-1][FIELD] = EventStream('fixture-session', (canonical(event),))
    compressor._session_id = 'fixture-session'
    before = copy.deepcopy(rows)
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: '## Overview\n' + 'x' * 30000)
    actual = compressor.compress(rows, force=True)
    assert rows == before
    if oversized:
        assert compressor._last_compress_aborted
        assert actual == before
    else:
        assert not compressor._last_compress_aborted, compressor._last_summary_error
        assert decode_events(carrier(actual)) == [event]
