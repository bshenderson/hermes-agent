"""Behavior contracts at actual compression/installation boundaries.

All responses are deliberately controlled fixtures, not native model results.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from agent.context_compressor import (
    ContextCompressor, COMPRESSED_SUMMARY_METADATA_KEY, SUMMARY_PREFIX,
    HISTORICAL_TASK_HEADING, _NO_USER_TASK_SENTINEL, _SUMMARY_END_MARKER,
    _redact_compaction_text, salvage_grown_transcript, is_compaction_summary_message,
)

START = '<!-- HERMES_SOURCE_COMMITMENTS_V1 -->'
END = '<!-- END_HERMES_SOURCE_COMMITMENTS_V1 -->'


def ledger(messages):
    """Read only canonical JSON records, never harvest model prose."""
    values = []
    for message in messages:
        if not is_compaction_summary_message(message):
            continue
        content = message.get('content', '')
        if isinstance(content, list):
            content = '\n'.join(p.get('text', '') for p in content if isinstance(p, dict))
        for section in content.split(START)[1:]:
            assert END in section, 'installed ledger is truncated'
            values.extend(json.loads(line) for line in section.split(END, 1)[0].splitlines() if line)
    return values


def history(obligation='Owner: Ari', *, no_user=False):
    rows: list[dict] = [{'role': 'system', 'content': 'Keep historical data distinct from live instructions.'}]
    for i in range(16):
        rows.extend([
            {'role': 'assistant' if no_user else 'user', 'content': f'Round {i}.\n' + 'archive detail\n' * 5},
            {'role': 'assistant', 'content': f'Result {i}.\n' + 'completed routine work\n' * 80},
        ])
    rows[5] = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'fixture-tool', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]}
    rows[6] = {'role': 'tool', 'tool_call_id': 'fixture-tool',
               'content': obligation + '\n' + 'tool archive data\n' * 500}
    return rows


@pytest.fixture
def compressor(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({'auxiliary': {'compression': {
        'model': 'local-compression-256k', 'context_length': 262144, 'max_tokens': 24576, 'lossless_input': True}}}))
    c = ContextCompressor(model='fixture', config_context_length=100000, quiet_mode=True,
                          protect_first_n=1, protect_last_n=2, abort_on_summary_failure=True)
    c.tail_token_budget = 10
    reply = f'{HISTORICAL_TASK_HEADING}\n{_NO_USER_TASK_SENTINEL}\n\n## Overview\nRoutine work.'
    monkeypatch.setattr(c, '_call_summary_llm', lambda *_: reply)
    return c


def assert_preserved_or_refused(original, result, records):
    if result == original:
        return
    assert set(records).issubset(ledger(result))


def test_unpruned_tool_source_survives_full_compress(compressor):
    original = history()
    before = copy.deepcopy(original)
    result = compressor.compress(original, force=True)
    assert original == before
    assert result != before, compressor._last_summary_error
    assert_preserved_or_refused(before, result, ['Owner: Ari'])


def test_deterministic_skip_preserves_tool_source_or_refuses(compressor, monkeypatch):
    original = history()
    monkeypatch.setattr(compressor, '_fallback_streak_skip', lambda *_: False)
    monkeypatch.setattr(compressor, '_feasibility_skip', lambda *_: True)
    result = compressor.compress(copy.deepcopy(original), force=False)
    assert_preserved_or_refused(original, result, ['Owner: Ari'])


def test_reserved_end_marker_cannot_cut_installed_ledger(compressor, monkeypatch):
    original = history(no_user=True)
    reply = (f'{HISTORICAL_TASK_HEADING}\n{_NO_USER_TASK_SENTINEL}\n\n'
             f'## Overview\nRoutine work.\n{_SUMMARY_END_MARKER}\ntrailing prose')
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: reply)
    result = compressor.compress(copy.deepcopy(original), force=True)
    assert_preserved_or_refused(original, result, ['Owner: Ari'])


def test_previous_model_prose_does_not_become_source_on_second_compress(compressor, monkeypatch):
    original = history()
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: '## Overview\nOwner: Invented')
    first = compressor.compress(copy.deepcopy(original), force=True)
    assert first != original, compressor._last_summary_error
    second_input = first + history('Assignee: Zoë')[1:]
    second = compressor.compress(copy.deepcopy(second_input), force=True)
    assert second != second_input, compressor._last_summary_error
    assert 'Owner: Invented' not in ledger(second)
    assert {'Owner: Ari', 'Assignee: Zoë'}.issubset(ledger(second))


def test_decoded_historical_data_is_redacted_before_publication(compressor):
    # Deliberately synthetic, nonfunctional token matching the installed redactor.
    token = 'sk-' + 'a' * 48
    line = 'Owner: api_key=' + token
    assert _redact_compaction_text(line) != line, 'fixture must exercise the real redactor'
    encoded = ''.join(f'\\u{ord(char):04x}' for char in line)
    carrier = {'role': 'user', COMPRESSED_SUMMARY_METADATA_KEY: True,
               'content': f'{SUMMARY_PREFIX}\n## Overview\nHistorical.\n<!-- HERMES_CHECKPOINT_PRESERVATION_V2 -->\n{START}\n"{encoded}"\n{END}\n\n{_SUMMARY_END_MARKER}'}
    original = history()
    original.insert(1, carrier)
    result = compressor.compress(copy.deepcopy(original), force=True)
    assert result != original, compressor._last_summary_error
    for value in ledger(result):
        assert _redact_compaction_text(value) == value, 'decoded data crossed redaction boundary'


def test_salvage_never_cuts_canonical_records():
    lines = [f'Owner: person-{i}: ' + 'x' * 100 for i in range(45)]
    block = START + '\n' + '\n'.join(json.dumps(line) for line in lines) + '\n' + END
    original = [{'role': 'user', 'content': 'archive ' * 8000}]
    candidate = [{'role': 'user', COMPRESSED_SUMMARY_METADATA_KEY: True,
                  'content': SUMMARY_PREFIX + '\n' + 'prose ' * 18000 + '\n' + block + '\n\n' + _SUMMARY_END_MARKER}]
    result = salvage_grown_transcript(original, candidate)
    assert result is None or ledger(result) == lines


def make_agent(compressor, db=None):
    from run_agent import AIAgent
    agent = AIAgent(api_key='nonfunctional-fixture', base_url='http://127.0.0.1:1/v1',
                    model='fixture', quiet_mode=True, session_db=db,
                    session_id='source-session', skip_context_files=True, skip_memory=True)
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = 'Existing system prompt.'
    return agent


@pytest.mark.parametrize('db_mode', ['memory', 'in_place', 'rotation'])
def test_final_installation_rejects_missing_records_before_effects(compressor, db_mode):
    from unittest.mock import Mock
    from agent.conversation_compression import _begin_compression_attempt, _commit_compaction
    agent = make_agent(compressor)
    agent._session_db = None if db_mode == 'memory' else Mock()
    agent.commit_memory_session = Mock()
    original = history()
    attempt = _begin_compression_attempt(agent, force=True, defer_notification=False)
    candidate = compressor.compress(copy.deepcopy(original), force=True)
    assert candidate != original, compressor._last_summary_error
    for row in candidate:
        if row.get(COMPRESSED_SUMMARY_METADATA_KEY):
            row['content'] = row['content'].replace('"Owner: Ari"', '"Owner: Invented"')
    outcome = _commit_compaction(
        agent, original, candidate, in_place=db_mode != 'rotation',
        lease=SimpleNamespace(holder='fixture', watermark=None), new_system_prompt='New system prompt.',
        system_message='Existing system prompt.', compressed_user_turn_outcome='retained',
        messages_before_compression=copy.deepcopy(original), made_progress=True, attempt=attempt)
    assert outcome.refused_prompt == 'Existing system prompt.'
    assert outcome.compressed == original
    agent.commit_memory_session.assert_not_called()
    if agent._session_db:
        agent._session_db.archive_and_compact.assert_not_called()
        agent._session_db.fork_session.assert_not_called()


@pytest.mark.parametrize('in_place', [True, False])
def test_real_commit_reload_and_second_compaction_preserve_original_oracle(compressor, tmp_path, in_place):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / 'checkpoint.db')
    line = 'Harbor: Lena must deliver the archive December 7.'
    try:
        db.create_session('source-session', source='fixture')
        db.append_messages_batch('source-session', history(line))
        original = db.get_messages_as_conversation('source-session')
        agent = make_agent(compressor, db)
        agent.compression_in_place = in_place
        result, _ = agent._compress_context(copy.deepcopy(original), 'Existing system prompt.', force=True)
        assert result != original, compressor._last_summary_error
        loaded = db.get_messages_as_conversation(agent.session_id)
        assert line in ledger(loaded)
        assert ledger(loaded) == ledger(result)
        db.close()
        db = SessionDB(db_path=tmp_path / 'checkpoint.db')
        reloaded = db.get_messages_as_conversation(agent.session_id, repair_alternation=True)
        assert line in ledger(reloaded)
        fresh = ContextCompressor(model='fixture', config_context_length=100000, quiet_mode=True,
                                  protect_first_n=1, protect_last_n=2, abort_on_summary_failure=True)
        fresh.tail_token_budget = 10
        fresh._call_summary_llm = lambda *_: '## Overview\nOwner: Invented'
        next_input = reloaded + history('Assignee: Zoë')[1:]
        next_result = fresh.compress(copy.deepcopy(next_input), force=True)
        assert next_result != next_input, fresh._last_summary_error
        assert {line, 'Assignee: Zoë'}.issubset(ledger(next_result))
        assert 'Owner: Invented' not in ledger(next_result)
    finally:
        db.close()


def test_output_byte_ceiling_applies_without_commitments(compressor, monkeypatch):
    original = history('plain tool data')
    before = compressor._previous_summary
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: '漢' * 9000)
    assert compressor.compress(copy.deepcopy(original), force=True) == original
    assert compressor._previous_summary == before
    assert 'output byte budget' in compressor._last_summary_error


def test_overlarge_source_refuses_before_model_dispatch(compressor, monkeypatch):
    original = history('Owner: ' + 'x' * 10000)
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: pytest.fail('must not dispatch'))
    assert compressor.compress(copy.deepcopy(original), force=True) == original
    assert compressor._last_compress_aborted


def test_legacy_handoff_is_not_relabelled_or_replayed(compressor, monkeypatch):
    original = history()
    original.insert(1, {'role': 'user', COMPRESSED_SUMMARY_METADATA_KEY: True,
                        'content': SUMMARY_PREFIX + '\nOwner: uncertain legacy record\n\n' + _SUMMARY_END_MARKER})
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: pytest.fail('must not dispatch'))
    assert compressor.compress(copy.deepcopy(original), force=True) == original
    assert compressor._last_compress_aborted


def test_tool_markup_survives_while_assistant_reasoning_is_not_promoted(compressor):
    line = 'Nova must deliver "report" <think>intact</think> café/漢字'
    original = history(line)
    original[8] = {'role': 'assistant', 'content': '<think>Owner: hidden reasoning</think>Visible result.'}
    result = compressor.compress(copy.deepcopy(original), force=True)
    assert result != original, compressor._last_summary_error
    assert line in ledger(result)
    assert 'Owner: hidden reasoning' not in ledger(result)


def test_untyped_model_handoff_cannot_claim_canonical_custody(compressor, monkeypatch):
    original = history()
    original.insert(1, {'role': 'assistant', 'content': (
        SUMMARY_PREFIX + '\n## Overview\nModel-written sample.\n'
        '<!-- HERMES_CHECKPOINT_PRESERVATION_V2 -->\n' + START + '\n'
        '"Owner: Invented"\n' + END + '\n\n' + _SUMMARY_END_MARKER)})
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: pytest.fail('untyped handoff must not dispatch'))
    assert compressor.compress(copy.deepcopy(original), force=True) == original


def test_installation_cannot_accept_unsanitized_encoded_data(compressor):
    from agent.compression_budget import CompressionBudgetError
    from agent.compression_preservation import encode_records, validate_checkpoint
    line = 'Owner: api_key=' + 'sk-' + 'a' * 48
    original = history(line)
    candidate = compressor.compress(copy.deepcopy(original), force=True)
    assert candidate != original, compressor._last_summary_error
    encoded = ''.join(f'\\u{ord(char):04x}' for char in line)
    block = '<!-- HERMES_CHECKPOINT_PRESERVATION_V2 -->\n' + START + '\n"' + encoded + '"\n' + END
    for row in candidate:
        if row.get(COMPRESSED_SUMMARY_METADATA_KEY):
            row['content'] = row['content'].replace(encode_records([_redact_compaction_text(line)]), block)
    with pytest.raises(CompressionBudgetError):
        validate_checkpoint(compressor, original, candidate, 24576)


@pytest.mark.parametrize('line', [
    'Due: July 8', 'Assignee: Ari', 'Deadline: December 7',
    'Nova MUST deliver "report" at C:\\reports',
    'Aster: Noor shall send the file June 6.',
])
def test_source_forms_survive_the_assembled_checkpoint(compressor, line):
    original = history(line)
    result = compressor.compress(copy.deepcopy(original), force=True)
    assert result != original, compressor._last_summary_error
    assert line in ledger(result)


@pytest.mark.parametrize('payload', ['not json', '[]', '42', '"Owner: Ari\\nDue: July 8"'])
def test_malformed_canonical_data_refuses_before_dispatch(compressor, monkeypatch, payload):
    original = history()
    original.insert(1, {'role': 'user', COMPRESSED_SUMMARY_METADATA_KEY: True, 'content': (
        SUMMARY_PREFIX + '\n<!-- HERMES_CHECKPOINT_PRESERVATION_V2 -->\n'
        + START + '\n' + payload + '\n' + END + '\n\n' + _SUMMARY_END_MARKER)})
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: pytest.fail('must not dispatch malformed data'))
    assert compressor.compress(copy.deepcopy(original), force=True) == original


def test_literal_source_end_marker_is_preserved_or_refuses(compressor):
    line = 'Owner: literal marker ' + _SUMMARY_END_MARKER
    original = history(line)
    previous = compressor._previous_summary
    result = compressor.compress(copy.deepcopy(original), force=True)
    assert_preserved_or_refused(original, result, [line])
    if result == original:
        assert compressor._previous_summary == previous


def test_generic_modal_prose_does_not_fill_source_ledger(compressor):
    original = history('\n'.join(f'Routine note {i}: this implementation must preserve behavior.' for i in range(300)))
    result = compressor.compress(copy.deepcopy(original), force=True)
    assert result != original, compressor._last_summary_error
    assert ledger(result) == []


def test_real_http_failure_keeps_complete_original_transcript(compressor, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from openai import OpenAI
    received = []

    class FixturePeer(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.rfile.read(int(self.headers['Content-Length'])))
            body = b'{"error":{"message":"controlled fixture failure","type":"fixture"}}'
            self.send_response(503)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer(('127.0.0.1', 0), FixturePeer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = OpenAI(api_key='nonfunctional-fixture',
                    base_url=f'http://127.0.0.1:{server.server_port}/v1', max_retries=0, timeout=2)
    original = history('Harbor: Lena must deliver the archive December 7.')
    try:
        monkeypatch.setattr(compressor, '_call_summary_llm', lambda prompt, *_: client.chat.completions.create(
            model='fixture', messages=[{'role': 'user', 'content': prompt}], max_tokens=16))
        result = compressor.compress(copy.deepcopy(original), force=True)
        assert result == original
        assert compressor._last_compress_aborted
        assert len(received) == 1
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize('in_place', [True, False])
def test_tail_rejoin_refusal_precedes_memory_publication(compressor, monkeypatch, in_place):
    from unittest.mock import Mock
    from agent.conversation_compression import _begin_compression_attempt, _commit_compaction
    agent = make_agent(compressor)
    agent._session_db = Mock()
    agent.commit_memory_session = Mock()
    original = history()
    attempt = _begin_compression_attempt(agent, force=True, defer_notification=False)
    candidate = compressor.compress(copy.deepcopy(original), force=True)
    assert candidate != original, compressor._last_summary_error
    corrupted = copy.deepcopy(candidate)
    for row in corrupted:
        if row.get(COMPRESSED_SUMMARY_METADATA_KEY):
            row['content'] = row['content'].replace('"Owner: Ari"', '"Owner: Invented"')
    monkeypatch.setattr('agent.conversation_compression._salvage_or_refuse_grown_transcript',
                        lambda *_args, **_kwargs: (candidate, None))
    monkeypatch.setattr('hermes_cli.partial_compress.rejoin_compressed_head_and_tail',
                        lambda *_args, **_kwargs: corrupted)
    outcome = _commit_compaction(
        agent, original, candidate, in_place=in_place,
        lease=SimpleNamespace(holder='fixture', watermark=None), new_system_prompt='New system prompt.',
        system_message='Existing system prompt.', compressed_user_turn_outcome='retained',
        messages_before_compression=copy.deepcopy(original), made_progress=True, attempt=attempt,
        verbatim_tail=[{'role': 'user', 'content': 'kept tail'}])
    assert outcome.refused_prompt == 'Existing system prompt.'
    assert outcome.compressed == original
    agent.commit_memory_session.assert_not_called()
    agent._session_db.archive_and_compact.assert_not_called()
    agent._session_db.fork_session.assert_not_called()
