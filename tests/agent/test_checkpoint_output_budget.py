"""GXTD-641: optional lean excerpts share the complete checkpoint byte envelope."""
import copy
import json

import pytest

from agent import context_compressor as cc
from agent.context_compressor import (
    ContextCompressor,
    COMPRESSED_SUMMARY_METADATA_KEY,
    _handoff_only_content,
)
from agent.compression_budget import CompressionBudgetError
from agent.compression_preservation import (
    _bounded_user_excerpts, augment_summary, decode_records, encode_records, frame_summary,
)


@pytest.fixture
def complete_case(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({'auxiliary': {'compression': {
        'model': 'local-compression-256k', 'context_length': 262144,
        'max_tokens': 24576, 'lossless_input': True,
    }}}))
    obligations = {
        3: 'Cedar: Maya must submit the report October 12.',
        14: 'Quartz: Felix must review the plan November 19.',
        25: 'Harbor: Lena must deliver the archive December 7.',
    }
    messages = [{'role': 'system', 'content': 'Summarize operational obligations faithfully.'}]
    for index in range(30):
        filler = '\n'.join(f'Archive row {index}-{row}: routine status record; no durable obligation.'
                           for row in range(220))
        messages.append({'role': 'user' if index % 2 == 0 else 'assistant',
                         'content': obligations.get(index, '') + '\n' + filler})
    messages.append({'role': 'user', 'content': 'Keep the Cedar, Quartz and Harbor obligations intact.'})

    compressor = ContextCompressor(model='local-compression-256k', quiet_mode=True,
                                   config_context_length=262144, protect_first_n=1,
                                   protect_last_n=3, abort_on_summary_failure=True)
    monkeypatch.setattr(compressor, '_summary_modality', 'text', raising=False)
    compressor._session_id = 'fixture-session'
    monkeypatch.setattr(compressor, '_call_summary_llm',
                        lambda *_: '## Overview\n' + 'Recorded operational work.\n' * 240)
    return messages, compressor, obligations


def test_complete_input_quotes_share_checkpoint_budget(complete_case):
    messages, compressor, obligations = complete_case
    original = copy.deepcopy(messages)
    result = compressor.compress(messages, force=True)
    assert result != original, compressor._last_summary_error
    assert len(result) < len(original)
    assert not compressor._last_compress_aborted
    assert messages == original
    assert result[-1]['content'] == original[-1]['content']
    handoffs = [_handoff_only_content(row['content']) for row in result
                if row.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert handoffs
    assert sum(len(text.encode()) for text in handoffs) <= 24576
    records = [record for text in handoffs for record in decode_records(text)]
    assert set(records) == set(obligations.values())


def test_reservation_includes_canonical_head_and_tail_records(complete_case):
    messages, compressor, obligations = complete_case
    head = 'Owner: protected head ' + 'h' * 3000
    tail = 'Owner: protected tail ' + 't' * 3000
    messages[1]['content'] = head + '\n' + messages[1]['content']
    messages[-1]['content'] = tail + '\n' + messages[-1]['content']
    original = copy.deepcopy(messages)
    result = compressor.compress(messages, force=True)
    assert len(result) < len(original), compressor._last_summary_error
    assert messages == original
    handoffs = [_handoff_only_content(row['content']) for row in result
                if row.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert sum(len(text.encode()) for text in handoffs) <= 24576
    assert {record for text in handoffs for record in decode_records(text)} == set(obligations.values()) | {head, tail}


@pytest.mark.parametrize('budget', [0, 1, 170, 400, 1024, 4096])
@pytest.mark.parametrize('body', ['漢🧭\n' * 1000, 'line\n' * 1000, 'ordinary text'])
def test_excerpts_bound_rendered_utf8_bytes(budget, body):
    turns = [{'role': 'user', 'content': 'Older request'},
             {'role': 'user', 'content': 'Newest request\n' + body}]
    before = copy.deepcopy(turns)
    constant = cc._LEAN_USER_MESSAGES_BUDGET_CHARS
    result = _bounded_user_excerpts(turns, budget)
    assert len(result.encode('utf-8')) <= budget
    assert result.encode('utf-8').decode('utf-8') == result
    assert turns == before
    assert cc._LEAN_USER_MESSAGES_BUDGET_CHARS == constant
    if result:
        assert 'Newest request' in result
        assert 'Quoted excerpts only' in result
        assert result == cc._redact_compaction_text(result)


@pytest.mark.parametrize('optional_bytes', [0, 100, 400, 2000])
def test_optional_sections_leave_room_for_full_framing(optional_bytes):
    compressor = ContextCompressor(model='fixture', quiet_mode=True)
    compressor._session_id = 'fixture-archive-pointer'
    records = ['Owner: Ari']
    body = '## Overview\nRequired body.'
    framing = cc.SUMMARY_PREFIX + '\n\n\n' + encode_records(records) + '\n\n' + cc._SUMMARY_END_MARKER
    ceiling = len((framing + body).encode()) + optional_bytes
    turns = [{'role': 'user', 'content': 'Newest request\n' + '漢\n' * 1000}]
    summary = augment_summary(compressor, body, turns, records, ceiling)
    final = frame_summary(compressor, summary, records, ceiling) + '\n\n' + cc._SUMMARY_END_MARKER
    assert len(final.encode()) <= ceiling
    assert decode_records(final) == records
    assert body in summary
    if optional_bytes == 0:
        assert summary == body
    if optional_bytes == 2000:
        assert 'fixture-archive-pointer' in summary


def test_unbounded_legacy_augmentation_keeps_existing_sections():
    compressor = ContextCompressor(model='fixture', quiet_mode=True)
    compressor._session_id = 'fixture-session'
    turns = [{'role': 'user', 'content': 'Keep src/feature.py unchanged.'}]
    body = '## Overview\nRequired body.'
    expected = (body + cc._redact_compaction_text(cc._build_anchor_index(turns))
                + cc._redact_compaction_text(cc._build_verbatim_user_section(turns))
                + cc._build_recovery_footer('fixture-session', len(turns)))
    assert compressor._augment_summary_lean(body, turns) == expected
    compressor.tail_mode = 'legacy'
    assert compressor._augment_summary_lean(body, turns, [], 100) == body


@pytest.mark.parametrize('body', ['[SKILL_PRUNED: reload this skill]\nRequired task.',
                                 cc._LEAN_USER_MESSAGES_HEADING + '\nModel-provided task quotation.'])
def test_mandatory_provenance_and_optional_anchor_pointer_survive(body):
    compressor = ContextCompressor(model='fixture', quiet_mode=True)
    compressor._session_id = 'fixture-session'
    turns = [{'role': 'user', 'content': 'Preserve /srv/project/config.yaml and GXTD-641.\n' + '漢' * 5000}]
    records = ['Owner: Ari']
    summary = augment_summary(compressor, body, turns, records, 5000)
    framed = frame_summary(compressor, summary, records, 5000) + '\n\n' + cc._SUMMARY_END_MARKER
    assert summary.startswith(body)
    assert cc._redact_compaction_text(cc._build_anchor_index(turns)) in summary
    assert 'fixture-session' in summary
    assert len(framed.encode()) <= 5000
    assert decode_records(framed) == records


def test_missing_source_output_allowance_refuses():
    compressor = ContextCompressor(model='fixture', quiet_mode=True)
    with pytest.raises(CompressionBudgetError, match='output budget is required'):
        augment_summary(compressor, 'Required body.', [], records=[])


def test_required_body_overflow_still_refuses(complete_case, monkeypatch):
    messages, compressor, _ = complete_case
    original = copy.deepcopy(messages)
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: '漢' * 9000)
    result = compressor.compress(messages, force=True)
    assert result == original and messages == original
    assert compressor._last_compress_aborted
    assert 'output byte budget' in compressor._last_summary_error
    assert compressor._previous_summary is None


@pytest.mark.parametrize('in_place', [True, False])
def test_real_commit_reload_keeps_custody_with_bounded_excerpts(complete_case, tmp_path, monkeypatch, in_place):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    messages, compressor, obligations = complete_case
    path = tmp_path / 'checkpoint.db'
    db = SessionDB(db_path=path)
    try:
        db.create_session('source-session', source='fixture')
        db.append_messages_batch('source-session', messages)
        original = db.get_messages_as_conversation('source-session')
        agent = AIAgent(api_key='nonfunctional-fixture', base_url='http://127.0.0.1:1/v1',
                        model='fixture', quiet_mode=True, session_db=db,
                        session_id='source-session', skip_context_files=True, skip_memory=True)
        agent.context_compressor = compressor
        agent._compression_feasibility_checked = True
        agent._cached_system_prompt = 'Existing system prompt.'
        agent.compression_in_place = in_place
        result, _ = agent._compress_context(copy.deepcopy(original), 'Existing system prompt.', force=True)
        assert len(result) < len(original), compressor._last_summary_error
        active_session = agent.session_id
    finally:
        db.close()
    db = SessionDB(db_path=path)
    try:
        reloaded = db.get_messages_as_conversation(active_session, repair_alternation=True)
        handoffs = [_handoff_only_content(row['content']) for row in reloaded
                    if row.get(COMPRESSED_SUMMARY_METADATA_KEY)]
        assert sum(len(text.encode()) for text in handoffs) <= 24576
        assert {record for text in handoffs for record in decode_records(text)} == set(obligations.values())
        assert reloaded[-1]['content'] == original[-1]['content']
        fresh = ContextCompressor(model='local-compression-256k', quiet_mode=True,
                                  config_context_length=262144, protect_first_n=1,
                                  protect_last_n=3, abort_on_summary_failure=True)
        monkeypatch.setattr(fresh, '_call_summary_llm', lambda *_: '## Overview\nRecorded operational work.')
        additional = [{'role': 'assistant' if index % 2 == 0 else 'user',
                       'content': ('Assignee: Zoë\n' if index == 1 else '') + 'More archived detail.\n' * 700}
                      for index in range(20)]
        next_input = reloaded + additional
        next_result = fresh.compress(copy.deepcopy(next_input), force=True)
        assert len(next_result) < len(next_input), fresh._last_summary_error
        next_handoffs = [_handoff_only_content(row['content']) for row in next_result
                         if row.get(COMPRESSED_SUMMARY_METADATA_KEY)]
        assert sum(len(text.encode()) for text in next_handoffs) <= 24576
        expected = set(obligations.values()) | {'Assignee: Zoë'}
        assert {record for text in next_handoffs for record in decode_records(text)} == expected
    finally:
        db.close()
