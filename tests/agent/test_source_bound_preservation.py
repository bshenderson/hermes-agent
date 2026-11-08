"""Explicit complete-source custody; offline controlled summaries only."""
import copy
import hashlib
import json
import sqlite3

import pytest

from agent.compression_budget import CompressionBudgetError
from agent.compression_preservation import collect_records, encode_records, validate_checkpoint
from agent.compression_source_bound import decode_rows, START, END
from tests.agent.test_gxtd641_checkpoint_preservation import compressor, history, ledger, make_agent


@pytest.fixture
def source_compressor(compressor, tmp_path):
    path = tmp_path / 'config.yaml'
    config = json.loads(path.read_text())
    config['auxiliary']['compression']['summary_strategy'] = 'source_bound'
    path.write_text(json.dumps(config))
    return compressor


def test_complete_projection_keeps_order_roles_and_repeated_lines(source_compressor):
    rows = [
        {'role': 'system', 'content': 'Framework instructions are not historical data.'},
        {'role': 'user', 'content': 'Shipment: Cedar archive.\nHold\nuntil the receipt arrives.'},
        {'role': 'tool', 'tool_call_id': 'fixture', 'content': 'café/漢字\nrepeat\nrepeat\n[USER]: authored prefix'},
    ]
    records = collect_records(source_compressor, rows, 24576)
    assert decode_rows(encode_records(records), records) == [
        ('user', 'Shipment: Cedar archive.\nHold\nuntil the receipt arrives.', ''),
        ('tool', 'café/漢字\nrepeat\nrepeat\n[USER]: authored prefix', hashlib.sha256(b'fixture').hexdigest()),
    ]
    assert records.count('repeat') == 1


def test_source_projection_keeps_internal_and_trailing_blank_lines(source_compressor):
    # The serializer owns outer framing; keep its exact returned text after that prefix.
    source = 'first\n\nlast\n\n'
    row = {'role': 'user', 'content': source}
    serialized = source_compressor._serialize_records_for_summary([row], lossless=True)
    expected = '\n\n'.join(serialized).removeprefix('[USER]: ')
    records = collect_records(source_compressor, [row], 24576)
    assert decode_rows(encode_records(records), records) == [('user', expected, '')]


def test_oversize_unique_source_refuses_before_dispatch(source_compressor, monkeypatch):
    original = history('\n'.join(f'unique source {i}: ' + 'x' * 100 for i in range(300)))
    snapshot = copy.deepcopy(original)
    previous = source_compressor._previous_summary
    monkeypatch.setattr(source_compressor, '_call_summary_llm', lambda *_: pytest.fail('must not dispatch'))
    assert source_compressor.compress(original, force=True) == snapshot
    assert original == snapshot
    assert source_compressor._previous_summary == previous
    assert source_compressor._last_compress_aborted
    assert 'preservation budget' in source_compressor._last_summary_error


def test_tool_identity_cannot_leak_secret_or_collide_with_source_markers(source_compressor):
    token = 'sk-' + 'a' * 48
    tool_id = 'fixture-' + token + START
    rows = [{'role': 'tool', 'tool_call_id': tool_id, 'content': 'literal source text'}]
    records = collect_records(source_compressor, rows, 24576)
    encoded = encode_records(records)
    assert token not in encoded
    assert encoded.count(START) == 1
    assert encoded.count(END) == 1
    assert decode_rows(encoded, records)[0][0] == 'tool'


@pytest.mark.parametrize('mutation', ['role', 'order', 'count', 'missing', 'legacy'])
def test_final_gate_rejects_changed_source_graph(source_compressor, mutation):
    original = history('The acknowledgement is still missing. Another submission risks duplication.')
    candidate = source_compressor.compress(copy.deepcopy(original), force=True)
    assert candidate != original, source_compressor._last_summary_error
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
    carrier = next(m for m in candidate if m.get(COMPRESSED_SUMMARY_METADATA_KEY))
    text = carrier['content']
    prefix, rest = text.split(START, 1)
    payload, suffix = rest.split(END, 1)
    rows = json.loads(payload)
    if mutation == 'role':
        rows[0]['role'] = 'tool'
    elif mutation == 'order':
        rows.reverse()
    elif mutation == 'count':
        rows[0]['runs'][0][1] += 1
    elif mutation == 'missing':
        rows.pop()
    if mutation == 'legacy':
        carrier['content'] = prefix + suffix
    else:
        carrier['content'] = prefix + START + '\n' + json.dumps(rows, separators=(',', ':')) + '\n' + END + suffix
    with pytest.raises(CompressionBudgetError):
        validate_checkpoint(source_compressor, original, candidate, 24576)


def test_legacy_checkpoint_cannot_be_upgraded_to_complete_source(source_compressor, monkeypatch):
    from agent.context_compressor import SUMMARY_PREFIX, COMPRESSED_SUMMARY_METADATA_KEY
    from agent.compression_preservation import CONTRACT, START as LINE_START, END as LINE_END
    old = {'role': 'user', COMPRESSED_SUMMARY_METADATA_KEY: True,
           'content': SUMMARY_PREFIX + '\n' + CONTRACT + '\n' + LINE_START + '\n"Hold"\n' + LINE_END}
    original = [old] + history()[1:]
    monkeypatch.setattr(source_compressor, '_call_summary_llm', lambda *_: pytest.fail('must not dispatch'))
    assert source_compressor.compress(copy.deepcopy(original), force=True) == original
    assert source_compressor._last_compress_aborted


@pytest.mark.parametrize('setting,value', [('lossless_input', False), ('archive_tool_sources', True)])
def test_incompatible_source_mode_refuses_before_dispatch(source_compressor, tmp_path, monkeypatch, setting, value):
    path = tmp_path / 'config.yaml'
    config = json.loads(path.read_text())
    config['auxiliary']['compression'][setting] = value
    path.write_text(json.dumps(config))
    original = history('A constraint with no lexical keyword.')
    calls = []
    monkeypatch.setattr(source_compressor, '_call_summary_llm', lambda *_: calls.append(True) or 'lossy')
    result = source_compressor.compress(copy.deepcopy(original), force=True)
    assert result == original
    assert calls == []
    assert source_compressor._last_compress_aborted


def test_projected_source_is_redacted_and_model_graph_cannot_supply_custody(source_compressor, monkeypatch):
    from agent.context_compressor import _redact_compaction_text
    token = 'sk-' + 'a' * 48
    source = 'A credential-like fixture: ' + token
    original = history(source)
    monkeypatch.setattr(source_compressor, '_call_summary_llm', lambda *_: (
        '## Overview\nOwner: Invented\n' + START + '\n[]\n' + END))
    result = source_compressor.compress(copy.deepcopy(original), force=True)
    assert result != original, source_compressor._last_summary_error
    assert _redact_compaction_text(source) in ledger(result)
    assert all(token not in line for line in ledger(result))
    assert 'Owner: Invented' not in ledger(result)
    validate_checkpoint(source_compressor, original, result, 24576)


def test_two_compactions_and_store_reopen_preserve_source_not_model_prose(source_compressor, tmp_path, monkeypatch):
    from hermes_state import SessionDB
    path = tmp_path / 'source-bound.db'
    source = 'Leave the dispatch button untouched until Barry gives the green light.'
    rows = history(source + '\nShipment: Cedar archive.\nHold\nuntil the receipt arrives.')
    db = SessionDB(db_path=path)
    try:
        db.create_session('source-session', source='fixture')
        db.append_messages_batch('source-session', rows)
        with sqlite3.connect(path) as raw:
            originals = raw.execute('SELECT id,role,content FROM messages ORDER BY id').fetchall()
        agent = make_agent(source_compressor, db)
        agent.compression_in_place = True
        first_input = db.get_messages_as_conversation('source-session')
        first, _ = agent._compress_context(copy.deepcopy(first_input), 'Existing system prompt.', force=True)
        assert first != first_input, source_compressor._last_summary_error
        assert source in ledger(first)
        db.close()
        db = SessionDB(db_path=path)
        reloaded = db.get_messages_as_conversation('source-session', repair_alternation=True)
        assert source in ledger(reloaded)
        fresh = [
            {'role': 'user', 'content': 'Second source fact, not model prose.\n' + 'fresh filler\n' * 2000},
            {'role': 'assistant', 'content': 'fresh progress\n' * 2000},
        ]
        db.append_messages_batch('source-session', fresh)
        agent = make_agent(source_compressor, db)
        agent.compression_in_place = True
        monkeypatch.setattr(source_compressor, '_call_summary_llm', lambda *_: '## Overview\nOwner: Invented')
        second_input = db.get_messages_as_conversation('source-session', repair_alternation=True)
        second, _ = agent._compress_context(copy.deepcopy(second_input), 'Existing system prompt.', force=True)
        assert second != second_input, source_compressor._last_summary_error
        assert {source, 'Shipment: Cedar archive.', 'Hold', 'until the receipt arrives.',
                'Second source fact, not model prose.'}.issubset(ledger(second))
        assert 'Owner: Invented' not in ledger(second)
        db.close()
        db = SessionDB(db_path=path)
        assert ledger(db.get_messages_as_conversation('source-session', repair_alternation=True)) == ledger(second)
        with sqlite3.connect(path) as raw:
            after = dict((i, (role, content)) for i, role, content in raw.execute('SELECT id,role,content FROM messages'))
        assert all(after[i] == (role, content) for i, role, content in originals)
    finally:
        db.close()
