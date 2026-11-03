"""Opt-in source byte caps preserve complete input and refusal semantics."""
import json
import pytest
from agent.compression_source_quotes import extract_summary
from agent.compression_budget import CompressionBudgetError


def test_cap_covers_entire_unicode_source_in_order():
    source = ''.join(f'Line {i}: café moss stone.\n' for i in range(100))
    seen = []
    def call(prompt):
        chunk = prompt.split('<source>\n')[1].split('\n</source>\n')[0]
        seen.append(chunk)
        return json.dumps({'excerpts': [chunk.splitlines()[0]]})
    extract_summary(source, call, context_length=131072, output_tokens=24576,
                    source_chunk_bytes=512)
    assert ''.join(seen) == source
    assert len(seen) > 1
    assert all(len(chunk.encode()) <= 512 for chunk in seen)


def test_cap_does_not_enable_all_empty_custody_rescue():
    with pytest.raises(CompressionBudgetError, match='required'):
        extract_summary('Owner: Mira.\n' + 'filler\n' * 200,
                        lambda p: '{"excerpts":[]}', context_length=131072,
                        output_tokens=24576, required=['Owner: Mira.'], custody=True,
                        source_chunk_bytes=512)


def test_configured_summary_passes_opt_in_cap(monkeypatch):
    from types import SimpleNamespace
    from agent import compression_source_quotes as quotes
    from agent import auxiliary_client as aux
    monkeypatch.setattr(aux, '_get_auxiliary_task_config', lambda task: {
        'summary_strategy': 'source_quotes_custody', 'source_chunk_bytes': 512})
    source = ''.join(f'Line {i}: stone.\n' for i in range(100))
    monkeypatch.setattr(quotes, '_canonical_source', lambda *args: source)
    seen = []
    def call(prompt, *args, **kwargs):
        chunk = prompt.split('<source>\n')[1].split('\n</source>\n')[0]
        seen.append(chunk)
        return json.dumps({'excerpts': [chunk.splitlines()[0]]})
    c = SimpleNamespace(_call_summary_llm=call,
                        _transcript_has_real_user_turn=lambda turns: True)
    quotes.configured_summary(c, [], source,
        {'context_length': 131072, 'output_tokens': 24576}, 0, [])
    assert ''.join(seen) == source and len(seen) > 1
    assert max(len(s.encode()) for s in seen) <= 512


@pytest.mark.parametrize('cap', [True, 0, -1, 255, '8192', 1.5])
def test_invalid_cap_refuses_before_dispatch(cap):
    def call(prompt):
        pytest.fail('invalid cap dispatched')
    with pytest.raises(CompressionBudgetError, match='chunk'):
        extract_summary('source', call, context_length=131072,
                        output_tokens=24576, source_chunk_bytes=cap)
