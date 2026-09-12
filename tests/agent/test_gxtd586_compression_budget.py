"""Every byte reaches a bounded summary call, including middle/tool content."""
import importlib.util
import pytest


def test_map_prompts_do_not_ask_empty_segments_for_full_checkpoints():
    from agent.compression_budget import summarize_bounded
    prompts = []
    def call(prompt):
        prompts.append(prompt)
        return 'NO_DURABLE_FACTS'
    summarize_bounded('filler ' * 1000, lambda text: 'FINAL:' + text, call,
                      context_length=4096, output_tokens=256, safety_tokens=1024,
                      build_chunk_prompt=lambda text: 'FACTS_ONLY:' + text)
    assert all(p.startswith('FACTS_ONLY:') for p in prompts[:-1])
    assert prompts[-1].startswith('FINAL:')
    assert all(len(p.encode()) <= 2816 for p in prompts)


def test_chunking_covers_unicode_input_without_omitting_middle():
    assert importlib.util.find_spec('agent.compression_budget') is not None
    from agent.compression_budget import summarize_bounded
    source = 'early\n' + 'record 漢字🙂\n' * 900 + 'MIDDLE_FACT\n' + 'archive\n' * 900 + 'late'
    seen = []
    def build(text): return 'Summarize:\n' + text + '\nEnd.'
    def call(prompt):
        assert len(prompt.encode()) + 400 + 256 <= 4096
        seen.append(prompt)
        return 'short summary'
    result = summarize_bounded(source, build, call, context_length=4096, output_tokens=400, safety_tokens=256)
    assert result == 'short summary'
    assert source == ''.join(s[len('Summarize:\n'):-len('\nEnd.')] for s in seen[:-1])
    assert any('MIDDLE_FACT' in s for s in seen)


def test_failed_chunk_never_returns_partial_summary():
    from agent.compression_budget import summarize_bounded
    calls = []
    def call(prompt):
        calls.append(prompt)
        if len(calls) == 2: raise RuntimeError('backend unavailable')
        return 'partial'
    with pytest.raises(RuntimeError, match='backend unavailable'):
        summarize_bounded('x' * 10000, lambda x: x, call, context_length=4096,
                          output_tokens=400, safety_tokens=256)
