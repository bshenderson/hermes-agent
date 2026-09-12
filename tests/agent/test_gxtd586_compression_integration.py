"""Lossless mode is opt-in and traverses real config/serialization/summary code."""
import json
from types import SimpleNamespace

from agent.context_compressor import ContextCompressor


def enable(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({'auxiliary': {'compression': {
        'model': 'local-compression-256k', 'context_length': 262144,
        'max_tokens': 12000, 'lossless_input': True}}}))


def test_full_middle_and_tool_arguments_reach_bounded_calls(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch)
    c = ContextCompressor(model='main', config_context_length=262144, quiet_mode=True)
    long = 'prefix\n' + 'unimportant entry\n' * 22000 + 'MIDDLE_KEEP_THIS' + 'archive\n' * 22000 + 'tail'
    turns = [{'role': 'user', 'content': long}, {'role': 'assistant', 'content': 'used tool',
             'tool_calls': [{'function': {'name': 'example', 'arguments': long}}]}]
    prompts = []
    def call(prompt, started):
        prompts.append(prompt)
        assert len(prompt.encode()) + 12000 + 1024 <= 262144
        return '## Goal\nPreserve MIDDLE_KEEP_THIS from the synthetic transcript.'
    monkeypatch.setattr(c, '_call_summary_llm', call)
    serialized = c._serialize_for_summary(turns, lossless=True)
    assert serialized.count('MIDDLE_KEEP_THIS') == 2
    assert '[truncated]' not in serialized
    result = c._generate_summary(turns)
    assert result and 'MIDDLE_KEEP_THIS' in result
    assert len(prompts) > 1
    assert sum(p.count('MIDDLE_KEEP_THIS') for p in prompts) >= 2


def test_failed_chunk_aborts_without_replacing_previous_summary(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch)
    c = ContextCompressor(model='main', config_context_length=262144, quiet_mode=True)
    previous = 'Existing checkpoint must survive'
    c._previous_summary = previous
    calls = []
    def call(prompt, started):
        calls.append(prompt)
        if len(calls) == 2: raise RuntimeError('capacity refused')
        return 'incomplete partial'
    monkeypatch.setattr(c, '_call_summary_llm', call)
    assert c._generate_summary([{'role': 'user', 'content': 'entry\n' * 100000}]) is None
    assert c._previous_summary == previous
    assert c._last_summary_truncated_failure
