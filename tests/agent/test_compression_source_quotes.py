"""GXTD-743 source-quoted compression: model selection is not fact authority."""
import importlib.util
import json
import pytest


def test_quote_selection_is_exact_and_rejects_invention():
    assert importlib.util.find_spec('agent.compression_source_quotes'), 'source-quoted compression missing'
    from agent.compression_source_quotes import extract_summary
    from agent.compression_budget import CompressionBudgetError
    source = '[USER]: Atlas: Priya must submit the plan November 8.\nOutcome UNKNOWN; do not replay.'
    def call(prompt):
        assert source in prompt
        return json.dumps({'excerpts': ['Atlas: Priya must submit the plan November 8.', 'Outcome UNKNOWN; do not replay.']})
    result = extract_summary(source, call, context_length=4096, output_tokens=1000, required=['Atlas: Priya must submit the plan November 8.'])
    assert 'Source-backed historical excerpts' in result
    assert 'Outcome UNKNOWN; do not replay.' in result
    with pytest.raises(CompressionBudgetError, match='not present'):
        extract_summary(source, lambda p: json.dumps({'excerpts': ['Priya submitted the plan.']}), context_length=4096, output_tokens=1000)
    with pytest.raises(CompressionBudgetError, match='required'):
        extract_summary(source, lambda p: '{"excerpts":[]}', context_length=4096, output_tokens=1000, required=['Atlas: Priya must submit the plan November 8.'])


def test_real_compressor_selects_source_quotes_only_when_opted_in(monkeypatch):
    from agent import auxiliary_client as aux
    from agent.context_compressor import ContextCompressor
    original_config = aux._get_auxiliary_task_config
    config = {'model': 'local-compression-256k', 'provider': 'llama-secondary',
              'lossless_input': True, 'context_length': 262144, 'max_tokens': 24576,
              'summary_strategy': 'source_quotes'}
    monkeypatch.setattr(aux, '_get_auxiliary_task_config', lambda task: config if task == 'compression' else original_config(task))
    c = ContextCompressor(model='local-compression-256k', config_context_length=262144, quiet_mode=True)
    calls = []
    def call(prompt, *args, **kwargs):
        calls.append((prompt, kwargs))
        return '{"excerpts":["Atlas: Priya must submit the plan November 8."]}'
    monkeypatch.setattr(c, '_call_summary_llm', call)
    summary = c._generate_summary([{'role': 'user', 'content': 'Atlas: Priya must submit the plan November 8.'}],
                                  focus_topic='Archive row: no durable obligation.')
    assert summary and 'Source-backed historical excerpts' in summary
    assert calls[0][1].get('response_format', {}).get('type') == 'json_schema'
    assert 'Atlas: Priya must submit the plan November 8.' in summary
    assert 'Archive row' not in calls[0][0]


def test_adjacent_sentence_quotes_cover_one_canonical_commitment():
    from agent.compression_source_quotes import extract_summary
    line = 'Cedar: Maya must submit October 12. Quartz: Felix must review November 19.'
    boundary = line.index('. ') + 1
    result = extract_summary(line, lambda prompt: json.dumps({'excerpts': [line[:boundary], line[boundary + 1:]]}),
                             context_length=4096, output_tokens=1600, required=[line])
    assert 'Cedar' in result and 'Quartz' in result


def test_custody_candidate_discloses_selection_omission_without_inventing_prose():
    from agent.compression_source_quotes import extract_summary
    from agent.compression_budget import CompressionBudgetError
    selected = 'Inez must review /srv/Atlas-Blue/config.v2.yaml November 21.'
    line = selected + ' Do not normalize Atlas-Blue.'
    findings = {}
    result = extract_summary(line, lambda p: json.dumps({'excerpts': [selected]}),
                             context_length=4096, output_tokens=1600, required=[line],
                             custody=True, findings=findings)
    assert line in result
    assert findings['model_omitted_required_records'] == [line]
    with pytest.raises(CompressionBudgetError):
        extract_summary(line, lambda p: '{"excerpts":[]}', context_length=4096, output_tokens=1600,
                        required=[line], custody=True)


def test_repeated_quote_cannot_guess_correction_order():
    from agent.compression_source_quotes import extract_summary
    from agent.compression_budget import CompressionBudgetError
    source = 'Maya owns Atlas.\nCorrection: Lena owns Atlas.\nMaya owns Atlas.'
    with pytest.raises(CompressionBudgetError, match='ambiguous'):
        extract_summary(source, lambda p: '{"excerpts":["Maya owns Atlas."]}',
                        context_length=4096, output_tokens=1000)


def test_repeated_quote_in_separate_chunks_is_also_ambiguous():
    from agent.compression_source_quotes import extract_summary
    from agent.compression_budget import CompressionBudgetError
    source = 'Maya owns Atlas.\n' + 'routine status\n' * 600 + 'Maya owns Atlas.'
    def call(prompt):
        return json.dumps({'excerpts': ['Maya owns Atlas.'] if 'Maya owns Atlas.' in prompt else []})
    with pytest.raises(CompressionBudgetError, match='ambiguous'):
        extract_summary(source, call, context_length=4096, output_tokens=1000)


def test_second_compaction_uses_canonical_custody_not_prior_model_prose(monkeypatch):
    import copy
    from agent import auxiliary_client as aux
    from agent.context_compressor import ContextCompressor, COMPRESSED_SUMMARY_METADATA_KEY
    from tests.agent.test_gxtd641_checkpoint_preservation import history, ledger
    old = aux._get_auxiliary_task_config
    config = {'model': 'local-compression-256k', 'provider': 'llama-secondary',
              'lossless_input': True, 'context_length': 262144, 'max_tokens': 24576,
              'summary_strategy': 'source_quotes_custody'}
    monkeypatch.setattr(aux, '_get_auxiliary_task_config', lambda task: config if task == 'compression' else old(task))
    c = ContextCompressor(model='fixture', config_context_length=100000, quiet_mode=True,
                          protect_first_n=1, protect_last_n=2, abort_on_summary_failure=True)
    c.tail_token_budget = 10
    def call(prompt, *a, **kw):
        assert 'INVENTED PRIOR ACTION' not in prompt
        return json.dumps({'excerpts': [q for q in ('Owner: Ari', 'Assignee: Zoë') if q in prompt]})
    monkeypatch.setattr(c, '_call_summary_llm', call)
    original = history()
    first = c.compress(copy.deepcopy(original), force=True)
    assert first != original, c._last_summary_error
    for message in first:
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY):
            message['content'] = message['content'].replace('## Source-backed historical excerpts',
                'INVENTED PRIOR ACTION\n\n## Source-backed historical excerpts')
    second_input = first + history('Assignee: Zoë')[1:]
    second = c.compress(copy.deepcopy(second_input), force=True)
    assert second != second_input, c._last_summary_error
    assert {'Owner: Ari', 'Assignee: Zoë'}.issubset(ledger(second))
    assert not any('INVENTED PRIOR ACTION' in item for item in ledger(second))


def test_real_config_opt_in_and_invalid_selection_preserve_full_history(tmp_path, monkeypatch):
    import copy
    from agent.auxiliary_client import _get_auxiliary_task_config
    from agent.context_compressor import ContextCompressor
    from tests.agent.test_gxtd641_checkpoint_preservation import history
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({'auxiliary': {'compression': {
        'model': 'local-compression-256k', 'provider': 'llama-secondary', 'context_length': 262144,
        'max_tokens': 24576, 'lossless_input': True, 'summary_strategy': 'source_quotes_custody'}}}))
    assert _get_auxiliary_task_config('compression')['summary_strategy'] == 'source_quotes_custody'
    c = ContextCompressor(model='fixture', config_context_length=100000, quiet_mode=True,
                          protect_first_n=1, protect_last_n=2, abort_on_summary_failure=True)
    monkeypatch.setattr(c, '_call_summary_llm', lambda *a, **kw: '{"excerpts":["INVENTED COMPLETED ACTION"]}')
    original = history()
    result = c.compress(copy.deepcopy(original), force=True)
    assert c._last_compress_aborted
    assert result == original
