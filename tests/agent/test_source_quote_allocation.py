"""Selector receives its actual rendered-handoff allocation, never a larger guard."""
import json
import re
import pytest
from agent import compression_source_quotes as q
from agent.compression_budget import CompressionBudgetError


def test_prompt_names_real_quote_budget_and_fixed_custody_reserve():
    source = 'Owner: Mira must review KESTREL.\nDo not send.\nRoutine data.'
    required = ['Owner: Mira must review KESTREL.', 'Do not send.']
    seen = []
    def call(prompt):
        seen.append(prompt)
        assert 'UTF-8 bytes' in prompt
        assert 'handoff limit: 1000 bytes' in prompt
        assert 'canonical custody records are reserved separately' in prompt
        return json.dumps({'excerpts': [required[0]]})
    result = q.extract_summary(source, call, context_length=8192, output_tokens=2000,
                              required=required, custody=True)
    assert len(result.encode()) <= 1000
    assert len(seen) == 1
    assert all(record in result for record in required)


def test_chunk_quotas_share_one_global_handoff_limit():
    source = '\n'.join(f'Unique row {n}: routine data.' for n in range(500))
    quotas = []
    def call(prompt):
        match = re.search(r'quote allocation: (\d+) UTF-8 bytes', prompt)
        assert match, 'selector must see its bounded allocation'
        quotas.append(int(match[1]))
        chunk = prompt.split('<source>\n',1)[1].rsplit('\n</source>',1)[0]
        return json.dumps({'excerpts': [chunk.splitlines()[0]]})
    result = q.extract_summary(source, call, context_length=8192, output_tokens=2000)
    assert len(quotas) > 1
    assert sum(quotas) <= 1000
    assert len(result.encode()) <= 1000


def test_impossible_custody_allocation_refuses_before_model_dispatch():
    source = 'Owner: ' + 'x'*1000
    def no_dispatch(_):
        pytest.fail('impossible mandatory custody must refuse before dispatch')
    with pytest.raises(CompressionBudgetError, match='handoff allocation'):
        q.extract_summary(source, no_dispatch, context_length=4096, output_tokens=1000,
                          required=[source], custody=True)


def test_model_ignoring_quote_budget_still_hits_original_guard():
    source = '\n'.join(f'Unique row {n}: some filler value.' for n in range(100))
    with pytest.raises(CompressionBudgetError, match='handoff allocation'):
        q.extract_summary(source, lambda p: json.dumps({'excerpts': source.splitlines()}),
                          context_length=8192, output_tokens=1000)
