"""GXTD-850: nested receipts are observations, not dispatch guarantees."""
import json
from types import SimpleNamespace

import pytest

from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry


def call(identity, tool='execute_code'):
    return {'role': 'assistant', 'tool_calls': [
        {'id': identity, 'function': {'name': tool, 'arguments': '{}'}}]}


def receipt(identity, value):
    return {'role': 'tool', 'tool_call_id': identity, 'content': json.dumps(value)}


def result(messages):
    return _build_result_entry(SimpleNamespace(), {
        'completed': True, 'final_response': 'fixture', 'api_calls': 4,
        'messages': messages}, 0, 12.5, _SchemaOutcome(None, None, [], 0))


def test_result_reports_nested_receipts_once_and_keeps_unmeasured_time_unknown():
    messages = [call('a'), receipt('a', {'status': 'success', 'tool_calls_made': 3}),
                call('a'), receipt('a', {'status': 'success', 'tool_calls_made': 3}),
                call('b'), receipt('b', {'status': 'error', 'tool_calls_made': 0})]
    entry = result(messages)
    accounting = entry['execution_accounting']
    assert accounting['api_calls'] == 4
    assert accounting['transcript_tool_calls'] == 2
    assert accounting['nested_tool_calls_reported'] == 3
    assert accounting['execute_code_receipts_observed'] == 2
    assert accounting['execute_code_receipts_unknown'] == 0
    assert accounting['elapsed_seconds'] == 12.5
    assert accounting['pending_seconds'] is None
    assert accounting['execution_seconds'] is None
    assert accounting['claim_type'] == 'transcript_observation'


@pytest.mark.parametrize('value', [None, [], {}, {'tool_calls_made': True},
                         {'tool_calls_made': -1}, {'tool_calls_made': '3'}])
def test_invalid_receipts_are_unknown_not_zero(value):
    accounting = result([call('a'), receipt('a', value)])['execution_accounting']
    assert accounting['nested_tool_calls_reported'] is None
    assert accounting['execute_code_receipts_unknown'] == 1


def test_missing_foreign_and_unidentified_receipts_cannot_supply_counts():
    messages = [call('a'), receipt('foreign', {'tool_calls_made': 50}),
                call(None), receipt(None, {'tool_calls_made': 10})]
    accounting = result(messages)['execution_accounting']
    assert accounting['nested_tool_calls_reported'] is None
    assert accounting['execute_code_receipts_unknown'] == 2


def test_conflicting_call_or_receipt_identity_excludes_ambiguous_counts():
    messages = [call('a'), receipt('a', {'tool_calls_made': 3}), call('a', 'terminal'),
                call('b'), receipt('b', {'tool_calls_made': 4}),
                receipt('b', {'tool_calls_made': 5})]
    accounting = result(messages)['execution_accounting']
    assert accounting['nested_tool_calls_reported'] is None
    assert accounting['execute_code_receipts_unknown'] == 2


def test_partial_failure_keeps_known_subset_without_claiming_total():
    entry = _build_result_entry(SimpleNamespace(), {
        'failed': True, 'final_response': 'partial',
        'messages': [call('a'), receipt('a', {'tool_calls_made': 2}), call('b')]},
        0, 4, _SchemaOutcome(None, None, [], 0))
    accounting = entry['execution_accounting']
    assert entry['status'] == 'failed'
    assert accounting['api_calls'] is None
    assert accounting['nested_tool_calls_reported'] == 2
    assert accounting['execute_code_receipts_unknown'] == 1


def test_missing_transcript_on_failure_is_not_zero_work():
    from tools.delegate_tool_child_run import _ChildRun
    run = _ChildRun(SimpleNamespace(), None, 0, 'fixture', None, None)
    entry = run.attach_worktree({'status': 'failed', 'duration_seconds': 7})
    accounting = entry['execution_accounting']
    assert accounting['transcript_tool_calls'] is None
    assert accounting['execute_code_receipts_unknown'] is None
    assert accounting['api_calls'] is None
    assert accounting['elapsed_seconds'] == 7


def test_missing_messages_and_non_json_output_remain_unknown():
    entry = _build_result_entry(SimpleNamespace(), {'final_response': 'partial'},
                               0, 1, _SchemaOutcome(None, None, [], 0))
    assert entry['execution_accounting']['transcript_tool_calls'] is None
    entry = result([call('a'), {'role': 'tool', 'tool_call_id': 'a',
                              'content': 'truncated output, tool_calls_made=99'}])
    assert entry['execution_accounting']['nested_tool_calls_reported'] is None
