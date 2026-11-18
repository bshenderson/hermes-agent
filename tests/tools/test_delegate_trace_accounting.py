"""Retained transcript rows are not unique tool dispatches (GXTD-850)."""
from copy import deepcopy

from tools.delegate_tool_child_run import _build_tool_trace


def call(identity, name='read_file'):
    return {'role': 'assistant', 'tool_calls': [
        {'id': identity, 'function': {'name': name, 'arguments': '{}'}}]}


def test_repeated_transcript_rows_do_not_multiply_identified_calls():
    messages = [call('a'), {'role': 'tool', 'tool_call_id': 'a', 'content': 'ok'},
                call('a'), call('b')]
    before = deepcopy(messages)
    trace = _build_tool_trace(messages)
    assert len(trace) == 2
    assert trace[0]['status'] == 'ok'
    assert 'status' not in trace[1]
    assert messages == before


def test_foreign_identified_result_does_not_attach_to_latest_call():
    trace = _build_tool_trace([call('a'),
                              {'role': 'tool', 'tool_call_id': 'foreign', 'content': 'ok'}])
    assert 'status' not in trace[0]


def test_unidentified_calls_are_not_collapsed_by_equal_arguments():
    trace = _build_tool_trace([call(None), call(None),
                              {'role': 'tool', 'content': 'legacy result'}])
    assert len(trace) == 2
    assert 'status' not in trace[0]
    assert trace[1]['status'] == 'ok'


def test_conflicting_identity_retains_both_calls_without_mispairing():
    trace = _build_tool_trace([call('a'), call('a', 'terminal'),
                              {'role': 'tool', 'tool_call_id': 'a', 'content': 'ok'}])
    assert len(trace) == 2
    assert all(row['identity_conflict'] for row in trace)
    assert all('status' not in row for row in trace)


def test_result_consumer_preserves_partial_and_uses_unique_trace():
    from types import SimpleNamespace
    from tools.delegate_tool_child_run import _build_result_entry, _SchemaOutcome

    result = {'final_response': 'Partial findings; artifact still needs review',
              'completed': False, 'api_calls': 50, 'turn_exit_reason': 'max_iterations',
              'messages': [call('a'), call('a')]}
    entry = _build_result_entry(SimpleNamespace(), result, 0, 1.0, _SchemaOutcome(None, None, [], 0))
    assert entry['summary'] == result['final_response']
    assert entry['task_outcome'] == 'budget_exhausted'
    assert entry['truncated'] is True
    assert entry['api_calls'] == 50
    assert len(entry['tool_trace']) == 1


def test_decoded_argument_mapping_does_not_break_result_delivery():
    message = call('decoded')
    message['tool_calls'][0]['function']['arguments'] = {'path': 'fixture.py'}
    trace = _build_tool_trace([message, deepcopy(message)])
    assert len(trace) == 1
