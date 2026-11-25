"""GXTD-921: resource ceilings must not invent a retry/non-progress diagnosis."""
import json
from unittest.mock import patch

import pytest

from agent.tool_guardrails import ToolGuardrailDecision
from run_agent import AIAgent
from tests.agent.test_tool_call_guardrail_runtime import (
    _make_agent, _mock_response, _mock_tool_call,
)


@pytest.mark.parametrize('cap', [8, 32])
def test_distinct_successful_execution_budget_reaches_truthful_streamed_final(cap):
    agent = _make_agent('execute_code', max_iterations=40, config={
        'tool_loop_guardrails': {'loop_caps': {'max_execute_code_calls': cap}},
    })
    agent.client.chat.completions.create.side_effect = [
        _mock_response('', 'tool_calls', [
            _mock_tool_call('execute_code', json.dumps({'code': f'print({i})'}), f'c{i}')
        ]) for i in range(cap + 1)
    ]
    deltas = []
    agent.stream_delta_callback = deltas.append
    agent._disable_streaming = True
    with (
        patch('model_tools.handle_function_call', side_effect=[
            json.dumps({'status': 'success', 'output': str(i), 'exit_code': 0})
            for i in range(cap)
        ]) as dispatch,
        patch.object(agent, '_persist_session'),
        patch.object(agent, '_save_trajectory'),
        patch.object(agent, '_cleanup_task_resources'),
    ):
        result = agent.run_conversation('Perform distinct calculations')
    assert dispatch.call_count == cap
    assert result['guardrail']['code'] == 'loop_execute_code_cap'
    assert result['guardrail']['count'] == cap
    assert result['turn_exit_reason'] == 'guardrail_halt'
    final = result['final_response']
    assert 'budget' in final.lower()
    assert 'kept running' not in final
    assert 'without making progress' not in final
    assert final in deltas
    assert result['guardrail']['message'] in final
    assert any('execution_budget_warning' in str(m.get('content', '')) for m in result['messages'])


def test_larger_execution_budget_keeps_identical_loop_halt():
    agent = _make_agent('execute_code', max_iterations=40, config={
        'tool_loop_guardrails': {
            'hard_stop_enabled': True,
            'hard_stop_after': {'idempotent_no_progress': 3},
            'loop_caps': {'max_execute_code_calls': 32},
        },
    })
    agent.client.chat.completions.create.side_effect = [
        _mock_response('', 'tool_calls', [
            _mock_tool_call('execute_code', '{"code":"print(1)"}', f'c{i}')
        ]) for i in range(5)
    ]
    with (
        patch('model_tools.handle_function_call', return_value='{"status":"success","output":"1"}') as dispatch,
        patch.object(agent, '_persist_session'),
        patch.object(agent, '_save_trajectory'),
        patch.object(agent, '_cleanup_task_resources'),
    ):
        result = agent.run_conversation('Exercise identical-call containment')
    assert dispatch.call_count == 3
    assert result['guardrail']['code'] == 'identical_call_streak_halt'
    assert result['guardrail']['message'] in result['final_response']


@pytest.mark.parametrize('code,message', [
    ('web_search_synthesis_redirect', 'Skipped web_search: synthesis policy requested an answer.'),
    ('loop_web_search_cap', 'Blocked web_search: the per-turn search budget is exhausted.'),
    ('loop_subagent_cap', 'Blocked delegate_task: the per-turn subagent budget is exhausted.'),
    ('repeated_exact_failure_block', 'Blocked read_file: the same call failed 3 times.'),
    ('identical_cycle_halt', 'Stopped terminal: identical two-call cycle repeated 3 times.'),
])
def test_final_preserves_actual_reason_instead_of_inventing_generic_retry(code, message):
    decision = ToolGuardrailDecision(action='block', code=code, message=message, count=3)
    final = AIAgent._toolguard_controlled_halt_response(None, decision)
    assert message in final
    assert 'without making progress' not in final
    assert 'send `continue`' not in final
