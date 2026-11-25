"""GXTD-921: count admitted work, never rejected requests or invented loops."""
import pytest

from agent.tool_guardrails import (
    LoopCapConfig, ToolCallGuardrailConfig, ToolCallGuardrailController,
)


def test_denied_replay_does_not_consume_execution_admission_budget():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    args = {'code': 'fails()'}
    for _ in range(controller.config.exact_failure_block_after):
        controller.after_call('execute_code', args, '{"error":"boom"}', failed=True)
    assert controller.before_call('execute_code', args).code == 'repeated_exact_failure_block'
    assert controller._turn_execute_code_count == 0


def test_subagent_batch_cannot_overshoot_remaining_capacity():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        loop_caps=LoopCapConfig(max_subagents=2),
    ))
    assert controller.before_call('delegate_task', {'tasks': [{}]}).allows_execution
    decision = controller.before_call('delegate_task', {'tasks': [{}, {}]})
    assert decision.code == 'loop_subagent_cap'
    assert decision.count == 1
    assert controller._turn_subagent_count == 1
    assert controller.before_call('delegate_task', {'action': 'list'}).allows_execution


def test_synthesis_redirect_counts_admitted_searches_not_rejected_requests():
    controller = ToolCallGuardrailController()
    for i in range(4):
        assert controller.before_call('web_search', {'query': str(i)}).allows_execution
        controller.after_call('web_search', {'query': str(i)}, '{"data":{"web":[1]}}', failed=False)
    for i in range(4, 6):
        decision = controller.before_call('web_search', {'query': str(i)})
        assert decision.code == 'web_search_synthesis_redirect'
        assert decision.count == 4
    assert controller._turn_web_search_count == 4


@pytest.mark.parametrize('name,cap_field,args', [
    ('execute_code', 'max_execute_code_calls', {'code': 'print(1)'}),
    ('web_search', 'max_web_searches', {'query': 'fact'}),
    ('delegate_task', 'max_subagents', {'tasks': [{}]}),
])
def test_budget_message_does_not_infer_runaway_loop(name, cap_field, args):
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        loop_caps=LoopCapConfig(**{cap_field: 1}),
    ))
    assert controller.before_call(name, args).allows_execution
    decision = controller.before_call(name, args)
    assert not decision.allows_execution
    assert 'runaway' not in decision.message
    assert 'budget' in decision.message


def test_execution_budget_warns_with_remaining_capacity_before_abrupt_stop():
    controller = ToolCallGuardrailController()
    for i in range(6):
        args = {'code': f'print({i})'}
        assert controller.before_call('execute_code', args).allows_execution
        decision = controller.after_call('execute_code', args, str(i), failed=False)
    assert decision.action == 'warn'
    assert decision.code == 'execution_budget_warning'
    assert '2' in decision.message
    assert not decision.should_halt
