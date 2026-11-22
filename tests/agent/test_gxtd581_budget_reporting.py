"""Cumulative tool budgets must not be labeled retry failures."""
import pytest
from agent.tool_guardrails import ToolGuardrailDecision
from run_agent import AIAgent


@pytest.mark.parametrize('tool,code', [
    ('execute_code', 'loop_execute_code_cap'), ('web_search', 'loop_web_search_cap'),
    ('delegate_task', 'loop_subagent_cap')])
def test_cumulative_budget_halt_does_not_claim_repeated_failure(tool, code):
    decision = ToolGuardrailDecision(action='halt', code=code, tool_name=tool, count=8)
    text = object.__new__(AIAgent)._toolguard_controlled_halt_response(decision)
    assert 'repeated non-progressing' not in text
    assert 'cumulative' in text and '8' in text
    assert all(label in text for label in ('Outcome:', 'Impact:', 'Caveats/blockers:', 'Next step:'))


def test_actual_failure_halt_retains_its_specific_cause():
    decision = ToolGuardrailDecision(action='halt', code='same_tool_failure_halt',
        tool_name='terminal', count=5, message='Stopped terminal: it failed 5 times this turn.')
    text = object.__new__(AIAgent)._toolguard_controlled_halt_response(decision)
    assert decision.message in text
