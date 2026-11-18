"""Execution admission budgets must not masquerade as replay detection (GXTD-850)."""

from agent.tool_guardrails import ToolCallGuardrailController


def test_distinct_successful_execution_exhausts_budget_without_loop_diagnosis():
    controller = ToolCallGuardrailController()
    for i in range(8):
        args = {"code": f"print({i})"}
        assert controller.before_call("execute_code", args).action == "allow"
        controller.after_call("execute_code", args, str(i), failed=False)
    decision = controller.before_call("execute_code", {"code": "print(9)"})
    assert decision.action == "block"
    assert "loop" not in decision.message.lower()
    assert decision.count == 8


def test_execution_arguments_cannot_change_budget_accounting():
    controller = ToolCallGuardrailController()
    for i in range(8):
        assert controller.before_call(
            "execute_code", {"code": f"print({i})", "action": "list", "tasks": [1, 2]}
        ).action == "allow"
    assert controller.before_call("execute_code", {"code": "print(9)"}).action == "block"
