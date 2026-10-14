"""GXTD-712 public-controller recovery and containment lifecycle regressions."""

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController
import pytest


def test_recoverable_block_does_not_suppress_unrelated_empty_plugin_loop():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True, exact_failure_block_after=2,
        same_tool_failure_halt_after=99, no_progress_block_after=3,
    ))
    args = {"path": "missing"}
    for _ in range(2):
        assert controller.before_call("read_file", args).allows_execution
        controller.after_call("read_file", args, '{"error":"missing"}', failed=True)
        controller.observe_call("read_file", args, '{"error":"missing"}', failed=True)
    assert controller.before_call("read_file", args).action == "block"
    for _ in range(3):
        assert controller.before_call("plugin_empty", {}).allows_execution
        controller.after_call("plugin_empty", {}, "", failed=False)
        controller.observe_call("plugin_empty", {}, "", failed=False)
    halt = controller.halt_decision
    assert halt is not None and halt.action == "halt"
    assert halt.code == "identical_call_streak_halt"
    assert controller.before_call("patch", {"path": "repair"}) is halt


@pytest.mark.parametrize("platform", ["api_server", " API_SERVER "])
@pytest.mark.parametrize("policy, expected", [
    ({}, True),
    ({"api_hard_stop_enabled": False}, False),
    ({"non_interactive_hard_stop_enabled": False}, True),
    ({"hard_stop_enabled": False}, False),
    ({"hard_stop_enabled": True, "api_hard_stop_enabled": False}, True),
    ({"hard_stop_enabled": " off ", "api_hard_stop_enabled": True}, False),
    ({"hard_stop_enabled": " ON ", "api_hard_stop_enabled": False}, True),
    ({"hard_stop_enabled": "invalid", "api_hard_stop_enabled": False}, False),
])
def test_api_policy_precedence(platform, policy, expected):
    assert ToolCallGuardrailConfig.from_mapping(policy, platform=platform).hard_stop_enabled is expected


@pytest.mark.parametrize("tool", ["process_manage", "plugin_get_result", "plugin_poll"])
def test_polling_full_lifecycle_wait_completion_and_turn_reset(tool):
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True, no_progress_block_after=3,
    ))
    args = {"action": "poll", "session_id": "job"}
    for result in ["running" * 100] * 8 + ["done"]:
        assert controller.before_call(tool, args).allows_execution
        assert not controller.after_call(tool, args, result, failed=False).should_halt
        observation = controller.observe_call(tool, args, result, failed=False)
        assert observation.notice is None
        assert controller.halt_decision is None
    controller.reset_for_turn()
    assert controller.before_call(tool, args).allows_execution


@pytest.mark.parametrize("mixed", [False, True])
def test_polling_cycle_full_lifecycle_after_recoverable_block(mixed):
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True, exact_failure_block_after=1,
        same_tool_failure_halt_after=99, no_progress_block_after=3,
    ))
    controller.after_call("read_file", {"path": "missing"}, '{"error":"missing"}', failed=True)
    block = controller.before_call("read_file", {"path": "missing"})
    assert block.action == "block"
    tools = ["plugin_poll", "plugin_empty" if mixed else "plugin_get_result"]
    for _ in range(3):
        for tool in tools:
            assert controller.before_call(tool, {}).allows_execution
            assert not controller.after_call(tool, {}, "", failed=False).should_halt
            controller.observe_call(tool, {}, "", failed=False)
    if mixed:
        halt = controller.halt_decision
        assert halt.action == "halt" and halt.code == "identical_cycle_halt"
        # Observation of a different loop cannot replace the terminal decision.
        for _ in range(4):
            controller.observe_call("other_plugin", {}, "same")
        assert controller.before_call("patch", {}) is halt
    else:
        assert controller.halt_decision is block
        assert controller.before_call("patch", {}).allows_execution


def test_call_local_block_allows_mutation_then_retry():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True, exact_failure_block_after=1,
    ))
    args = {"command": "run-test"}
    assert controller.before_call("terminal", args).allows_execution
    controller.after_call("terminal", args, '{"exit_code":1}', failed=True)
    assert controller.before_call("terminal", args).action == "block"
    assert controller.before_call("patch", {}).allows_execution
    controller.after_call("patch", {}, '{"success":true}', failed=False)
    assert controller.before_call("terminal", args).allows_execution
