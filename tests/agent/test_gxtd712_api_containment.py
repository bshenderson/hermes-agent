import json

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


def test_api_empty_plugin_recall_stops_before_sixth_dispatch():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig.from_mapping({}, platform="api_server")
    )
    args = {"query": "hermes architecture diagram layered stack render png svg mermaid plantuml"}
    result = json.dumps({"success": True, "workflows": [], "count": 0})
    for _ in range(5):
        assert controller.before_call("media_workflow_recall", args).action == "allow"
        controller.after_call("media_workflow_recall", args, result, failed=False)
        controller.observe_call("media_workflow_recall", args, result, failed=False)
    assert controller.halt_decision is not None
    assert controller.before_call("media_workflow_recall", args).should_halt


def test_api_explicit_policy_can_preserve_warning_only():
    config = ToolCallGuardrailConfig.from_mapping(
        {"api_hard_stop_enabled": False}, platform="api_server"
    )
    assert not config.hard_stop_enabled


def test_attended_cli_policy_unchanged():
    for platform in ("cli", "tui", "desktop", "acp"):
        assert not ToolCallGuardrailConfig.from_mapping({}, platform=platform).hard_stop_enabled


def test_api_alternating_plugin_empty_results_halt():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig.from_mapping({}, platform="api_server")
    )
    for _ in range(5):
        for name in ("media_workflow_recall", "custom_readonly_recall"):
            controller.observe_call(name, {"query": "diagram"}, "[]", failed=False)
    assert controller.halt_decision is not None
    assert controller.before_call("write_file", {"path": "must-not-dispatch"}).should_halt


def test_api_polling_and_changing_results_remain_allowed():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig.from_mapping({}, platform="api_server")
    )
    for index in range(20):
        controller.observe_call("process_manage", {"action": "poll"}, "running", failed=False)
        controller.observe_call("terminal", {"command": "test"}, str(index), failed=False)
    assert controller.halt_decision is None


def test_explicit_hard_stop_override_and_reset():
    assert not ToolCallGuardrailConfig.from_mapping(
        {"hard_stop_enabled": False}, platform="api_server"
    ).hard_stop_enabled
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig.from_mapping({}, platform="api_server")
    )
    for _ in range(5):
        controller.observe_call("plugin_recall", {}, "[]", failed=False)
    assert controller.before_call("another_tool", {}).should_halt
    controller.reset_for_turn()
    assert controller.before_call("another_tool", {}).allows_execution
