"""GXTD-728 real merged-default policy regression."""
from copy import deepcopy

from agent.tool_guardrails import ToolCallGuardrailConfig
from hermes_cli.config_defaults import DEFAULT_CONFIG
import json
import pytest
from agent.display import _detect_tool_failure
from agent.tool_guardrails import classify_tool_failure


@pytest.mark.parametrize("status", ["denied", "blocked", "failed", "unknown"])
@pytest.mark.parametrize("suffix", ["", "\n[hermes note: repeated call]"])
def test_typed_non_success_is_failure_in_display_and_controller(status, suffix):
    result = json.dumps({"status": status, "reason": "synthetic_refusal"}) + suffix
    assert _detect_tool_failure("media_file", result)[0]
    assert classify_tool_failure("media_file", result)[0]


def test_merged_defaults_enable_api_containment_without_changing_cli():
    policy = deepcopy(DEFAULT_CONFIG["tool_loop_guardrails"])
    assert ToolCallGuardrailConfig.from_mapping(policy, platform="api_server").hard_stop_enabled
    assert not ToolCallGuardrailConfig.from_mapping(policy, platform="cli").hard_stop_enabled
    policy["hard_stop_enabled"] = False
    assert not ToolCallGuardrailConfig.from_mapping(policy, platform="api_server").hard_stop_enabled
    policy["hard_stop_enabled"] = True
    assert ToolCallGuardrailConfig.from_mapping(policy, platform="cli").hard_stop_enabled


@pytest.mark.parametrize("platform, override, expected", [
    ("api_server", "", True), ("cli", "", False),
    ("api_server", "tool_loop_guardrails:\n  hard_stop_enabled: false\n", False),
    ("cli", "tool_loop_guardrails:\n  hard_stop_enabled: true\n", True),
])
def test_actual_loader_and_constructor(tmp_path, monkeypatch, platform, override, expected):
    from hermes_cli.config import load_config_readonly
    from tests.agent.test_tool_call_guardrail_runtime import _make_agent

    path = tmp_path / "config.yaml"
    path.write_text(override, encoding="utf-8")
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: path)
    config = load_config_readonly()
    agent = _make_agent("media_file", platform=platform, config=config)
    assert agent._tool_guardrails.config.hard_stop_enabled is expected


def test_typed_refusal_real_turn_stops_without_another_provider_call():
    from unittest.mock import patch
    from tests.agent.test_tool_call_guardrail_runtime import _make_agent, _mock_response, _mock_tool_call

    agent = _make_agent("media_file", "media_file_vision", platform="api_server", config=deepcopy(DEFAULT_CONFIG))
    calls = [_mock_tool_call("media_file" if i % 2 else "media_file_vision", json.dumps({"path": str(i)}), f"refusal-{i}") for i in range(20)]
    agent.client.chat.completions.create.return_value = _mock_response("", "tool_calls", calls)
    refusal = json.dumps({"status": "denied", "reason": "synthetic_input_required"})

    with (
        patch("model_tools.handle_function_call", return_value=refusal) as dispatch,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Synthetic non-sensitive refusal containment")
    assert dispatch.call_count == 15
    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert result["final_response"]
    assert result["guardrail"]["code"] in {"repeated_exact_failure_block", "identical_cycle_halt", "same_tool_failure_halt"}


@pytest.mark.parametrize("result", [None, "not json", "[]", '{"status": []}', '{"status": {}}', '{"status":"completed"}', '{"status":"running"}'])
def test_nonterminal_or_malformed_typed_result_is_not_a_terminal_failure(result):
    from agent.tool_result_classification import typed_tool_non_success
    assert not typed_tool_non_success(result)


def test_unknown_media_operations_remain_effect_capable():
    from agent.tool_result_classification import tool_may_have_side_effect
    assert tool_may_have_side_effect("media_file")
    assert tool_may_have_side_effect("media_file_vision")
