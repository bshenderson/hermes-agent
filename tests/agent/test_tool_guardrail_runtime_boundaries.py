"""GXTD-712 real turn disposition and tool-round observation boundaries."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

from tests.agent.test_tool_call_guardrail_runtime import (
    _make_agent, _mock_response, _mock_tool_call, _hard_stop_config,
    _seed_exact_failures,
)


def test_block_then_plugin_loop_reports_terminal_disposition_without_next_turn():
    agent = _make_agent("read_file", "plugin_empty", config=_hard_stop_config())
    calls = [_mock_tool_call("read_file", '{"path":"missing"}', "blocked")]
    calls += [_mock_tool_call("plugin_empty", json.dumps({"request": i}), f"plugin-{i}") for i in range(6)]

    def provider(**kwargs):
        _seed_exact_failures(agent, "read_file", {"path": "missing"})
        return _mock_response("", "tool_calls", calls)

    agent.client.chat.completions.create.side_effect = provider
    # Unknown plugins are sequential barriers in the real segment planner.
    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=lambda name, args, **kw: (None, {} if name == "plugin_empty" else args)),
        patch("model_tools.handle_function_call", return_value="") as dispatch,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Exercise one bounded tool round")
    assert agent.client.chat.completions.create.call_count == 1
    assert dispatch.call_count == 5
    assert agent._tool_guardrails.halt_decision.action == "halt"
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert result["guardrail"]["action"] == "halt"
    assert result["guardrail"]["code"] == "identical_call_streak_halt"


def test_concurrent_dispatched_results_survive_halt_but_later_segment_cannot_dispatch():
    from agent.tool_executor import execute_tool_calls_segmented

    agent = _make_agent("plugin_empty", "patch", config=_hard_stop_config())
    calls = [_mock_tool_call("plugin_empty", '{}', f"parallel-{i}") for i in range(6)]
    later = _mock_tool_call("patch", '{}', "forbidden")
    rendezvous = threading.Barrier(6, timeout=10)
    executed = []
    lock = threading.Lock()

    def dispatch(name, args, task_id, **kwargs):
        with lock:
            executed.append(kwargs["tool_call_id"])
        rendezvous.wait()
        assert agent._tool_guardrails.halt_decision is None
        return ""

    messages = []
    # Explicit segment plan qualifies executor boundaries, not plugin parallel eligibility.
    with patch("model_tools.handle_function_call", side_effect=dispatch):
        execute_tool_calls_segmented(
            agent, SimpleNamespace(tool_calls=calls + [later]), messages, "bounded",
            segments=[("parallel", calls), ("sequential", [later])],
        )
    assert sorted(executed) == sorted(tc.id for tc in calls)
    assert [m["tool_call_id"] for m in messages] == [tc.id for tc in calls] + [later.id]
    assert agent._tool_guardrail_halt_decision.action == "halt"
    assert "identical_call_streak_halt" in messages[-1]["content"]


def test_block_without_escalation_still_stops_provider_turn():
    agent = _make_agent("read_file", "plugin_empty", config=_hard_stop_config())

    def provider(**kwargs):
        _seed_exact_failures(agent, "read_file", {"path": "missing"})
        return _mock_response("", "tool_calls", [
            _mock_tool_call("read_file", '{"path":"missing"}', "blocked"),
            _mock_tool_call("plugin_empty", '{}', "allowed"),
        ])

    agent.client.chat.completions.create.side_effect = provider
    with (
        patch("model_tools.handle_function_call", return_value="") as dispatch,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("One block, no terminal escalation")
    assert dispatch.call_count == 1
    assert agent.client.chat.completions.create.call_count == 1
    assert result["guardrail"]["action"] == "block"
    assert result["turn_exit_reason"] == "guardrail_halt"


def test_late_failure_can_replace_controller_identity_not_first_runtime_terminal():
    agent = _make_agent("plugin_empty", "terminal", config=_hard_stop_config(
        hard_stop_after={"same_tool_failure": 1, "idempotent_no_progress": 3},
    ))
    for i in range(3):
        agent._append_guardrail_observation("plugin_empty", {}, "", failed=False, tool_call_id=str(i))
    first = agent._tool_guardrail_halt_decision
    assert first.action == "halt" and first.code == "identical_call_streak_halt"
    agent._append_guardrail_observation("plugin_failure", {}, '{"error":"bad"}', failed=True)
    late = agent._tool_guardrails.halt_decision
    assert late.action == "halt" and late.code == "same_tool_failure_halt"
    assert late is not first
    assert agent._tool_guardrail_halt_decision is first
    assert agent._tool_guardrails.before_call("patch", {}) is late
    agent._append_guardrail_observation("patch", {}, '{"success":true}', failed=False)
    assert agent._tool_guardrails.before_call("terminal", {}) is late
