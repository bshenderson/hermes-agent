"""Bounded answer-only synthesis: execution limits and response acceptance."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agent.iteration_budget import IterationBudget
from agent.web_synthesis import UNAVAILABLE, synthesize_collected_web


def make_agent():
    transport = SimpleNamespace(normalize_response=lambda response: response)
    agent = SimpleNamespace(
        _interrupt_requested=False, max_iterations=8, api_mode="chat_completions",
        iteration_budget=IterationBudget(8), _current_api_request_id="original",
        _fire_stream_delta=Mock(), interim_assistant_callback=Mock(),
        _build_api_kwargs=lambda messages: {"messages": messages, "tools": [{"name": "web_search"}]},
        _touch_activity=Mock(), _interruptible_api_call=Mock(), provider="local", model="unchanged",
        _get_transport=lambda: transport,
    )
    return agent


@pytest.mark.parametrize("gate", ["interrupted", "round_limit", "shared_budget", "unsupported_transport"])
def test_no_synthesis_when_execution_gate_is_closed(gate):
    agent = make_agent()
    count = 4
    if gate == "interrupted":
        agent._interrupt_requested = True
    elif gate == "round_limit":
        count = agent.max_iterations
    elif gate == "shared_budget":
        for _ in range(8):
            assert agent.iteration_budget.consume()
    else:
        agent.api_mode = "codex_app_server"
    with patch("agent.relay_llm.execute_current") as wire:
        assert synthesize_collected_web(agent, [], count) == (UNAVAILABLE, False, False)
    wire.assert_not_called()
    assert agent._last_web_synthesis_diagnostic["reason"] == {
        "interrupted": "interrupted", "round_limit": "iteration_limit",
        "shared_budget": "shared_budget", "unsupported_transport": "unsupported_transport",
    }[gate]


def test_exception_diagnostics_do_not_leak_payload(caplog):
    agent = make_agent()
    with (
        patch("agent.web_synthesis._iteration_summary_api_messages", return_value=[]),
        patch("agent.relay_llm.execute_current", side_effect=RuntimeError("private-token-example")),
        patch("agent.relay_llm.complete_logical_call"),
        caplog.at_level("WARNING", logger="agent.web_synthesis"),
    ):
        assert synthesize_collected_web(agent, [], 4) == (UNAVAILABLE, False, True)
    assert agent._last_web_synthesis_diagnostic["reason"] == "exception"
    assert "private-token-example" not in caplog.text


def test_unavailable_does_not_claim_retrievals_failed():
    assert "failed or timed-out pages" not in UNAVAILABLE


@pytest.mark.parametrize("finish,text,tools,accepted", [
    ("stop", "Bounded comparison; AMD extraction timed out.", [], True),
    ("length", "Truncated comparison", [], False),
    ("tool_calls", "Do more work", [{"name": "web_search"}], False),
    ("stop", "", [], False),
])
def test_response_is_buffered_and_accepted_only_when_complete(finish, text, tools, accepted):
    agent = make_agent()
    delta, interim = agent._fire_stream_delta, agent.interim_assistant_callback
    messages = [{"role": "tool", "content": "collected content and FETCH_TIMEOUT"}]

    def complete(kwargs, callback, **metadata):
        assert kwargs["tool_choice"] == "none"
        assert kwargs["tools"] == [{"name": "web_search"}]
        assert kwargs["messages"][:-1] == messages
        assert kwargs["messages"][-1]["role"] == "user"
        assert "Do not call tools" in kwargs["messages"][-1]["content"]
        assert "unavailable" in kwargs["messages"][-1]["content"]
        agent._fire_stream_delta("unaccepted partial output")
        assert agent.interim_assistant_callback is None
        return SimpleNamespace(content=text, tool_calls=tools, finish_reason=finish)

    with (
        patch("agent.web_synthesis._iteration_summary_api_messages", return_value=messages),
        patch("agent.web_synthesis.record_response_usage") as usage,
        patch("agent.relay_llm.execute_current", side_effect=complete) as wire,
        patch("agent.relay_llm.complete_logical_call") as finish_call,
    ):
        result = synthesize_collected_web(agent, messages, 4)
    assert result == (text if accepted else UNAVAILABLE, accepted, True)
    assert agent.iteration_budget.used == 1
    assert wire.call_count == usage.call_count == finish_call.call_count == 1
    assert usage.call_args.kwargs["api_call_count"] == 5
    assert agent._fire_stream_delta is delta
    assert agent.interim_assistant_callback is interim
    assert agent._current_api_request_id == "original"
    delta.assert_not_called()
    interim.assert_not_called()


@pytest.mark.parametrize("finish,text,tools,reason", [
    ("stop", "sensitive response", [], "complete"),
    ("length", "sensitive partial", [], "nonterminal_finish"),
    ("tool_calls", "sensitive plan", [{"name": "web_search"}], "unexpected_tool_calls"),
    ("stop", "", [], "empty_content"),
])
def test_synthesis_records_safe_reason_not_response_text(finish, text, tools, reason, caplog):
    agent = make_agent()
    response = SimpleNamespace(content=text, tool_calls=tools, finish_reason=finish)
    with (
        patch("agent.web_synthesis._iteration_summary_api_messages", return_value=[]),
        patch("agent.web_synthesis.record_response_usage"),
        patch("agent.relay_llm.execute_current", return_value=response),
        patch("agent.relay_llm.complete_logical_call"),
        caplog.at_level("INFO", logger="agent.web_synthesis"),
    ):
        synthesize_collected_web(agent, [], 4)
    assert agent._last_web_synthesis_diagnostic["reason"] == reason
    assert agent._last_web_synthesis_diagnostic["content_chars"] == len(text)
    assert f"reason={reason}" in caplog.text
    assert "sensitive" not in caplog.text


def test_interrupt_during_synthesis_restores_callbacks_and_propagates():
    agent = make_agent()
    delta, interim = agent._fire_stream_delta, agent.interim_assistant_callback
    with (
        patch("agent.web_synthesis._iteration_summary_api_messages", return_value=[]),
        patch("agent.relay_llm.execute_current", side_effect=InterruptedError),
        patch("agent.relay_llm.complete_logical_call") as finish_call,
        pytest.raises(InterruptedError),
    ):
        synthesize_collected_web(agent, [], 4)
    assert agent._fire_stream_delta is delta
    assert agent.interim_assistant_callback is interim
    assert agent._current_api_request_id == "original"
    finish_call.assert_called_once()
