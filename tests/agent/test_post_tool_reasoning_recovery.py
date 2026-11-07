"""GXTD-813 / upstream #34655: post-tool reasoning is not a genuinely empty reply."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.turn_empty_response import recover_empty_response


def _agent():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent._current_streamed_assistant_text = ""
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._post_tool_empty_retried = False
    agent._thinking_prefill_retries = 0
    agent._empty_content_retries = 0
    agent._session_messages = []
    agent._fallback_chain = []
    agent._buffer_diagnostic_status = lambda text: None
    agent._emit_diagnostic_status = lambda text: None
    return agent


def _history():
    return [
        {"role": "user", "content": "Read the fixture and answer."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_fixture", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path":"fixture.txt"}',
            }},
        ]},
        {"role": "tool", "tool_call_id": "call_fixture", "content": "fixture:42"},
    ]


def _recover(agent, message, messages):
    return recover_empty_response(
        agent, message, None, "stop", final_response="", messages=messages,
        api_messages=deepcopy(messages), conversation_history=[], active_system_prompt="stable",
        api_call_count=2, turn_exit_reason=None, preflight_compression_blocked=False,
    )


@pytest.mark.parametrize("field,value", [
    ("reasoning", "I can now use the fixture."),
    ("reasoning_content", "I can now use the fixture."),
    ("reasoning_details", [{"type": "reasoning.text", "text": "Use the fixture."}]),
])
def test_post_tool_structured_reasoning_prefills_without_synthetic_user(field, value):
    agent = _agent()
    messages = _history()
    original = deepcopy(messages)
    message = SimpleNamespace(content="", tool_calls=None, **{field: value})

    verdict = _recover(agent, message, messages)

    assert verdict.action == "continue"
    assert agent._thinking_prefill_retries == 1
    assert agent._post_tool_empty_retried is False
    assert verdict.api_call_count == 2
    assert messages[:len(original)] == original
    assert len(messages) == len(original) + 1
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["_thinking_prefill"] is True
    assert not any(row.get("_empty_recovery_synthetic") for row in messages)


def test_genuine_post_tool_empty_still_nudges_once():
    agent = _agent()
    messages = _history()
    verdict = _recover(agent, SimpleNamespace(content="", tool_calls=None), messages)
    assert verdict.action == "continue"
    assert agent._post_tool_empty_retried is True
    assert agent._thinking_prefill_retries == 0
    assert [row["role"] for row in messages[-2:]] == ["assistant", "user"]
    assert all(row["_empty_recovery_synthetic"] for row in messages[-2:])


def test_exhausted_reasoning_prefill_does_not_fall_back_to_post_tool_nudge(monkeypatch):
    from agent import turn_empty_response

    agent = _agent()
    agent._thinking_prefill_retries = 2
    agent._empty_content_retries = 3
    messages = _history()
    original = deepcopy(messages)
    # Preserve the real retry/terminal path; remove only its optional UI flush.
    agent._flush_status_buffer = lambda: None
    agent.model = "fixture/reasoner"
    agent.provider = "custom"
    monkeypatch.setattr(turn_empty_response._empty_guard, "empty_retry_budget", lambda *a: 0)
    verdict = _recover(agent, SimpleNamespace(content="", tool_calls=None,
                       reasoning="private reasoning"), messages)
    assert verdict.action == "break"
    assert verdict.turn_exit_reason == "empty_response_exhausted"
    assert verdict.final_response == "(empty)"
    assert agent._thinking_prefill_retries == 2
    assert agent._post_tool_empty_retried is False
    assert messages[:len(original)] == original
    assert messages[-1]["_empty_terminal_sentinel"] is True
    assert messages[-1]["content"] == "(empty)"
    assert not any(row.get("_empty_recovery_synthetic") for row in messages)


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
@pytest.mark.parametrize("model,echo", [
    ("gptoss-120b-openai-mxfp4-tp1-128k", True),
    ("precision-default", False),
])
def test_post_tool_recovery_wire_projection_keeps_history_private(field, model, echo):
    from agent.turn_context import build_api_messages

    agent = _agent()
    agent.provider = "custom"
    agent.model = model
    agent.base_url = "http://127.0.0.1:8090/v1"
    agent.api_mode = "chat_completions"
    agent.ephemeral_system_prompt = None
    agent._current_turn_timestamp = 10000.0
    agent._should_sanitize_tool_calls = lambda: False
    messages = _history()
    original = deepcopy(messages)
    secret = "private fixture reasoning"
    verdict = _recover(agent, SimpleNamespace(content="", tool_calls=None,
                       **{field: secret}), messages)
    recovered = deepcopy(messages)
    wire, _ = build_api_messages(
        agent, messages, current_turn_user_idx=0, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    assert agent._needs_thinking_reasoning_pad() is echo
    if echo:
        assert wire[-1]["reasoning_content"] == secret
    else:
        assert "reasoning_content" not in wire[-1]
    # Request assembly runs this real pass on the API copy, not durable history.
    projected = agent._drop_thinking_only_and_merge_users(wire)
    assert [row["role"] for row in projected] == ["user", "assistant", "tool"]
    assert projected[-1]["tool_call_id"] == "call_fixture"
    assert projected[1]["tool_calls"] == original[1]["tool_calls"]
    assert messages == recovered
    assert messages[:len(original)] == original
    assert verdict.final_response == ""
    assert not any(row.get("_empty_recovery_synthetic") for row in messages)
    assert all(secret not in str(row.get("content", "")) for row in projected)
