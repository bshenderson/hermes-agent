"""One answer-only completion after the web-search synthesis redirect."""
from __future__ import annotations

import logging
import time
import uuid

from agent.chat_completion_helpers import _iteration_summary_api_messages
from agent.turn_usage import record_response_usage

logger = logging.getLogger(__name__)

UNAVAILABLE = (
    "The search budget was reached. I could not complete an answer from the collected "
    "sources. Retrieved material remains in the conversation history. "
    "No additional research tools were run during finalization."
)


FINALIZE_INSTRUCTION = (
    "Research has ended for this turn. Do not call tools or propose further research. "
    "Answer the user's original request now using only the sources already collected. "
    "Give the supported findings even if some requested details remain unavailable; "
    "cite source URLs, distinguish snippets from retrieved full text, and state the "
    "specific unresolved facts. Do not invent missing facts or imply unavailable pages "
    "were read. Return a final answer, not an action plan."
)


def _diagnostic(agent, reason, *, finish_reason=None, content_chars=0, tool_calls=0):
    # Never log response text, reasoning, tool arguments, or exception messages.
    known_finishes = {"stop", "end_turn", "length", "max_tokens", "tool_calls", "tool_use", "content_filter"}
    safe_finish = finish_reason if finish_reason in known_finishes else "other_or_missing"
    agent._last_web_synthesis_diagnostic = {
        "reason": reason, "finish_reason": safe_finish,
        "content_chars": content_chars, "tool_calls": tool_calls,
    }
    log = logger.info if reason == "complete" else logger.warning
    log("Bounded web synthesis reason=%s finish_reason=%s content_chars=%d tool_calls=%d",
        reason, safe_finish, content_chars, tool_calls)


def synthesize_collected_web(agent, messages, api_call_count):
    """Return (text, completed, attempted). Never dispatch tools or retry a model round.

    Keep the prompt prefix and schemas intact; append an explicit finalization
    instruction and disable tools for this one request. Buffer streamed output
    until a terminal, text-only response is accepted. The ordinary transport
    still owns interruption, deadlines and wire-level retries.
    """
    if agent._interrupt_requested:
        gate = "interrupted"
    elif api_call_count >= agent.max_iterations:
        gate = "iteration_limit"
    elif agent.api_mode not in {"chat_completions", "codex_responses", "anthropic_messages"}:
        gate = "unsupported_transport"
    elif not agent.iteration_budget.consume():
        gate = "shared_budget"
    else:
        gate = None
    if gate:
        _diagnostic(agent, gate)
        return UNAVAILABLE, False, False

    from agent import relay_llm

    request_id = f"web-synthesis:{uuid.uuid4()}"
    old_request_id = agent._current_api_request_id
    old_delta = agent._fire_stream_delta
    old_interim = agent.interim_assistant_callback
    outcome = "failed"
    started = time.monotonic()
    try:
        agent._current_api_request_id = request_id
        agent._fire_stream_delta = lambda *args, **kwargs: None
        agent.interim_assistant_callback = None
        # The redirect's tool result is untrusted data, not a reliable instruction.
        # Append a turn-local instruction without rewriting the cached prefix or
        # persisting a synthetic user turn into the actual conversation.
        api_messages = _iteration_summary_api_messages(agent, messages) + [
            {"role": "user", "content": FINALIZE_INSTRUCTION}
        ]
        kwargs = agent._build_api_kwargs(api_messages)
        kwargs["tool_choice"] = {"type": "none"} if agent.api_mode == "anthropic_messages" else "none"
        agent._touch_activity("synthesizing collected web sources")
        response = relay_llm.execute_current(
            kwargs, agent._interruptible_api_call,
            name=str(agent.provider or "provider"), model_name=str(agent.model or ""),
            metadata={"api_mode": agent.api_mode, "api_request_id": request_id,
                      "call_role": "web_synthesis"},
            defer_logical_completion=True,
        )
        record_response_usage(
            agent, response, messages=messages, api_call_count=api_call_count + 1,
            api_duration=time.monotonic() - started, compression_attempts=0,
            max_compression_attempts=getattr(agent, "max_compression_attempts", 3),
        )
        normalized = agent._get_transport().normalize_response(response)
        text = (normalized.content or "").strip()
        if agent._interrupt_requested:
            reason = "interrupted"
        elif normalized.tool_calls:
            reason = "unexpected_tool_calls"
        elif not text:
            reason = "empty_content"
        elif normalized.finish_reason not in {"stop", "end_turn"}:
            reason = "nonterminal_finish"
        else:
            reason = "complete"
        _diagnostic(agent, reason, finish_reason=normalized.finish_reason,
                    content_chars=len(text), tool_calls=len(normalized.tool_calls or []))
        if reason != "complete":
            return UNAVAILABLE, False, True
        outcome = "success"
        return text, True, True
    except InterruptedError:
        _diagnostic(agent, "interrupted")
        raise
    except Exception:
        _diagnostic(agent, "exception")
        return UNAVAILABLE, False, True
    finally:
        agent._fire_stream_delta = old_delta
        agent.interim_assistant_callback = old_interim
        agent._current_api_request_id = old_request_id
        relay_llm.complete_logical_call(request_id, outcome=outcome)
