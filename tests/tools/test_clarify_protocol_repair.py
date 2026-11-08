"""Delivery failures must not masquerade as deliberate unanswered forms."""
import json

import pytest

from tools.clarify_tool import clarify_tool


@pytest.mark.parametrize("reply", [
    "", "not-json", "null", "[]", 12345, {}, {"answers": None},
    {"answers": []}, {"answers": {"q9": "wrong question"}},
    {"answers": {"q0": 42}, "outcome": "submitted"},
    {"answers": {}, "outcome": "not-an-outcome"},
    {"answers": {"q9": "wrong question"}, "outcome": "submitted"},
])
def test_malformed_batch_reply_reports_protocol_failure(reply):
    def callback(questions):
        return reply

    result = json.loads(clarify_tool(questions=[{"question": "One?"}], callback=callback))
    assert "error" in result
    assert "protocol" in result["error"].lower()
    assert "responses" not in result


@pytest.mark.parametrize("answers,status", [({}, "unanswered"), ({"q0": None}, "skipped")])
def test_explicit_batch_skip_is_not_delivery_failure(answers, status):
    def callback(questions):
        return {"answers": answers, "outcome": "submitted"}

    result = json.loads(clarify_tool(questions=[{"question": "One?"}], callback=callback))
    assert result["responses"][0]["user_response"] is None
    assert result["responses"][0]["status"] == status
    assert result["outcome"] == "submitted"
    assert "error" not in result
    assert "timed_out" not in result


@pytest.mark.parametrize("answers", [{}, {"q0": "kept"}])
def test_batch_timeout_remains_distinct(answers):
    def callback(questions):
        return {"answers": answers, "outcome": "timed_out"}

    result = json.loads(clarify_tool(questions=[{"question": "One?"}, {"question": "Two?"}], callback=callback))
    assert result["outcome"] == "timed_out"
    assert result["responses"][1]["user_response"] is None
    assert result["responses"][1]["status"] == "unanswered"
    if answers:
        assert result["responses"][0]["user_response"] == "kept"
