"""Delivery failures must not masquerade as deliberate unanswered forms."""
import json

import pytest

from tools.clarify_tool import TIMEOUT_RESPONSE, clarify_tool


@pytest.mark.parametrize("reply", [
    "", "not-json", "null", "[]", 12345, {}, {"answers": None},
    {"answers": []}, {"answers": {"q9": "wrong question"}},
    {"answers": {"q0": 42}}, {"answers": {}, "timed_out": "false"},
])
def test_malformed_batch_reply_reports_protocol_failure(reply):
    def callback(question, choices, questions=None):
        return reply

    result = json.loads(clarify_tool("", questions=[{"question": "One?"}], callback=callback))
    assert "error" in result
    assert "protocol" in result["error"].lower()
    assert "responses" not in result


@pytest.mark.parametrize("reply", [{"answers": {}}, '{"answers": {}}'])
def test_explicit_batch_skip_is_not_delivery_failure(reply):
    def callback(question, choices, questions=None):
        return reply

    result = json.loads(clarify_tool("", questions=[{"question": "One?"}], callback=callback))
    assert result["responses"][0]["user_response"] == ""
    assert "error" not in result
    assert "timed_out" not in result


@pytest.mark.parametrize("reply", [None, TIMEOUT_RESPONSE, {"answers": {"q0": "kept"}, "timed_out": True}])
def test_batch_timeout_remains_distinct(reply):
    def callback(question, choices, questions=None):
        return reply

    result = json.loads(clarify_tool("", questions=[{"question": "One?"}, {"question": "Two?"}], callback=callback))
    assert result["timed_out"] is True
    assert result["responses"][1]["user_response"] == ""
    if isinstance(reply, dict):
        assert result["responses"][0]["user_response"] == "kept"
