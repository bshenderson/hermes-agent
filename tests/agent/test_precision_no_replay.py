"""GXTD-743 selected agent errors terminate without recovery/replay."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agent.turn_api_error import handle_api_error


@pytest.mark.parametrize('status', [400, 409, 429, 500, None])
@pytest.mark.parametrize('generation', [1, 3])
def test_precision_never_retries_after_dispatch_or_tools(monkeypatch, status, generation):
    import agent.turn_api_error as errors
    recovery = Mock(side_effect=AssertionError('no compression/rotation/recovery allowed'))
    monkeypatch.setattr(errors, 'recover_before_classification', recovery)
    agent = SimpleNamespace(_hermes_precision_no_replay=True, thinking_callback=None)
    history = [{'role': 'user', 'content': 'private input'}]
    if generation > 1:
        history += [{'role': 'assistant', 'tool_calls': [{'id': 'effect-once'}]},
            {'role': 'tool', 'tool_call_id': 'effect-once', 'content': 'effect completed'}]
    before = list(history)
    verdict = handle_api_error(agent, api_error=SimpleNamespace(status_code=status),
        _retry=None, thinking_spinner=None, messages=history, api_messages=history,
        api_kwargs={}, system_message=None, active_system_prompt='stable',
        conversation_history=history, approx_tokens=100, retry_count=0, max_retries=3,
        compression_attempts=0, max_compression_attempts=2, api_call_count=generation,
        api_request_id='fixture', api_start_time=0, effective_task_id='fixture', turn_id='fixture')
    assert verdict.action == 'return'
    assert verdict.result['failure_reason'] == 'precision_no_replay'
    assert verdict.result['failure_retryable'] is False
    assert verdict.result['completed'] is False
    assert verdict.result['messages'] is history
    assert history == before
    assert verdict.retry_count == 0
    recovery.assert_not_called()
