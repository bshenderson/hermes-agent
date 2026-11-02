"""GXTD-743 physical request retries honor the selected no-replay owner."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import run_agent  # Bootstrap before the runner installs its private-home guard.
from agent import chat_completion_helpers as helpers
from agent.turn_response_check import retry_invalid_response


def stream_call():
    call = object.__new__(helpers._StreamingCall)
    call.agent = SimpleNamespace(
        _hermes_precision_no_replay=True, _interrupt_requested=False,
        _stream_options_unsupported=False, _stream_5xx_probe_ts=None,
        _is_provider_stream_parse_error=lambda error: False,
        _log_stream_retry=Mock(), _buffer_diagnostic_status=Mock(),
        _disable_streaming=False, api_mode='chat_completions', base_url='http://fixture',
        provider='custom', model='fixture')
    call.api_kwargs = {'model': 'fixture', 'messages': []}
    call.result = {'response': None, 'error': None, 'partial_tool_names': []}
    call.clients = SimpleNamespace(diag={}, close_once=Mock())
    call._request_cancelled = {'value': False}
    call.deltas_were_sent = {'yes': False}
    call.first_delta_fired = {'done': False}
    call.provider_tool_in_flight = {'yes': False}
    call._start_stream_attempt = Mock(return_value=1)
    call._close_managed_stream = Mock()
    call._cancel_current_stream_attempt = Mock()
    call._stream_stale_timeout = 180.0
    call._retry_after_drop = Mock()
    call._route_switched_under_request = lambda: False
    return call


@pytest.mark.parametrize('owner', ['parent', 'substantial-child'])
def test_real_stream_loop_dispatches_once_on_unknown_read(monkeypatch, owner):
    call = stream_call()
    error = httpx.ReadTimeout('unknown native outcome')
    call._call_wire = Mock(side_effect=error)
    monkeypatch.setattr(helpers, '_with_stream_emitters', lambda agent, invoke: invoke())
    monkeypatch.setenv('HERMES_STREAM_RETRIES', '2')
    from agent import stream_diag
    monkeypatch.setattr(stream_diag, 'buffer_connect_exhausted_notice', lambda *a, **kw: None)
    call._call()
    assert call._call_wire.call_count == 1, owner
    assert call.result['error'] is error
    call._retry_after_drop.assert_not_called()


@pytest.mark.parametrize('partial', [False, True])
def test_stream_errors_cannot_reopen_for_compatibility_or_partial_tool(monkeypatch, partial):
    call = stream_call()
    call.deltas_were_sent['yes'] = partial
    call.provider_tool_in_flight['yes'] = partial
    call._visible_text_delivered = lambda: False
    call.agent._reset_stream_delivery_tracking = Mock()
    call.agent._warning_presentation_enabled = lambda: False
    monkeypatch.setattr(helpers, '_rejects_stream_options', lambda error: not partial)
    error = httpx.ReadTimeout('unknown response')
    assert call._handle_stream_error(error, attempt=0, max_retries=2) is False
    assert call.result['error'] is error
    assert call.agent._stream_options_unsupported is False
    call._retry_after_drop.assert_not_called()


def test_http500_unmask_cannot_issue_second_request(monkeypatch):
    call = stream_call()
    second_request = Mock(side_effect=RuntimeError('second dispatch'))
    monkeypatch.setattr(helpers, 'interruptible_api_call', second_request)
    assert call._unmask_server_error_with_nonstreaming(SimpleNamespace(status_code=500)) is False
    second_request.assert_not_called()


def test_malformed_response_is_terminal_with_effect_bearing_history():
    agent = SimpleNamespace(_hermes_precision_no_replay=True, thinking_callback=None,
        _invoke_api_request_error_hook=Mock(side_effect=AssertionError('retry owner reached')))
    history = [{'role': 'assistant', 'tool_calls': [{'id': 'effect-once'}]},
               {'role': 'tool', 'tool_call_id': 'effect-once', 'content': 'completed'}]
    verdict = retry_invalid_response(agent, response=None, error_details=['missing choices'],
        _retry=None, thinking_spinner=None, messages=history, api_messages=history,
        api_kwargs={}, active_system_prompt='stable', conversation_history=history,
        retry_count=0, max_retries=3, compression_attempts=0, api_call_count=3,
        api_request_id='fixture', api_start_time=0, api_duration=0,
        effective_task_id='fixture', turn_id='fixture')
    assert verdict.action == 'return'
    assert verdict.result['failure_reason'] == 'precision_no_replay'
    assert verdict.result['failure_retryable'] is False
    assert verdict.result['messages'] is history
    assert verdict.retry_count == 0


@pytest.mark.parametrize('kind', ['text', 'incomplete-tool', 'parseable-tool'])
def test_unconfirmed_clean_eof_cannot_become_a_success_or_continuation(kind):
    call = stream_call()
    tools = {} if kind == 'text' else {0: {'id': 'new-call', 'type': 'function',
        'function': {'name': 'terminal', 'arguments': '{"command":' if kind == 'incomplete-tool' else '{"command":"pwd"}'}}}
    with pytest.raises(helpers.EmptyStreamError, match='unconfirmed'):
        call._finish_chat_stream(SimpleNamespace(response=None), 'assistant',
            ['partial'] if kind == 'text' else [], ['analysis'], tools, None,
            'fixture', None, flush_pending=lambda: None)


@pytest.mark.parametrize('kind', ['text', 'tool'])
def test_confirmed_stream_results_remain_available_to_normal_tool_loop(kind):
    call = stream_call()
    tools = {} if kind == 'text' else {0: {'id': 'new-call', 'type': 'function',
        'function': {'name': 'terminal', 'arguments': '{"command":"pwd"}'}}}
    reason = 'stop' if kind == 'text' else 'tool_calls'
    response = call._finish_chat_stream(SimpleNamespace(response=None), 'assistant',
        ['complete'] if kind == 'text' else [], ['analysis'], tools, reason,
        'fixture', None, flush_pending=lambda: None)
    assert response.choices[0].finish_reason == reason
    if kind == 'tool':
        assert response.choices[0].message.tool_calls[0].function.arguments == '{"command":"pwd"}'


def test_retained_error_after_visible_deltas_is_not_translated_to_partial_stub(monkeypatch):
    call = stream_call()
    error = httpx.ReadTimeout('native outcome unknown after text')
    call.result['error'] = error
    call.deltas_were_sent['yes'] = True
    call._resolve_stale_timeout = Mock()
    call._run_call = Mock()
    call._monitor_loop = Mock()
    call._partial_stream_stub = Mock(return_value=object())
    monkeypatch.setattr(helpers, 'should_use_direct_api_call', lambda agent: True)
    with pytest.raises(httpx.ReadTimeout):
        call.run()
    call._partial_stream_stub.assert_not_called()


@pytest.mark.parametrize('kind', ['text', 'incomplete-tool'])
def test_partial_stub_is_terminal_before_response_recovery(monkeypatch, kind):
    from agent import turn_response_check as checks
    from agent.turn_retry_state import TurnRetryState
    from agent.transports.chat_completions import ChatCompletionsTransport
    from hermes_cli.observability import shared_metrics_harness
    agent = stream_call().agent
    agent.thinking_callback = None
    agent.quiet_mode = True
    agent.verbose_logging = False
    agent._get_transport = lambda: ChatCompletionsTransport()
    agent._should_treat_stop_as_truncated = lambda *args: False
    agent.log_prefix = ''
    forbidden = Mock(side_effect=AssertionError('uncertain response reached continuation'))
    monkeypatch.setattr(checks, 'recover_from_truncation', forbidden)
    monkeypatch.setattr(shared_metrics_harness, 'record_reply_finish', lambda *args: None)
    stub = helpers._build_partial_stream_stub('assistant', 'partial', None, 'fixture', None,
        clean_eof=True, dropped_tool_names=['terminal'] if kind != 'text' else None)
    history = [{'role': 'tool', 'tool_call_id': 'effect-once', 'content': 'completed'}]
    state = TurnRetryState()
    verdict = checks.check_api_response(agent, response=stub, _retry=state,
        thinking_spinner=None, messages=history, api_messages=history, api_kwargs={},
        active_system_prompt='stable', conversation_history=history, finish_reason=None,
        retry_count=0, max_retries=3, compression_attempts=0, max_compression_attempts=2,
        length_continue_retries=0, truncated_response_parts=[], truncated_tool_call_retries=0,
        current_turn_user_idx=0, api_call_count=3, api_request_id='fixture', api_start_time=0,
        effective_task_id='fixture', turn_id='fixture', _preflight_compression_blocked=False,
        _last_preflight_pressure=None)
    assert verdict.action == 'return'
    assert verdict.result['failure_reason'] == 'precision_no_replay'
    assert verdict.result['failure_retryable'] is False
    assert verdict.result['messages'] is history
    assert state.restart_with_length_continuation is False
    forbidden.assert_not_called()
