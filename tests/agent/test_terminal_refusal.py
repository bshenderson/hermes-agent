"""GXTD-755: a typed refusal must bypass every agent recovery rung."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import openai
import pytest

from agent.turn_api_error import handle_api_error

REFUSAL = {'error': {'type': 'no_replay_refusal', 'code': 'dispatch_refused',
                     'contract': 'inference.no_replay.v1', 'retryable': False,
                     'message': 'Request refused by ownership gate.'}}


def test_sdk_and_agent_terminal_refusal(monkeypatch):
    calls = []

    class Peer(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            data = json.dumps(REFUSAL).encode()
            self.send_response(412)
            self.send_header('Content-Type', 'application/json')
            self.send_header('x-should-retry', 'false')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    def forbidden(*args, **kwargs):
        raise AssertionError('terminal refusal entered recovery')

    monkeypatch.setattr('agent.turn_api_error.recover_before_classification', forbidden)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        with openai.OpenAI(base_url=f'http://127.0.0.1:{server.server_port}/v1', api_key='fixture', max_retries=2) as client:
            with pytest.raises(openai.APIStatusError) as raised:
                client.chat.completions.create(model='fixture', messages=[{'role': 'user', 'content': 'fixture'}])
        verdict = handle_api_error(
            SimpleNamespace(thinking_callback=None), api_error=raised.value,
            _retry=None, thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
            system_message='', active_system_prompt='', conversation_history=[], approx_tokens=0,
            retry_count=0, max_retries=3, compression_attempts=0, max_compression_attempts=2,
            api_call_count=1, api_request_id='fixture', api_start_time=0,
            effective_task_id='fixture', turn_id='fixture')
        assert verdict.action == 'return'
        assert verdict.result['failure_reason'] == 'dispatch_refused'
        assert verdict.result['failed'] is True
        assert verdict.result['failure_retryable'] is False
        assert len(calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize('status,body,attempts,terminal', [
    (412, REFUSAL, 1, True),
    (503, {'error': {'message': 'temporary'}}, 3, False),
    (409, REFUSAL, 3, False),
    (412, {'error': {'message': 'no replay'}}, 1, False),
    (412, {'error': {**REFUSAL['error'], 'retryable': 'false'}}, 1, False),
])
def test_sdk_retry_controls_and_classifier(status, body, attempts, terminal):
    import httpx
    from agent.error_classifier import classify_api_error, FailoverReason
    from agent.turn_recovery_autorecover import ladder_eligible
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(status, json=body)
    with openai.OpenAI(api_key='fixture', max_retries=2,
                       http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        with pytest.raises(openai.APIStatusError) as raised:
            client.chat.completions.create(model='fixture', messages=[])
    assert len(calls) == attempts
    verdict = classify_api_error(raised.value)
    assert (verdict.reason == FailoverReason.dispatch_refused) is terminal
    if terminal:
        assert not any((verdict.retryable, verdict.should_compress,
                        verdict.should_fallback, verdict.should_rotate_credential))
        assert not ladder_eligible(SimpleNamespace(_auto_recovery_cycles=5), verdict)
    if status == 503:
        assert verdict.retryable
