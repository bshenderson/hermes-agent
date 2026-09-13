"""Protected OpenAI attempts must close only their own HTTP request on stop."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import OpenAI
from agent import auxiliary_client as aux
from agent.process_bootstrap import build_keepalive_http_client


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('headers_first', [False, True])
def test_cancel_closes_owned_socket_without_closing_shared_client(headers_first, stream):
    accepted = threading.Event()
    eof = threading.Event()
    release = threading.Event()
    provider_done = threading.Event()
    owner_done = threading.Event()
    cancel = threading.Event()
    outcomes = []
    sibling_accepted = threading.Event()
    sibling_release = threading.Event()
    sibling_eof = threading.Event()
    sibling_results = []
    owned_requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if body['messages'][0]['content'] == 'sibling':
                sibling_accepted.set()
                self.connection.settimeout(.1)
                while not sibling_release.is_set():
                    try:
                        if self.connection.recv(1, socket.MSG_PEEK) == b'':
                            sibling_eof.set()
                            return
                    except socket.timeout:
                        pass
                data = json.dumps({'id':'sibling', 'object':'chat.completion', 'created':0, 'model':'fixture', 'choices':[{'index':0,'message':{'role':'assistant','content':'ok'},'finish_reason':'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            owned_requests.append(body['messages'][0]['content'])
            if headers_first:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.flush()
            accepted.set()
            self.connection.settimeout(.1)
            while not release.is_set():
                try:
                    if self.connection.recv(1, socket.MSG_PEEK) == b'':
                        eof.set()
                        break
                except socket.timeout:
                    pass
                except OSError:
                    eof.set()
                    break

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f'http://127.0.0.1:{server.server_port}/v1'
    client = OpenAI(base_url=base_url, api_key='fixture', timeout=3, max_retries=2,
                    http_client=build_keepalive_http_client(base_url))

    def sibling():
        sibling_results.append(client.chat.completions.create(model='fixture', messages=[{'role':'user','content':'sibling'}]))

    sibling_thread = threading.Thread(target=sibling, daemon=True)

    def provider(kwargs):
        try:
            return aux._create_with_progress_once(client, kwargs, task='compression', force_stream=stream)
        finally:
            provider_done.set()

    def owner():
        try:
            with aux.aux_interrupt_protection(cancel_event=cancel):
                aux._run_protected_sync_provider_call(provider, {'model':'fixture','messages':[{'role':'user','content':'owned'}]})
        except aux.AuxiliaryExplicitCancellation:
            outcomes.append('cancelled')
        finally:
            owner_done.set()

    caller = threading.Thread(target=owner, daemon=True)
    try:
        sibling_thread.start()
        assert sibling_accepted.wait(3)
        caller.start()
        assert accepted.wait(3)
        cancel.set()
        assert owner_done.wait(1)
        assert eof.wait(1), 'cancelled provider left owned socket open'
        assert provider_done.wait(2), 'provider daemon did not unwind'
        assert owned_requests == ['owned'], 'cancelled SDK retried generation'
        assert not sibling_eof.is_set(), 'unrelated shared-pool request was closed'
        sibling_release.set()
        sibling_thread.join(3)
        assert sibling_results[0].choices[0].message.content == 'ok'
        assert outcomes == ['cancelled']
        assert not client.is_closed()
        response = client.chat.completions.create(model='fixture', messages=[{'role':'user','content':'sibling'}])
        assert response.choices[0].message.content == 'ok'
    finally:
        release.set()
        sibling_release.set()
        sibling_thread.join(4)
        caller.join(4)
        provider_done.wait(4)
        client.close()
        server.shutdown()
        server.server_close()
