"""GXTD-814 offline signed ingress -> production compression dispatch -> SDK transport."""
import json
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from tests.gateway.test_archive_chat_continuity import harness, signed, PROFILE, CHAT
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_archive_chat import owned_session_id


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['reasoning', 'content', 'think', 'details', 'plain', 'http', 'length', 'extraction_error',
    'invented', 'v1', 'foreign_subject', 'foreign_profile', 'foreign_chat', 'foreign_root',
    'unsigned', 'bad_signature', 'stale_signature', 'expired', 'unknown_version',
    'wrong_kind', 'unsafe_mode', 'missing_manifest'])
async def test_signed_response_capture_vertical(harness, monkeypatch, mode):
    from agent import auxiliary_client as aux, auxiliary_request as owned
    from agent import context_compressor as cc, compression_source_quotes as q
    from agent import conversation_compression as conv, selection_capture as cap
    from hermes_constants import get_hermes_home
    _, adapter, _, post = harness
    with adapter._profile_scope(PROFILE):
        root = get_hermes_home() / 'runtime'
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(campaign_id='response-vertical', profile=PROFILE, subject='owner-a', chat=CHAT,
        root_session_id=owned_session_id({'subject_id': 'owner-a', 'chat_id': CHAT}, PROFILE),
        armed_at=time.time(), expires_at=time.time() + 60, synthetic_only=True,
        version=2, capture_kind='response-boundary-v1')
    path = root / 'selection-capture.json'
    path.write_text(json.dumps(manifest))
    path.chmod(0o600)
    if mode == 'v1':
        del manifest['version'], manifest['capture_kind']
    elif mode.startswith('foreign_'):
        key = {'foreign_subject': 'subject', 'foreign_profile': 'profile',
               'foreign_chat': 'chat', 'foreign_root': 'root_session_id'}[mode]
        manifest[key] = 'foreign'
    elif mode == 'expired':
        manifest.update(armed_at=time.time()-120, expires_at=time.time()-60)
    elif mode == 'unknown_version':
        manifest['version'] = 3
    elif mode == 'wrong_kind':
        manifest['capture_kind'] = 'foreign'
    path.write_text(json.dumps(manifest))
    if mode == 'unsafe_mode':
        path.chmod(0o644)
    elif mode == 'missing_manifest':
        path.unlink()
    selected = mode in ('reasoning', 'content', 'think', 'details', 'plain', 'invented', 'http', 'length', 'extraction_error')
    source = 'Mira must review KESTREL; do not send.'
    answer = json.dumps({'excerpts': ['invented' if mode == 'invented' else source]})
    payload = dict(content=answer)
    if mode == 'reasoning':
        payload = dict(content='', reasoning_content=answer)
    elif mode == 'think':
        payload = dict(content='<think>synthetic analysis</think>' + answer)
    elif mode == 'details':
        payload = dict(content='', reasoning_details=[dict(text=answer)])
    requests = []
    def transport(request):
        assert threading.current_thread().name == 'hermes-protected-aux-provider'
        assert (cap._current.get() is not None) == (selected or mode == 'v1')
        if selected:
            assert cap._current.get().campaign.identity == {k: manifest[k] for k in ('profile', 'subject', 'chat', 'root_session_id')}
        requests.append(json.loads(request.content))
        if mode == 'plain':
            return httpx.Response(200, json=dict(id='synthetic', created=0, object='chat.completion',
                model='fixture', choices=[dict(index=0, message={'role': 'assistant', **payload}, finish_reason='stop')]))
        body = ''.join('data: ' + json.dumps(dict(id='synthetic', created=0,
            object='chat.completion.chunk', model='fixture', choices=[dict(index=0,
            delta=delta, finish_reason=finish)])) + '\n\n' for delta, finish in
            [(payload, None), ({}, 'length' if mode == 'length' else 'stop')]) + 'data: [DONE]\n\n'
        return httpx.Response(200, text=body, headers={'content-type': 'text/event-stream'})
    cached_http = httpx.Client(transport=httpx.MockTransport(transport))
    client = OpenAI(api_key='fixture-key', base_url='http://offline.invalid/v1',
        http_client=cached_http, max_retries=0)
    def builder(*a, **kw):
        return httpx.Client(transport=httpx.MockTransport(transport))
    monkeypatch.setattr(owned, 'build_keepalive_http_client', builder)
    monkeypatch.setattr(aux, '_resolve_task_provider_model', lambda *a: (
        'custom', 'fixture', 'http://offline.invalid/v1', 'fixture-key', 'chat_completions'))
    monkeypatch.setattr(aux, '_resolve_call_client', lambda *a, **kw: (client, 'fixture', 'custom', 'custom'))
    monkeypatch.setattr(aux, '_get_auxiliary_task_config', lambda *a: {'recovery_policy': 'fail_closed'})
    monkeypatch.setattr(aux, '_get_task_extra_body', lambda *a: {'chat_template_kwargs': {'enable_thinking': False}})
    monkeypatch.setattr(cc, 'lossless_settings', lambda: {'output_tokens': 12000})
    monkeypatch.setattr(cc, '_pinned_summary_call_kwargs', dict)
    original_error = ValueError('synthetic extraction failure')
    if mode == 'extraction_error':
        def extraction_failure(*a, **kw):
            raise original_error
        monkeypatch.setattr(cc, 'extract_content_or_reasoning', extraction_failure)
    if mode == 'plain':
        monkeypatch.setattr(aux, '_aux_progress_active', lambda: False)
    def create(**kwargs):
        c = cc.ContextCompressor.__new__(cc.ContextCompressor)
        for k, v in dict(model='fixture', provider='custom', base_url='http://offline.invalid/v1',
            api_key='fixture-key', api_mode='chat_completions', summary_model=None, context_length=131072).items():
            setattr(c, k, v)
        c._compression_cancelled = lambda: False
        c._record_aux_compression_call = lambda **kw: None
        agent = SimpleNamespace(session_id=kwargs['session_id'], context_compressor=c)
        def run(**turn):
            assert (cap._current.get() is not None) == (selected or mode == 'v1')
            def compress(messages):
                return q.extract_summary(source,
                    lambda prompt: c._call_summary_llm(prompt, time.monotonic(), response_format=q.FORMAT),
                    context_length=131072, output_tokens=12000)
            if mode in ('length', 'extraction_error'):
                with pytest.raises(RuntimeError if mode == 'length' else ValueError) as caught:
                    conv._run_summary_dispatch(agent, [], compress, {}, commit_fence=None,
                        attempt_generation=1, hard_cancel_event=threading.Event())
                if mode == 'length':
                    assert 'truncated' in str(caught.value)
                else:
                    assert caught.value is original_error
            elif mode == 'invented':
                from agent.compression_budget import CompressionBudgetError
                with pytest.raises(CompressionBudgetError, match='not present'):
                    conv._run_summary_dispatch(agent, [], compress, {}, commit_fence=None,
                        attempt_generation=1, hard_cancel_event=threading.Event())
            else:
                summary = conv._run_summary_dispatch(agent, [], compress, {}, commit_fence=None,
                    attempt_generation=1, hard_cancel_event=threading.Event())
                assert source in summary
            return {'final_response': 'SYNTHETIC', 'completed': True}
        agent.run_conversation = run
        return agent
    monkeypatch.setattr(adapter, '_create_agent', create)
    monkeypatch.setattr(adapter._memory_sessions, 'checkin', lambda agent: None)
    monkeypatch.setattr(adapter, '_run_agent', APIServerAdapter._run_agent.__get__(adapter))
    original_socket = socket.socket
    class OfflineSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *a, **kw):
            if family in (socket.AF_INET, socket.AF_INET6):
                raise AssertionError('no Internet/model sockets in offline vertical')
            super().__init__(family, *a, **kw)
    monkeypatch.setattr(socket, 'socket', OfflineSocket)
    try:
        body = signed([{'role': 'user', 'content': 'Synthetic observation'}])
        if mode == 'unsigned':
            del body['hermes_chat_context']
        elif mode == 'bad_signature':
            body['hermes_chat_context']['signature'] = 'wrong'
        elif mode == 'stale_signature':
            old = time.time() - 120
            with monkeypatch.context() as local:
                local.setattr(time, 'time', lambda: old)
                body = signed([{'role': 'user', 'content': 'Synthetic observation'}])
        if mode == 'http':
            import os
            import tempfile
            from aiohttp import web, ClientSession, UnixConnector
            from gateway.platforms.api_server import _api_request_profile
            from tests.gateway.test_archive_chat_continuity import KEY
            @web.middleware
            async def profile(request, handler):
                token = _api_request_profile.set(PROFILE)
                try:
                    with adapter._profile_scope(PROFILE):
                        return await handler(request)
                finally:
                    _api_request_profile.reset(token)
            app = web.Application(middlewares=[profile])
            handler = next(h for method, route, h in adapter._http_route_table()
                if method == 'POST' and route == '/v1/chat/completions')
            app.router.add_post('/v1/chat/completions', handler)
            runner = web.AppRunner(app)
            await runner.setup()
            try:
                with tempfile.TemporaryDirectory(prefix='r814-', dir=os.environ['TMPDIR']) as directory:
                    await web.UnixSite(runner, directory+'/s').start()
                    async with ClientSession(connector=UnixConnector(path=directory+'/s')) as session:
                        async with session.post('http://localhost/v1/chat/completions', json=body,
                            headers={'Authorization': 'Bearer wrong'}) as rejected:
                            assert rejected.status == 401
                        assert not requests
                        async with session.post('http://localhost/v1/chat/completions', json=body,
                            headers={'Authorization': 'Bearer '+KEY}) as accepted:
                            assert accepted.status == 200
                            await accepted.json()
                        response = accepted
            finally:
                await runner.cleanup()
        else:
            with adapter._profile_scope(PROFILE):
                response = await post(body)
        if mode in ('unsigned', 'bad_signature', 'stale_signature'):
            assert response.status == 400
            assert not requests
            assert not list((root / 'selection-captures').glob('*/event-*.json'))
            return
        assert response.status == 200
        assert len(requests) == 1
        assert requests[0]['messages'] and requests[0]['response_format'] == q.FORMAT
        assert requests[0]['chat_template_kwargs']['enable_thinking'] is False
        files = list((root / 'selection-captures' / 'response-vertical').glob('event-*.json'))
        records = [json.loads(x.read_text()) for x in files]
        if not selected:
            assert not records
            return
        kinds = {'transport_request', 'summary_response', 'extraction', 'selection'}
        if mode in ('length', 'extraction_error'):
            kinds.discard('selection')
        if mode == 'extraction_error':
            kinds.discard('extraction')
            kinds.add('extraction_error')
        if mode != 'plain':
            kinds.add('sdk_delta')
        assert kinds <= {x['kind'] for x in records}
        if mode == 'extraction_error':
            assert next(x for x in records if x['kind'] == 'extraction_error')['data']['error_class'] == 'ValueError'
            return
        extracted = next(x for x in records if x['kind'] == 'extraction')
        branch = 'reasoning_fallback' if mode in ('reasoning', 'details') else 'inline_think_content' if mode == 'think' else 'content'
        assert extracted['data']['branch'] == branch
        assert extracted['data']['selected'] == answer
        assert len({x['request_id'] for x in records}) == 1
        assert len({x['selection_call_id'] for x in records}) == 1
        assert all(x['identity']['chat'] == CHAT for x in records)
        assert not cached_http.is_closed and cached_http.event_hooks['request'] == []
    finally:
        client.close()
