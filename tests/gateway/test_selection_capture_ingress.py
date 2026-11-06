"""Synthetic selector through signed ingress and the real executor; no model I/O."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tests.gateway.test_archive_chat_continuity import harness, signed, PROFILE, CHAT
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_archive_chat import owned_session_id


@pytest.fixture
def observer_ingress(harness, monkeypatch):
    from agent import compression_source_quotes, selection_capture
    from agent.compression_budget import CompressionBudgetError
    from hermes_constants import get_hermes_home

    _, adapter, _, post = harness
    with adapter._profile_scope(PROFILE):
        root = get_hermes_home() / 'runtime'
    root.mkdir(parents=True, exist_ok=True)
    sid = owned_session_id({'subject_id': 'owner-a', 'chat_id': CHAT}, PROFILE)
    manifest = {'campaign_id': 'synthetic-ingress', 'profile': PROFILE,
                'subject': 'owner-a', 'chat': CHAT, 'root_session_id': sid,
                'armed_at': time.time(), 'expires_at': time.time() + 60,
                'synthetic_only': True}
    path = root / 'selection-capture.json'
    path.write_text(json.dumps(manifest))
    path.chmod(0o600)
    seen = []
    controls = {}

    def create(**kwargs):
        agent = SimpleNamespace(session_id=kwargs['session_id'])

        def run(**turn):
            assert threading.current_thread() is not threading.main_thread()
            if controls.get('expect_scope'):
                assert selection_capture._current.get() is not None
            if 'entered' in controls:
                controls['entered'].set()
                assert controls['release'].wait(5)
            if 'error' in controls:
                raise controls['error']
            try:
                compression_source_quotes._selection('duplicate duplicate', '{"excerpts":["duplicate"]}')
            except CompressionBudgetError as error:
                seen.append(str(error))
            return {'final_response': 'SYNTHETIC', 'completed': True}
        agent.run_conversation = run
        return agent

    monkeypatch.setattr(adapter, '_create_agent', create)
    monkeypatch.setattr(adapter._memory_sessions, 'checkin', lambda agent: None)
    monkeypatch.setattr(adapter, '_run_agent', APIServerAdapter._run_agent.__get__(adapter))

    async def scoped_post(body):
        with adapter._profile_scope(PROFILE):
            return await post(body)

    return SimpleNamespace(root=root, manifest=manifest, path=path, post=scoped_post,
                           seen=seen, controls=controls, adapter=adapter, module=selection_capture)


def captures(fixture):
    return list((fixture.root / 'selection-captures' / 'synthetic-ingress').glob('selection-*.json'))


@pytest.mark.asyncio
async def test_selected_ingress_observes_before_refusal(observer_ingress):
    f = observer_ingress
    response = await f.post(signed([{'role': 'user', 'content': 'Synthetic warmup'}]))
    assert response.status == 200
    assert f.seen == ['ambiguous source excerpt occurrence']
    files = captures(f)
    assert len(files) == 1, 'selected signed ingress did not observe the real executor selector'
    record = json.loads(files[0].read_text())
    assert record['response'] == '{"excerpts":["duplicate"]}'
    assert record['occurrences'] == [2]
    assert files[0].stat().st_mode & 0o777 == 0o600
    assert files[0].parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
@pytest.mark.parametrize('control', ['subject', 'chat', 'profile', 'root_session_id',
                                     'disabled', 'expired', 'oversized', 'unsafe-mode',
                                     'symlink', 'unsigned', 'wrong-signature', 'client-path'])
async def test_nonselected_ingress_never_captures(observer_ingress, control):
    f = observer_ingress
    body = signed([{'role': 'user', 'content': 'Synthetic control'}])
    if control in ('subject', 'chat', 'profile', 'root_session_id'):
        f.manifest[control] = 'foreign'
    elif control == 'expired':
        f.manifest['armed_at'] = time.time() - 100
        f.manifest['expires_at'] = time.time() - 1
    elif control == 'unsigned':
        del body['hermes_chat_context']
    elif control == 'wrong-signature':
        body['hermes_chat_context']['signature'] = 'wrong'
    elif control == 'client-path':
        f.manifest['subject'] = 'foreign'
        body['selection_capture'] = {'root': str(f.root), **f.manifest, 'subject': 'owner-a'}
    f.path.write_text(json.dumps(f.manifest))
    if control == 'disabled':
        f.path.unlink()
    elif control == 'oversized':
        f.path.write_text(' ' * 5000)
    elif control == 'unsafe-mode':
        f.path.chmod(0o644)
    elif control == 'symlink':
        target = f.path.with_suffix('.target')
        f.path.rename(target)
        f.path.symlink_to(target)
    response = await f.post(body)
    assert response.status in (200, 400, 403)
    assert captures(f) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['normal', 'refusal', 'cancelled'])
async def test_worker_context_cleans_up_on_every_exit(observer_ingress, outcome):
    f = observer_ingress
    loop = asyncio.get_running_loop()
    # Reuse exactly the same worker for the turn and the post-turn context probe.
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    error = RuntimeError('synthetic refusal')
    f.controls['expect_scope'] = True
    if outcome == 'refusal':
        f.controls['error'] = error
    if outcome == 'cancelled':
        f.controls.update(entered=threading.Event(), release=threading.Event())
    from agent.owned_chat_history import OwnedChatHistory
    from gateway.platforms.api_server import _api_request_profile
    owned = OwnedChatHistory(f.manifest['root_session_id'], PROFILE, 'owner-a', CHAT)
    token = _api_request_profile.set(PROFILE)
    try:
        with f.adapter._profile_scope(PROFILE):
            task = asyncio.create_task(f.adapter._run_agent(
                'synthetic', [], session_id=owned.root_session_id, owned_chat_history=owned))
    finally:
        _api_request_profile.reset(token)
    try:
        if outcome == 'cancelled':
            deadline = loop.time() + 5
            while not f.controls['entered'].is_set() and loop.time() < deadline:
                await asyncio.sleep(0.01)
            assert f.controls['entered'].is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            f.controls['release'].set()
        elif outcome == 'refusal':
            with pytest.raises(RuntimeError) as caught:
                await task
            assert caught.value is error
        else:
            await task
        assert await loop.run_in_executor(None, f.module._current.get) is None
    finally:
        if 'release' in f.controls:
            f.controls['release'].set()
        pool.shutdown(wait=True)


def test_validator_return_and_exception_identity(tmp_path):
    from agent import compression_source_quotes as quotes, selection_capture as capture
    from agent.compression_budget import CompressionBudgetError
    observer = capture.SelectionCapture(tmp_path / 'captures', time.time() + 60)
    with capture.capture_scope(observer):
        assert quotes._selection('unique', '{"excerpts":["unique"]}') == ['unique']
        with pytest.raises(CompressionBudgetError, match='invalid source selection JSON'):
            quotes._selection('unique', 'invalid JSON')
    assert observer.events == 2
    assert capture._current.get() is None


def test_retention_failures_expiry_and_budget(tmp_path, monkeypatch):
    from agent import compression_source_quotes as quotes, selection_capture as capture
    observer = capture.SelectionCapture(tmp_path / 'captures', time.time() + 60)
    real_fsync = capture.os.fsync
    monkeypatch.setattr(capture.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError('synthetic I/O failure')))
    with capture.capture_scope(observer):
        assert quotes._selection('unique', '{"excerpts":["unique"]}') == ['unique']
    assert observer.bytes > 0 and observer.errors > 0
    monkeypatch.setattr(capture.os, 'fsync', real_fsync)
    before = observer.bytes
    observer.deadline = 0
    with capture.capture_scope(observer):
        quotes._selection('unique', '{"excerpts":["unique"]}')
    assert observer.omitted == 1 and observer.bytes == before
    observer.deadline = time.monotonic() + 60
    observer.bytes = 1048576
    observer.record('unique', '{"excerpts":["unique"]}')
    assert observer.omitted == 2
    observer.events = 128
    observer.record('unique', '{"excerpts":["unique"]}')
    assert observer.omitted == 3


@pytest.mark.asyncio
async def test_registered_http_handler_reaches_selected_executor(observer_ingress):
    import os
    import tempfile
    from aiohttp import web, ClientSession, UnixConnector
    from gateway.platforms.api_server import _api_request_profile
    from tests.gateway.test_archive_chat_continuity import KEY
    f = observer_ingress

    @web.middleware
    async def profile(request, handler):
        token = _api_request_profile.set(PROFILE)
        try:
            with f.adapter._profile_scope(PROFILE):
                return await handler(request)
        finally:
            _api_request_profile.reset(token)

    app = web.Application(middlewares=[profile])
    handler = next(handler for method, path, handler in f.adapter._http_route_table()
                   if method == 'POST' and path == '/v1/chat/completions')
    app.router.add_post('/v1/chat/completions', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        with tempfile.TemporaryDirectory(prefix='sc-', dir=os.environ['TMPDIR']) as directory:
            path = directory + '/s'
            await web.UnixSite(runner, path).start()
            async with ClientSession(connector=UnixConnector(path=path)) as client:
                body = signed([{'role': 'user', 'content': 'Synthetic HTTP warmup'}])
                async with client.post('http://localhost/v1/chat/completions', json=body,
                                       headers={'Authorization': 'Bearer wrong'}) as response:
                    assert response.status == 401
                assert captures(f) == []
                async with client.post('http://localhost/v1/chat/completions', json=body,
                                       headers={'Authorization': 'Bearer ' + KEY}) as response:
                    assert response.status == 200
                    assert (await response.json())['choices'][0]['message']['content'] == 'SYNTHETIC'
                assert len(captures(f)) == 1
    finally:
        await runner.cleanup()
