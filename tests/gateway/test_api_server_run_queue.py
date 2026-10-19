"""Bounded API run admission above active limits (GXTD-655)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from gateway.platforms import api_server_runs as _api_runs
from gateway.platforms import api_server as facade


class Adapter:
    _pending_agent_requests = 0
    _max_concurrent_runs = 1
    _max_pending_runs = 1
    _pending_run_timeout = 2.0

    def __init__(self):
        self._api_run_waiters = []

    def _check_auth(self, request):
        return None

    def _draining_response(self):
        return None

    def active_agent_work_count(self):
        return self._pending_agent_requests


@pytest.mark.asyncio
async def test_queued_turn_waits_then_runs_and_overflow_refuses():
    adapter = Adapter()
    release = asyncio.Event()
    entered = []

    @facade._admit_api_agent_request
    async def handler(self, request):
        limited = facade.APIServerAdapter._concurrency_limited_response(self)
        assert limited is None
        entered.append(request.name)
        if request.name == 'first':
            await release.wait()
        return 'ok'

    with patch.object(facade._api_runs, '_uses_room_run_auth', return_value=False):
        first = asyncio.create_task(handler(adapter, SimpleNamespace(name='first', transport=None)))
        await asyncio.sleep(.01)
        second = asyncio.create_task(handler(adapter, SimpleNamespace(name='second', transport=None)))
        await asyncio.sleep(.1)
        try:
            assert entered == ['first']
            overflow = await handler(adapter, SimpleNamespace(name='overflow', transport=None))
            assert overflow.status == 429
            assert len(adapter._api_run_waiters) == 1
        finally:
            release.set()
        assert await first == 'ok' and await second == 'ok'
    assert entered == ['first', 'second']
    assert adapter._pending_agent_requests == 0 and not adapter._api_run_waiters


@pytest.mark.asyncio
async def test_cancelled_waiter_releases_only_its_reservation():
    from gateway.platforms.api_server_run_queue import wait_for_capacity
    adapter = Adapter()
    adapter._pending_agent_requests = 2
    task = asyncio.create_task(wait_for_capacity(adapter, SimpleNamespace(transport=None)))
    await asyncio.sleep(.05)
    assert len(adapter._api_run_waiters) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not adapter._api_run_waiters
    # The outer admitted-request owner, not the queue, owns this count.
    assert adapter._pending_agent_requests == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('condition,status', [('timeout', 429), ('draining', 503), ('disconnect', 499)])
async def test_waiter_expiry_drain_and_disconnect_do_not_dispatch(condition, status):
    from gateway.platforms.api_server_run_queue import wait_for_capacity
    from aiohttp import web
    adapter = Adapter()
    adapter._pending_agent_requests = 2
    adapter._pending_run_timeout = .02
    if condition == 'draining':
        adapter._draining_response = lambda: web.Response(status=503)
    transport = SimpleNamespace(is_closing=lambda: True) if condition == 'disconnect' else None
    response = await wait_for_capacity(adapter, SimpleNamespace(transport=transport))
    assert response.status == status
    assert not adapter._api_run_waiters


@pytest.mark.asyncio
async def test_real_http_adapter_queues_then_completes_without_inference():
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig
    adapter = facade.APIServerAdapter(PlatformConfig(enabled=True))
    adapter._max_concurrent_runs = 1
    adapter._max_pending_runs = 1
    adapter._pending_run_timeout = 3
    entered = asyncio.Event()
    release = asyncio.Event()

    @facade._admit_api_agent_request
    async def action(self, request):
        assert self._concurrency_limited_response() is None
        if request.match_info['name'] == 'first':
            entered.set()
            await release.wait()
        return web.json_response({'complete': True})

    app = web.Application()
    app.router.add_post('/test/{name}', lambda request: action(adapter, request))
    async with TestClient(TestServer(app)) as client:
        first = asyncio.ensure_future(client.post('/test/first'))
        await asyncio.wait_for(entered.wait(), 3)
        second = asyncio.ensure_future(client.post('/test/second'))
        try:
            for _ in range(100):
                if adapter._api_run_waiters:
                    break
                await asyncio.sleep(.01)
            assert len(adapter._api_run_waiters) == 1
            overflow = await client.post('/test/overflow')
            assert overflow.status == 429
        finally:
            release.set()
        assert (await first).status == 200
        assert (await second).status == 200
    assert adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_disconnected_real_http_waiter_never_enters_handler():
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig
    adapter = facade.APIServerAdapter(PlatformConfig(enabled=True))
    adapter._max_concurrent_runs = 1
    adapter._max_pending_runs = 1
    adapter._pending_run_timeout = 3
    release = asyncio.Event()
    active = asyncio.Event()
    entered = []

    @facade._admit_api_agent_request
    async def action(self, request):
        entered.append(request.match_info['name'])
        if request.match_info['name'] == 'first':
            active.set()
            await release.wait()
        return web.json_response({'complete': True})

    app = web.Application()
    app.router.add_post('/test/{name}', lambda request: action(adapter, request))
    async with TestClient(TestServer(app)) as client:
        first = asyncio.ensure_future(client.post('/test/first'))
        await asyncio.wait_for(active.wait(), 3)
        url = client.make_url('/test/disconnected')
        _, writer = await asyncio.open_connection(url.host, url.port)
        writer.write(b'POST /test/disconnected HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n')
        await writer.drain()
        try:
            for _ in range(100):
                if adapter._api_run_waiters:
                    break
                await asyncio.sleep(.01)
            assert len(adapter._api_run_waiters) == 1
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(.1)
        finally:
            release.set()
            writer.close()
        assert (await first).status == 200
        await asyncio.sleep(.15)
    assert entered == ['first']
    assert adapter.active_agent_work_count() == 0 and not adapter._api_run_waiters


@pytest.mark.asyncio
async def test_cancelled_decorated_waiter_retires_reservation():
    adapter = Adapter()
    adapter._pending_agent_requests = 1  # unrelated active turn

    @facade._admit_api_agent_request
    async def action(self, request):
        raise AssertionError('cancelled waiter dispatched')

    with patch.object(facade._api_runs, '_uses_room_run_auth', return_value=False):
        task = asyncio.create_task(action(adapter, SimpleNamespace(transport=None)))
        await asyncio.sleep(.1)
        assert adapter._pending_agent_requests == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert adapter._pending_agent_requests == 1 and not adapter._api_run_waiters


@pytest.mark.asyncio
async def test_aiohttp_request_with_disappeared_transport_is_refused():
    from aiohttp.test_utils import make_mocked_request
    from gateway.platforms.api_server_run_queue import wait_for_capacity
    request = make_mocked_request('POST', '/test')
    request._protocol.transport = None
    assert request.transport is None
    adapter = Adapter()
    adapter._pending_agent_requests = 2
    task = asyncio.create_task(wait_for_capacity(adapter, request))
    await asyncio.sleep(.1)
    adapter._pending_agent_requests = 1  # unrelated active work finishes
    response = await task
    assert response is not None and response.status == 499
    assert not adapter._api_run_waiters
