
import asyncio
import time
import pytest
from aiohttp import web, ClientSession, TCPConnector
from gateway.platforms.api_server import APIServerAdapter
from gateway.config import PlatformConfig
from gateway.platforms.api_server_openai_routes import _iter_stream_items

@pytest.mark.asyncio
async def test_silent_sse_disconnect_detection():
    """
    Verify that a silent SSE stream (no items in queue) detects 
    client disconnect within 2 seconds, rather than waiting for the keepalive.
    """
    # Setup adapter and mock request/response
    adapter = APIServerAdapter(PlatformConfig(enabled=True, token='test-token'))
    interrupted = asyncio.Event()
    
    # We need a real aiohttp server to test actual socket closure observation
    async def handler(request):
        from gateway.platforms.api_server import ThreadSafeAsyncQueue
        queue = ThreadSafeAsyncQueue()
        
        # Synthetic agent task that just sleeps (silent)
        async def silent_agent():
            try:
                await asyncio.sleep(10)
                return {"final_response": "done"}, {}
            except asyncio.CancelledError:
                interrupted.set()
                raise

        task = asyncio.create_task(silent_agent())
        
        # Use the production writer logic
        return await adapter._write_sse_responses(
            request, 'diag', 'diag', 0, queue, task, [None], [], 'test',
            None, None, False, 'disposable'
        )

    app = web.Application()
    app.router.add_post('/v1/chat/completions', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        async with ClientSession(connector=TCPConnector(force_close=True)) as session:
            # Start the request
            response = await session.post(f'http://127.0.0.1:{port}/v1/chat/completions', json={})
            assert response.status == 200
            
            # Read first chunk to ensure connection is established
            await response.content.readany()
            
            start_time = time.monotonic()
            # SILENT DISCONNECT: Close the client session immediately
            response.close()
            
        await asyncio.wait_for(interrupted.wait(), timeout=2)
        
    finally:
        await runner.cleanup()

# Refined test with signaling
@pytest.mark.asyncio
async def test_silent_disconnect_interrupts_fast():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, token='test-token'))
    interrupted_event = asyncio.Event()

    async def handler(request):
        from gateway.platforms.api_server import ThreadSafeAsyncQueue
        queue = ThreadSafeAsyncQueue()
        
        # Mock agent that signals when it is interrupted (cancelled)
        async def mock_agent():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                interrupted_event.set()
                raise

        task = asyncio.create_task(mock_agent())
        
        # The production writer calls _abandon_agent_task on disconnect
        return await adapter._write_sse_chat_completion(
            request, 'diag', 'diag', 0, queue, task, agent_ref=[None]
        )

    app = web.Application()
    app.router.add_post('/v1/chat/completions', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        async with ClientSession(connector=TCPConnector(force_close=True)) as session:
            response = await session.post(f'http://127.0.0.1:{port}/v1/chat/completions', json={})
            await response.content.readany()
            response.close()

        # Assert that the agent task was interrupted within 2 seconds
        try:
            await asyncio.wait_for(interrupted_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("Agent was not interrupted within 2s after silent disconnect")

    finally:
        await runner.cleanup()
