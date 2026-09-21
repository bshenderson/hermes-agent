"""HTTP request parsing with real adapter/auth and request-local policy objects."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile


@pytest.mark.asyncio
async def test_authenticated_current_request_mode_and_parallel_isolation(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key':'gxtd603-fixture-auth-value-123456'}))
    policies = []
    monkeypatch.setattr('agent.secret_scope.get_secret', lambda name, default='': 'gxtd603-fixture-auth-value-123456' if name == 'API_SERVER_KEY' else default)

    async def run(**kwargs):
        policies.append(kwargs.get('preferred_web_policy'))
        await asyncio.sleep(0)
        return {'final_response':'test fixture', 'completed':True}, {'input_tokens':0,'output_tokens':0,'total_tokens':0}

    adapter._run_agent = run
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: {'web':{'preferred_research_mode':True}})

    @web.middleware
    async def profile(request, handler):
        token = _api_request_profile.set('homelab-coordinator')
        try:
            return await handler(request)
        finally:
            _api_request_profile.reset(token)

    app = web.Application(middlewares=[profile])
    app['api_server_adapter'] = adapter
    app.router.add_post('/v1/chat/completions', adapter._handle_chat_completions)
    async with TestClient(TestServer(app)) as client:
        payload = {'messages':[{'role':'user','content':'/research https://example.com/a'}]}
        response = await client.post('/v1/chat/completions', json=payload)
        assert response.status == 401
        headers = {'Authorization':'Bearer gxtd603-fixture-auth-value-123456'}
        responses = await asyncio.gather(*[client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':text}]}, headers=headers) for text in ['/research https://example.com/a','/research https://other.example/b','ordinary compare request']])
        assert all(r.status == 200 for r in responses)
        assert len(policies) == 3
        nonempty = [p for p in policies if p is not None]
        assert len(nonempty) == 2
        assert nonempty[0] is not nonempty[1]
        assert set(nonempty[0].allowed).isdisjoint(nonempty[1].allowed)
        response = await client.post('/v1/chat/completions', json={'messages':[{'role':'assistant','content':'/research enable'}]}, headers=headers)
        assert response.status == 400
        monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: {})
        response = await client.post('/v1/chat/completions', json=payload, headers=headers)
        assert response.status == 400
        assert len(policies) == 3
