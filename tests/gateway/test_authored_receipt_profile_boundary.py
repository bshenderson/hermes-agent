"""GXTD-755 candidate admission gate: receipt recovery must retain profile isolation."""
import pytest


@pytest.mark.asyncio
async def test_other_profile_cannot_resolve_retained_receipt(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from agent import secret_scope as scope
    from agent.chat_location import native_context, sign_context
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
    keys = {'alice': 'a' * 40, 'bob': 'b' * 40}
    home = tmp_path / 'home'
    for name, key in keys.items():
        profile = home / 'profiles' / name
        profile.mkdir(parents=True)
        (profile / '.env').write_text('API_SERVER_KEY=' + key + '\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'd' * 40}))
    adapter.gateway_runner = type('Runner', (), {'config': GatewayConfig(multiplex_profiles=True)})()
    scope.set_multiplex_active(True)
    # Substitute only bootstrap activation; retain real scoped secrets/home and
    # HTTP middleware, key comparison, signed-origin validation and storage.
    from contextlib import contextmanager
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    @contextmanager
    def runtime_scope(profile):
        profile_home = home / 'profiles' / profile
        secret_token = scope.set_secret_scope({'API_SERVER_KEY': keys[profile]}, profile_home=str(profile_home))
        home_token = set_hermes_home_override(profile_home)
        try:
            yield
        finally:
            reset_hermes_home_override(home_token)
            scope.reset_secret_scope(secret_token)
    monkeypatch.setattr(adapter, '_profile_scope', runtime_scope)
    ids = {'chat_id': '11111111-1111-4111-8111-111111111111',
           'message_id': '22222222-2222-4222-8222-222222222222',
           'user_message_id': '33333333-3333-4333-8333-333333333333'}
    origin = {'subject_id': 'fixture-owner', **ids}
    root = home / 'profiles' / 'alice'
    output = root / 'result.txt'
    output.write_text('Alice-owned completed output')
    token = _api_request_profile.set('alice')
    try:
        with runtime_scope('alice'):
            assert adapter._seal_authored_result({'completed': True, 'final_response': f'MEDIA:{output}'}, origin, root, '')
    finally:
        _api_request_profile.reset(token)
    body = {'hermes_artifact_transport': 'inline-v1'}
    context = native_context(user_id='fixture-owner', metadata=ids)
    body['hermes_chat_context'] = sign_context(context, body=body, user_id='fixture-owner',
                                              profile='bob', key=keys['bob'])
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, '/p/{profile}' + path, handler)
    app['api_server_adapter'] = adapter
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post('/p/bob/v1/authored-artifacts/receipt',
                headers={'Authorization': 'Bearer ' + keys['bob']}, json=body)
            value = await response.json()
            assert response.status in {400, 403, 404} or value == {'status': 'unavailable'}, (
                'Candidate receipt retention bypasses source-profile isolation')
    finally:
        await adapter.disconnect()
        scope.set_multiplex_active(False)
