"""HTTP authenticated origin/profile lookup of a producer-owned disposition."""
from contextlib import contextmanager
import pytest


@pytest.mark.asyncio
async def test_failure_lookup_exact_origin_profile_auth_and_expiry(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from agent.chat_location import native_context, sign_context
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
    from gateway.platforms.api_server_terminal import run_with_disposition, FailureDispositionStore
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'fixture-key'}))
    @contextmanager
    def scope(profile):
        home = tmp_path / profile
        home.mkdir(exist_ok=True)
        token = set_hermes_home_override(home)
        try:
            yield
        finally:
            reset_hermes_home_override(token)
    monkeypatch.setattr(adapter, '_profile_scope', scope)
    monkeypatch.setattr(adapter, '_expected_api_key', lambda: 'fixture-key')
    ids = {'chat_id': '11111111-1111-4111-8111-111111111111',
           'message_id': '22222222-2222-4222-8222-222222222222',
           'user_message_id': '33333333-3333-4333-8333-333333333333'}
    origin = {'subject_id': 'owner', **ids}
    flags = {'failed': True, 'completed': False, 'failure_reason': 'dispatch_refused', 'failure_retryable': False}
    run_with_disposition(adapter, lambda: (flags, {}), 'alice', origin)
    @web.middleware
    async def routing(request, handler):
        profile = request.match_info['profile']
        token = _api_request_profile.set(profile)
        try:
            with scope(profile):
                return await handler(request)
        finally:
            _api_request_profile.reset(token)
    app = web.Application(middlewares=[routing])
    app.router.add_post('/p/{profile}/v1/authored-artifacts/receipt', adapter._handle_authored_receipt)
    def body(profile, subject='owner'):
        value = {'hermes_artifact_transport': 'inline-v1'}
        value['hermes_chat_context'] = sign_context(native_context(user_id=subject, metadata=ids),
            body=value, user_id=subject, profile=profile, key='fixture-key')
        return value
    async with TestClient(TestServer(app)) as client:
        for profile, subject, expected in [('alice', 'owner', 'failed'), ('bob', 'owner', 'unavailable'), ('alice', 'foreign', 'unavailable')]:
            response = await client.post(f'/p/{profile}/v1/authored-artifacts/receipt',
                headers={'Authorization': 'Bearer fixture-key'}, json=body(profile, subject))
            assert response.status == 200
            value = await response.json()
            assert value['status'] == expected
            assert 'receipt' not in value
            if expected == 'failed':
                assert value['disposition']['origin'] == origin
        response = await client.post('/p/alice/v1/authored-artifacts/receipt', json=body('alice'))
        assert response.status == 401
    store = FailureDispositionStore(tmp_path / 'alice' / 'authored-failures.sqlite3')
    assert store.get(origin, now=10**12) is None
