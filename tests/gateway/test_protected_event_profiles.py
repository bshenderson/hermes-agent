"""GXTD-810: genuine signed ingress through production profile middleware/mirrors.

No live profiles, listener service or operator keys; topology is disposable.
"""
import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB
from tests.gateway.test_protected_event_ingress import signing_bytes, submission


@pytest.mark.parametrize('topology', ['standalone-default', 'standalone-named', 'multiplex'])
def test_signed_ingress_profile_middleware(tmp_path, monkeypatch, topology):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    root = tmp_path / '.hermes'
    named = root / 'profiles' / 'researcher'
    named.mkdir(parents=True)
    (named / 'config.yaml').write_text('{}')
    (root / 'config.yaml').write_text('{}')
    home = named if topology == 'standalone-named' else root
    monkeypatch.setenv('HERMES_HOME', str(home))
    key = Ed25519PrivateKey.generate()
    public = base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    db = SessionDB(home / 'state.db')
    db.create_session('source-session', source='api_server')
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        'key': 'disposable-api-bearer', 'protected_event_public_keys': [public]}))
    adapter._session_db = db
    if topology == 'multiplex':
        from gateway.session_authorities import SessionAuthorities
        authorities = SessionAuthorities(home, multiplexed=True)
        authorities.add(home, SimpleNamespace(db=db, profile_id=str(home)))
        assert authorities.for_home(home).db is db
        assert authorities.for_home(named) is None
        adapter.gateway_runner = SimpleNamespace(
            config=SimpleNamespace(multiplex_profiles=True), session_authorities=authorities)
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, '/p/{profile}' + path, handler)

    async def run():
        async with TestClient(TestServer(app)) as client:
            async def send(prefix, identity):
                body = submission(submission_id=identity)
                return await client.post(prefix + '/api/sessions/source-session/protected-events', json=body,
                    headers={'Authorization': 'Bearer disposable-api-bearer',
                             'X-Hermes-Operator-Signature': base64.b64encode(key.sign(signing_bytes(body))).decode()})
            own = '/p/researcher' if topology == 'standalone-named' else '/p/default'
            response = await send(own, 'own-profile')
            assert response.status == 201, await response.text()
            original = db.get_messages('source-session', include_inactive=True)
            foreign = '/p/default' if topology == 'standalone-named' else '/p/researcher'
            response = await send(foreign, 'foreign-profile')
            assert response.status == (403 if topology == 'multiplex' else 404), await response.text()
            assert db.get_messages('source-session', include_inactive=True) == original
            assert (await send('/p/ghost', 'unknown-profile')).status == 404
            response = await send('', 'bare-native-profile')
            assert response.status == 201, await response.text()
    try:
        asyncio.run(run())
    finally:
        db.close()
