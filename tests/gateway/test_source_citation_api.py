"""GXTD-810 real authenticated HTTP citation consumer, no native inference."""
import asyncio
from contextlib import asynccontextmanager
import base64
import copy
import json
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
from tests.agent.test_gxtd641_checkpoint_preservation import compressor, history, make_agent
from tests.gateway.test_protected_event_ingress import signing_bytes, submission

AUTH = {'Authorization': 'Bearer disposable-api-bearer'}
PATH = '/api/sessions/source-session/source-citations'


@asynccontextmanager
async def client_store(tmp_path, *, enabled=True, key='disposable-api-bearer'):
    signer = Ed25519PrivateKey.generate()
    public = base64.b64encode(signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    db = SessionDB(tmp_path / 'api-citations.db')
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        'key': key, 'source_citations_enabled': enabled, 'protected_event_public_keys': [public]}))
    adapter._session_db = db
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    async with TestClient(TestServer(app)) as client:
        yield client, adapter, db, signer
    db.close()


@pytest.mark.parametrize('quote', [
    'Source [B]: Felix must deliver Cedar October 19.',
    'The Juniper parcel stays parked; only Barry can green-light its departure.',
    'Scope: Juniper [Ω].\nKeep it here.\nReceipt marker: <!-- HERMES_SOURCE_COMMITMENTS_V1 -->',
], ids=['legacy-and-event', 'explicit-event-only', 'multiline-reserved-marker'])
def test_actual_signed_session_compaction_and_citation_http(compressor, tmp_path, quote):
    async def run():
        async with client_store(tmp_path) as (client, adapter, db, signer):
            created = await client.post('/api/sessions', json={'id': 'source-session'}, headers=AUTH)
            assert created.status == 201
            assert db.get_session('source-session')['profile_name'] == 'default'
            body = submission(text=quote, kind='commitment')
            headers = dict(AUTH, **{'X-Hermes-Operator-Signature': base64.b64encode(signer.sign(signing_bytes(body))).decode()})
            accepted = await client.post('/api/sessions/source-session/protected-events', json=body, headers=headers)
            assert accepted.status == 201
            agent = make_agent(compressor, db)
            agent.compression_in_place = True
            for round_index in range(2):
                db.append_messages_batch('source-session', history('plain archival data')[1:])
                original = await adapter._conversation_history_for_session('source-session')
                frozen = copy.deepcopy(original)
                stored_before = db.get_messages('source-session', include_inactive=True)
                active_before = db.get_active_message_ids('source-session')
                result, _ = agent._compress_context(original, 'Existing system prompt.', force=True)
                if 'HERMES_SOURCE_COMMITMENTS_V1' in quote and round_index == 1:
                    # Existing reserved-marker guard is containment, not successful compaction.
                    assert result == frozen
                    assert compressor._last_compress_aborted
                    assert db.get_messages('source-session', include_inactive=True) == stored_before
                    assert db.get_active_message_ids('source-session') == active_before
                else:
                    assert result != original, compressor._last_summary_error
                # Separate real SQLite connection, no trusted metadata manually reconstructed.
                with SessionDB(tmp_path / 'api-citations.db') as reopened:
                    adapter._session_db = reopened
                    snapshot = await client.get(PATH, headers=AUTH)
                    assert snapshot.status == 200, await snapshot.text()
                    checkpoint = (await snapshot.json())['checkpoint']
                    response = await client.post(PATH, json={'checkpoint': checkpoint, 'citations': [quote]}, headers=AUTH)
                    assert response.status == 200, await response.text()
                    answer = await response.json()
                    assert answer['profile'] == 'default'
                    assert answer['citations'][0]['quote'] == quote
                    assert answer['citations'][0]['sources']
                adapter._session_db = db
    asyncio.run(run())


@pytest.mark.parametrize('method', ['GET', 'POST'])
@pytest.mark.parametrize('condition,status', [('missing-bearer', 401), ('wrong-bearer', 401),
                                             ('no-server-key', 401), ('disabled', 403),
                                             ('truthy-not-enabled', 403)])
def test_endpoint_requires_explicit_enablement_and_bearer(tmp_path, method, condition, status):
    async def run():
        enabled = False if condition == 'disabled' else 'true' if condition == 'truthy-not-enabled' else True
        key = '' if condition == 'no-server-key' else 'disposable-api-bearer'
        async with client_store(tmp_path, enabled=enabled, key=key) as (client, _, db, _):
            headers = {} if condition == 'missing-bearer' else {'Authorization': 'Bearer wrong'} if condition == 'wrong-bearer' else AUTH
            before = db._read_one('SELECT count(*) AS n FROM sessions')['n']
            response = await client.request(method, PATH, headers=headers, json={'checkpoint': '0' * 64, 'citations': ['source']})
            assert response.status == status, await response.text()
            assert db._read_one('SELECT count(*) AS n FROM sessions')['n'] == before
    asyncio.run(run())


@pytest.mark.parametrize('body,status', [
    ({'checkpoint': '0' * 64, 'citations': ['source'], 'profile': 'research'}, 400),
    ({'citations': ['source']}, 400),
    ([], 400),
    ({'checkpoint': '0' * 64, 'citations': ['source']}, 409),
])
def test_citation_request_cannot_supply_authority(tmp_path, body, status):
    async def run():
        async with client_store(tmp_path) as (client, _, _, _):
            response = await client.post(PATH, json=body, headers=AUTH)
            assert response.status == status, await response.text()
    asyncio.run(run())


@pytest.mark.parametrize('profile', ['research', None])
def test_citation_http_refuses_wrong_or_unowned_durable_profile(tmp_path, profile):
    async def run():
        async with client_store(tmp_path) as (client, _, db, _):
            db.create_session('source-session', 'api_server', profile_name=profile)
            db.append_message('source-session', 'user', 'private source')
            response = await client.get(PATH, headers=AUTH)
            assert response.status == 409
            assert 'private source' not in await response.text()
            assert db.get_session('source-session')['profile_name'] == profile
    asyncio.run(run())


def test_citation_http_store_failure_is_503(tmp_path, monkeypatch):
    async def run():
        async with client_store(tmp_path) as (client, adapter, _, _):
            async def unavailable():
                raise OSError('private diagnostic must not leak')
            monkeypatch.setattr(adapter, '_ensure_session_db_async', unavailable)
            response = await client.get(PATH, headers=AUTH)
            assert response.status == 503
            assert 'private diagnostic' not in await response.text()
    asyncio.run(run())


def test_new_session_ownership_is_server_derived(tmp_path):
    async def run():
        async with client_store(tmp_path) as (client, _, db, _):
            response = await client.post('/api/sessions', json={
                'id': 'source-session', 'profile_name': 'research', 'profile': 'research'}, headers=AUTH)
            assert response.status == 201
            assert db.get_session('source-session')['profile_name'] == 'default'
    asyncio.run(run())


def test_named_profile_mirror_cannot_inherit_citation_access(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    root = tmp_path / '.hermes'
    named = root / 'profiles' / 'researcher'
    named.mkdir(parents=True)
    (named / 'config.yaml').write_text('{}')
    (root / 'config.yaml').write_text('{}')
    monkeypatch.setenv('HERMES_HOME', str(root))
    async def run():
        async with client_store(tmp_path) as (_, adapter, db, _):
            from gateway.session_authorities import SessionAuthorities
            authorities = SessionAuthorities(root, multiplexed=True)
            authorities.add(root, SimpleNamespace(db=db, profile_id=str(root)))
            assert authorities.for_home(root).db is db
            assert authorities.for_home(named) is None
            adapter.gateway_runner = SimpleNamespace(
                config=SimpleNamespace(multiplex_profiles=True), session_authorities=authorities)
            app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
            for method, path, handler in adapter._http_route_table():
                app.router.add_route(method, path, handler)
                app.router.add_route(method, '/p/{profile}' + path, handler)
            db.create_session('source-session', 'api_server', profile_name='default')
            db.append_message('source-session', 'user', 'source')
            async with TestClient(TestServer(app)) as client:
                for method in ('GET', 'POST'):
                    response = await client.request(method, '/p/researcher' + PATH, headers=AUTH,
                        json={'checkpoint': '0' * 64, 'citations': ['source']})
                    assert response.status == 403
                response = await client.get('/p/default' + PATH, headers=AUTH)
                assert response.status == 200
    asyncio.run(run())


@pytest.mark.parametrize('fault', ['stale-journal', 'missing-source', 'changed-source-uid', 'invalid-journal'])
def test_journal_citation_refuses_stale_or_detached_custody(tmp_path, fault):
    async def run():
        async with client_store(tmp_path) as (client, _, db, signer):
            assert (await client.post('/api/sessions', json={'id': 'source-session'}, headers=AUTH)).status == 201
            quote = 'The parcel stays parked; only Barry can green-light its departure.'
            body = submission(text=quote)
            headers = dict(AUTH, **{'X-Hermes-Operator-Signature': base64.b64encode(signer.sign(signing_bytes(body))).decode()})
            accepted = await client.post('/api/sessions/source-session/protected-events', json=body, headers=headers)
            event = (await accepted.json())['event']
            snapshot = await client.get(PATH, headers=AUTH)
            checkpoint = (await snapshot.json())['checkpoint']
            initial = await client.post(PATH, json={'checkpoint': checkpoint, 'citations': [quote]}, headers=AUTH)
            assert initial.status == 200
            if fault == 'stale-journal':
                body = submission(text='Another parcel stays here.', submission_id='two')
                headers['X-Hermes-Operator-Signature'] = base64.b64encode(signer.sign(signing_bytes(body))).decode()
                assert (await client.post('/api/sessions/source-session/protected-events', json=body, headers=headers)).status == 201
            elif fault == 'missing-source':
                db._write_sql('DELETE FROM messages WHERE message_uid = ?', (event['source_uid'],))
            elif fault == 'changed-source-uid':
                db._write_sql('UPDATE messages SET message_uid = ? WHERE message_uid = ?', ('different-source', event['source_uid']))
            else:
                raw = db._read_one('SELECT event FROM operator_protected_events')['event']
                invalid = json.loads(raw)
                invalid['version'] = 999
                db._write_sql('UPDATE operator_protected_events SET event = ?', (json.dumps(invalid),))
            if fault != 'stale-journal':
                rebound = await client.get(PATH, headers=AUTH)
                if fault == 'invalid-journal':
                    assert rebound.status == 409
                    return
                assert rebound.status in (200, 409)
                if rebound.status == 200:
                    checkpoint = (await rebound.json())['checkpoint']
            response = await client.post(PATH, json={'checkpoint': checkpoint, 'citations': [quote]}, headers=AUTH)
            assert response.status == 409
    asyncio.run(run())
