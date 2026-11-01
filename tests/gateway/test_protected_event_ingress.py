"""GXTD-810: actual signed HTTP ingress -> durable SessionDB -> lossy compressor.

Only the summary model output is substituted. All keys/state are disposable.
"""
import asyncio
from contextlib import asynccontextmanager
import base64
import copy
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB
from tests.agent.test_gxtd641_checkpoint_preservation import compressor, history, make_agent


def signing_bytes(body):
    return b'HERMES_OPERATOR_PROTECTED_EVENT_V1\n' + json.dumps(
        body, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()


def submission(text='Shipment: Cedar archive.\nPark this until Barry approves the receipt.', **updates):
    return dict(dict(version=1, session_id='source-session', submission_id='submit-one',
                     kind='hold', text=text, corrects=[]), **updates)


@asynccontextmanager
async def ingress(tmp_path):
    key = Ed25519PrivateKey.generate()
    public = base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    db = SessionDB(tmp_path / 'events.db')
    db.create_session('source-session', source='api_server')
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        'key': 'disposable-api-bearer', 'protected_event_public_keys': [public]}))
    adapter._session_db = db
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    async with TestClient(TestServer(app)) as client:
        yield client, key, db
    db.close()


async def post(ingress, body, *, key=None, session='source-session', signed=True):
    client, actual_key, _ = ingress
    headers = {'Authorization': 'Bearer disposable-api-bearer'}
    if signed:
        headers['X-Hermes-Operator-Signature'] = base64.b64encode(
            (key or actual_key).sign(signing_bytes(body))).decode()
    return await client.post(f'/api/sessions/{session}/protected-events', json=body, headers=headers)


def test_signed_ingress_real_compaction_close_reopen(compressor, tmp_path):
    async def run():
        async with ingress(tmp_path) as channel:
            await signed_ingress_real_compaction_close_reopen(channel, compressor, tmp_path)
    asyncio.run(run())


async def signed_ingress_real_compaction_close_reopen(ingress, compressor, tmp_path):
    body = submission()
    response = await post(ingress, body)
    assert response.status == 201, await response.text()
    event = (await response.json())['event']
    assert event['text'] == body['text']
    assert event['session_id'] == 'source-session'
    assert event['source_uid'] and event['issuer'] and event['order'] == 1
    db = ingress[2]
    db.append_messages_batch('source-session', history('plain archival data')[1:])
    original = db.get_messages_as_conversation('source-session', repair_alternation=True)
    before = copy.deepcopy(original)
    agent = make_agent(compressor, db)
    agent.compression_in_place = True
    result, _ = agent._compress_context(original, 'Existing system prompt.', force=True)
    assert result != before, compressor._last_summary_error
    from agent.protected_events import collect_events, decode_events
    assert collect_events(compressor, result) == [event]
    assert any(decode_events(row.get('content', '')) == [event] for row in result)
    # Separate connection exercises durable metadata reload, not returned live dictionaries.
    reopened = SessionDB(tmp_path / 'events.db')
    try:
        restored = reopened.get_messages_as_conversation('source-session', repair_alternation=True)
        assert collect_events(compressor, restored) == [event]
        raw = reopened.get_messages('source-session', include_compacted=True)
        assert any(row['message_uid'] == event['source_uid'] and row['content'] == body['text'] for row in raw)
    finally:
        reopened.close()


def test_event_bearing_rotation_refuses_original(compressor, tmp_path):
    async def run():
        async with ingress(tmp_path) as channel:
            assert (await post(channel, submission())).status == 201
            db = channel[2]
            db.append_messages_batch('source-session', history('plain archive')[1:])
            # Match the caller's canonical durable snapshot, not the repaired display projection.
            original = db.get_messages_as_conversation('source-session')
            agent = make_agent(compressor, db)
            agent.compression_in_place = False
            result, _ = agent._compress_context(copy.deepcopy(original), 'Existing system prompt.', force=True)
            assert result == original
            assert agent.session_id == 'source-session'
            assert 'session' in compressor._last_summary_error
    asyncio.run(run())


def test_detached_checkpoint_cannot_silently_lose_events(compressor, tmp_path):
    async def run():
        from agent.protected_events import FIELD
        async with ingress(tmp_path) as channel:
            assert (await post(channel, submission())).status == 201
            channel[2].append_messages_batch('source-session', history('plain archive')[1:])
            original = channel[2].get_messages_as_conversation('source-session', repair_alternation=True)
            first = compressor.compress(copy.deepcopy(original), force=True)
            assert first != original, compressor._last_summary_error
            for row in first:
                row.pop(FIELD, None)
            detached = first + history('plain archive')[1:]
            assert compressor.compress(copy.deepcopy(detached), force=True) == detached
            assert compressor._last_compress_aborted
    asyncio.run(run())
