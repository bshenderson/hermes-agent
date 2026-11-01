"""Explicit operator-only protected input, never a model/tool metadata channel."""
import asyncio
import base64
import binascii
import hashlib

from aiohttp import web
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from agent.protected_events import signing_bytes, validate_request
from hermes_state_protected_events import append_verified_event
from hermes_state_errors import SessionTurnLeaseLostError, SessionCompressionInProgressError, CompressionSessionClosedError


def http_routes(adapter):
    async def receive(request):
        return await receive_event(adapter, request)
    return [('POST', '/api/sessions/{session_id}/protected-events', receive)]


def verified_issuer(keys, body, signature):
    try:
        signature = base64.b64decode(signature, validate=True)
        for encoded in keys:
            raw = base64.b64decode(encoded, validate=True)
            public = Ed25519PublicKey.from_public_bytes(raw)
            try:
                public.verify(signature, signing_bytes(body))
            except InvalidSignature:
                continue
            return 'ed25519:' + hashlib.sha256(raw).hexdigest()
    except (ValueError, TypeError, binascii.Error):
        pass
    raise ValueError('operator signature is not authorized')


async def receive_event(adapter, request):
    from gateway.platforms.api_server import _api_request_profile
    # Never inherit the launch profile's issuer keys on /p/<other>/ routes.
    if _api_request_profile.get() not in (None, 'default'):
        return web.json_response({'error': 'protected ingress requires the default native API profile'}, status=403)
    if not adapter._expected_api_key():
        return adapter._auth_failed_response()
    auth = adapter._check_auth(request)
    if auth is not None:
        return auth
    keys = (adapter.config.extra or {}).get('protected_event_public_keys', [])
    if not isinstance(keys, list) or not keys:
        return web.json_response({'error': 'protected operator ingress is disabled'}, status=403)
    body, error = await adapter._read_json_body(request)
    if error is not None:
        return error
    try:
        validate_request(body, request.match_info['session_id'])
        issuer = verified_issuer(keys, body, request.headers.get('X-Hermes-Operator-Signature', ''))
    except (ValueError, TypeError, UnicodeError):
        return web.json_response({'error': 'invalid or unauthorized protected event'}, status=400)
    db = await adapter._ensure_session_db_async()
    try:
        event, inserted = await asyncio.to_thread(append_verified_event, db, body, issuer)
    except (ValueError, SessionTurnLeaseLostError, SessionCompressionInProgressError, CompressionSessionClosedError) as error:
        return web.json_response({'error': str(error)}, status=409)
    return web.json_response({'event': event, 'inserted': inserted}, status=201 if inserted else 200)
