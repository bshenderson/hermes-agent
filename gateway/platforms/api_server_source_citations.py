"""Opt-in authenticated default-profile citation consumer; no model/tool authority."""
import asyncio
import sqlite3

from aiohttp import web

from agent.source_citations import checkpoint_identity, validate_citations


def http_routes(adapter):
    async def receive(request):
        return await source_citations(adapter, request)
    return [('GET', '/api/sessions/{session_id}/source-citations', receive),
            ('POST', '/api/sessions/{session_id}/source-citations', receive)]


async def source_citations(adapter, request):
    from gateway.platforms.api_server import _api_request_profile
    from hermes_cli.profiles import get_active_profile_name
    profile = _api_request_profile.get() or get_active_profile_name()
    if profile != 'default':
        return web.json_response({'error': 'source citations require the default native API profile'}, status=403)
    if not adapter._expected_api_key():
        return adapter._auth_failed_response()
    auth = adapter._check_auth(request)
    if auth is not None:
        return auth
    if (adapter.config.extra or {}).get('source_citations_enabled') is not True:
        return web.json_response({'error': 'source citations are disabled'}, status=403)
    body = None
    if request.method == 'POST':
        body, error = await adapter._read_json_body(request)
        if error is not None:
            return error
        if not isinstance(body, dict) or set(body) != {'checkpoint', 'citations'}:
            return web.json_response({'error': 'only checkpoint and citations are accepted'}, status=400)
    try:
        db = await adapter._ensure_session_db_async()
        if db is None:
            return adapter._session_db_unavailable()
        session_id = request.match_info['session_id']
        if body is None:
            checkpoint = await asyncio.to_thread(checkpoint_identity, db, session_id=session_id, profile=profile)
            result = {'version': 1, 'session_id': session_id, 'profile': profile, 'checkpoint': checkpoint}
        else:
            result = await asyncio.to_thread(validate_citations, db, session_id=session_id,
                profile=profile, checkpoint=body['checkpoint'], citations=body['citations'])
    except ValueError:
        return web.json_response({'error': 'source citation validation refused'}, status=409)
    except (OSError, sqlite3.Error):
        return web.json_response({'error': 'source citation store unavailable'}, status=503)
    return web.json_response(result, headers={'Cache-Control': 'no-store'})
