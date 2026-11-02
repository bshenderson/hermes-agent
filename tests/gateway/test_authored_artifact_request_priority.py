"""GXTD-755: attachment support must not become an output-file request.

Exercises the real HTTP normalization/instruction boundary, not model compliance.
Existing signed-binding and byte-transport tests own authentication/publication.
"""
from unittest.mock import patch
import hashlib
import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('multimodal', [False, True])
async def test_artifact_capability_preserves_inline_request_priority(tmp_path, monkeypatch, multimodal):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    app.router.add_post('/v1/chat/completions', adapter._handle_chat_completions)
    prompt = 'Give only the saved counts inline. Do not use any tools or create a file.'
    content = [{'type': 'text', 'text': prompt}] if multimodal else prompt
    origin = {'subject_id': 'fixture-owner', 'chat_id': 'fixture-chat',
              'message_id': 'fixture-message', 'user_message_id': 'fixture-input'}
    captured = {}

    async def run(**kwargs):
        captured.update(kwargs)
        return {'final_response': '118 total, 51 files, 67 directories', 'completed': True}, {}

    async with TestClient(TestServer(app)) as client:
        with patch('gateway.platforms.api_server_artifacts.artifact_binding', return_value=origin), \
             patch.object(adapter, '_run_agent', side_effect=run):
            response = await client.post('/v1/chat/completions', json={
                'model': 'hermes-agent', 'messages': [
                    {'role': 'user', 'content': 'prior question'},
                    {'role': 'assistant', 'content': 'prior answer'},
                    {'role': 'user', 'content': content}]})
            assert response.status == 200, await response.text()
    user = captured['user_message']
    if isinstance(user, list):
        assert user[0] == {'type': 'text', 'text': prompt}
        instruction = user[-1]['text']
    else:
        assert user.startswith(prompt + '\n')
        instruction = user[len(prompt):]
    assert 'capability, not a request to create a file' in instruction
    assert 'If the user did not request an output file, answer inline' in instruction
    assert 'Honor explicit no-file and no-tool instructions' in instruction
    assert 'Only for output files actually requested by the user' in instruction
    namespace = hashlib.sha256(json.dumps(origin, sort_keys=True).encode()).hexdigest()
    assert f'{tmp_path}/cache/scratch/authored-artifacts/{namespace}/' in instruction
    assert 'MEDIA:<absolute-path>' in instruction

    assert captured['conversation_history'] == [
        {'role': 'user', 'content': 'prior question'},
        {'role': 'assistant', 'content': 'prior answer'}]
