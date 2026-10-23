"""GXTD-749: signed authored files reach the real Chat Completions writer."""
import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, ThreadSafeAsyncQueue


def test_real_writer_emits_artifact_event_before_done(tmp_path, monkeypatch):
    asyncio.run(_exercise_writer(tmp_path, monkeypatch))


async def _exercise_writer(tmp_path, monkeypatch):
    from PIL import Image
    png = tmp_path / 'diagram.png'
    Image.new('RGB', (8, 8), 'green').save(png)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    response = MagicMock()
    response.write = AsyncMock()
    monkeypatch.setattr(adapter, '_prepare_sse_response', AsyncMock(return_value=response))
    queue = ThreadSafeAsyncQueue()
    final = f'Diagram attached.\n\nMEDIA:{png}'
    async def agent():
        queue.put_nowait(final[:12])
        queue.put_nowait(final[12:])
        queue.put_nowait(None)
        return {'final_response': final, 'completed': True}, None
    task = asyncio.create_task(agent())
    origin = {'subject_id': 'owned-user', 'chat_id': 'chat', 'message_id': 'assistant', 'user_message_id': 'input'}
    await adapter._write_sse_chat_completion(
        MagicMock(), 'completion', 'model', 1, queue, task, [None], artifact_origin=origin, artifact_root=tmp_path)
    frames = [call.args[0] for call in response.write.call_args_list]
    receipt = next(raw for raw in frames if raw.startswith(b'event: hermes.artifacts\n'))
    event = json.loads(receipt.decode().split('data: ', 1)[1])
    assert event['origin'] == origin
    assert base64.b64decode(event['artifacts'][0]['data_base64']) == png.read_bytes()
    assert frames.index(receipt) < frames.index(b'data: [DONE]\n\n')
