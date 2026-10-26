"""GXTD-755: completed receipt retention, not producer execution replay."""
import importlib
import pytest


def test_completed_receipt_survives_reopen_without_recapturing_files(tmp_path):
    artifacts = importlib.import_module('gateway.platforms.api_server_artifacts')
    store_type = getattr(artifacts, 'CompletedReceiptStore', None)
    assert store_type is not None, 'Completed authored receipts have no durable transport owner'
    origin = {'subject_id': 'owner', 'chat_id': 'chat', 'message_id': 'assistant', 'user_message_id': 'input'}
    receipt = {'schema': 'hermes.artifacts.v1', 'origin': origin, 'artifacts': [], 'errors': []}
    store = store_type(tmp_path / 'receipts.sqlite3')
    store.save(origin, receipt, now=100)
    assert store_type(tmp_path / 'receipts.sqlite3').get(origin, now=101) == receipt


@pytest.mark.parametrize('failure_at', ['receipt', 'role', 'prepare'])
def test_real_writer_retains_completed_receipt_before_lost_sse_write(tmp_path, monkeypatch, failure_at):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    import asyncio
    import base64
    from unittest.mock import AsyncMock, MagicMock
    from PIL import Image
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, ThreadSafeAsyncQueue
    from gateway.platforms.api_server_artifacts import CompletedReceiptStore

    async def exercise():
        png = tmp_path / 'diagram.png'
        Image.new('RGB', (8, 8), 'green').save(png)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
        adapter._response_store_home = str(tmp_path)
        response = MagicMock()
        async def write(data):
            if (data.startswith(b'event: hermes.artifacts\n') or
                    (failure_at == 'role' and b'"role": "assistant"' in data)):
                raise ConnectionResetError('owned receipt write lost')
        response.write = AsyncMock(side_effect=write)
        prepare = AsyncMock(side_effect=ConnectionResetError('owned headers lost')) if failure_at == 'prepare' else AsyncMock(return_value=response)
        monkeypatch.setattr(adapter, '_prepare_sse_response', prepare)
        monkeypatch.setattr('gateway.platforms.api_server._abandon_agent_task', AsyncMock())
        queue = ThreadSafeAsyncQueue()
        async def agent():
            queue.put_nowait(None)
            return {'final_response': f'MEDIA:{png}', 'completed': True}, None
        origin = {'subject_id': 'owner', 'chat_id': 'chat', 'message_id': 'assistant', 'user_message_id': 'input'}
        task = asyncio.create_task(agent())
        if failure_at != 'receipt':
            await task  # Producer is already complete before downstream failure.
        operation = adapter._write_sse_chat_completion(MagicMock(), 'id', 'model', 1, queue,
            task, [None], artifact_origin=origin, artifact_root=tmp_path)
        if failure_at == 'prepare':
            with pytest.raises(ConnectionResetError):
                await operation
        else:
            await operation
        saved = CompletedReceiptStore(tmp_path / 'authored-receipts.sqlite3').get(origin)
        assert saved is not None, 'Completed receipt vanished when its SSE write failed'
        assert base64.b64decode(saved['artifacts'][0]['data_base64']) == png.read_bytes()
    asyncio.run(exercise())


def test_signed_receipt_lookup_reuses_exact_origin_without_agent_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    import asyncio
    import json
    from unittest.mock import AsyncMock
    from agent.chat_location import native_context, sign_context
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
    from gateway.platforms.api_server_artifacts import CompletedReceiptStore

    async def exercise():
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
        handler = getattr(adapter, '_handle_authored_receipt', None)
        assert handler is not None, 'No authenticated completed-receipt recovery operation'
        adapter._response_store_home = str(tmp_path)
        monkeypatch.setattr(adapter, '_check_auth', lambda request: None)
        monkeypatch.setattr(adapter, '_expected_api_key', lambda: 'fixture-key')
        monkeypatch.setattr(adapter, '_run_agent', AsyncMock(side_effect=AssertionError('replay')))
        ids = {'chat_id': '11111111-1111-4111-8111-111111111111',
               'message_id': '22222222-2222-4222-8222-222222222222',
               'user_message_id': '33333333-3333-4333-8333-333333333333'}
        origin = {'subject_id': 'owner', **ids}
        receipt = {'schema': 'hermes.artifacts.v1', 'origin': origin, 'artifacts': [], 'errors': []}
        CompletedReceiptStore(tmp_path / 'authored-receipts.sqlite3').save(origin, receipt)
        body = {'hermes_artifact_transport': 'inline-v1'}
        context = native_context(user_id='owner', metadata=ids)
        body['hermes_chat_context'] = sign_context(context, body=body, user_id='owner',
                                                  profile='homelab-coordinator', key='fixture-key')
        request = type('Request', (), {'json': AsyncMock(return_value=body)})()
        token = _api_request_profile.set('homelab-coordinator')
        try:
            response = await handler(request)
            assert response.status == 200
            assert json.loads(response.text) == {'status': 'completed', 'receipt': receipt}
            body['hermes_artifact_transport'] = 'tampered'
            assert (await handler(request)).status == 400
        finally:
            _api_request_profile.reset(token)
        adapter._run_agent.assert_not_called()
    asyncio.run(exercise())


def test_receipt_total_bytes_refuse_without_replacing_retained_outputs(tmp_path):
    from gateway.platforms.api_server_artifacts import CompletedReceiptStore
    store = CompletedReceiptStore(tmp_path / 'receipts.sqlite3')
    store.MAX_TOTAL_BYTES = 1
    origin = {'subject_id': 'owner', 'chat_id': 'chat', 'message_id': 'assistant', 'user_message_id': 'input'}
    receipt = {'schema': 'hermes.artifacts.v1', 'origin': origin, 'artifacts': [], 'errors': []}
    with pytest.raises(RuntimeError, match='retention'):
        store.save(origin, receipt, now=100)
    assert store.get(origin, now=101) is None


@pytest.mark.parametrize('flags', [{'completed': False}, {'completed': True, 'partial': True},
                                  {'completed': True, 'failed': True}])
def test_interrupted_or_partial_turn_never_promotes_draft(tmp_path, flags):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._response_store_home = str(tmp_path)
    draft = tmp_path / 'draft.txt'
    draft.write_text('Unfinished output')
    origin = {'subject_id': 'owner', 'chat_id': 'chat', 'message_id': 'assistant', 'user_message_id': 'input'}
    event = adapter._seal_authored_result({**flags, 'final_response': f'MEDIA:{draft}'}, origin, tmp_path, '')
    assert event is None, 'Interrupted-turn draft was promoted as a completed receipt'
    assert not (tmp_path / 'authored-receipts.sqlite3').exists()
