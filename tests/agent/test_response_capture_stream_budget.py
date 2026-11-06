"""Long streams must retain exact delta sequence and terminal attribution."""
import json
import time
from types import SimpleNamespace as NS

import httpx
import pytest
from agent import response_capture as r, selection_capture as s


def test_long_stream_preserves_all_deltas_and_terminal_records(tmp_path):
    c = r.ResponseCapture(tmp_path / 'capture', time.time()+60,
        identity=dict(profile='synthetic', subject='owner', chat='chat', root_session_id='root'),
        campaign_id='long-stream')
    pieces = [f'{n:x}ü' for n in range(8192)]
    with s.capture_scope(c), r.selection_scope('source'):
        with r.provider_scope('compression', NS(base_url=httpx.URL('http://offline.invalid/v1/'))):
            for piece in pieces:
                r.observe_sdk(NS(choices=[NS(delta=NS(content=piece), finish_reason=None)]), delta=True)
        r.observe_sdk(NS(choices=[NS(message=NS(content=''.join(pieces)), finish_reason='stop')]))
        r.extraction_observer()('source', 'content')
        s.observe_selection('source', '{"excerpts":["source"]}')
    events = [json.loads(p.read_text()) for p in sorted(c.root.glob('event-*.json'))]
    assert c.omitted == c.errors == 0, c.status()
    deltas = [v for e in events if e['kind'] == 'sdk_delta_batch' for v in e['data']['deltas']]
    assert [v['content'] for v in deltas] == pieces
    assert [e['kind'] for e in events][-3:] == ['summary_response', 'extraction', 'selection']
    assert c.events <= 128 and c.bytes <= 1048576
    assert c.bytes == sum(p.stat().st_size for p in c.root.glob('event-*.json'))


@pytest.mark.parametrize('mode', ['byte_overflow', 'writer_failure', 'expiry', 'cancel', 'late'])
def test_partial_batch_refuses_without_primary_change(tmp_path, monkeypatch, mode):
    from contextvars import copy_context
    c = r.ResponseCapture(tmp_path / 'capture', time.time()+60,
        identity={}, campaign_id=mode)
    marker = RuntimeError('original provider failure')
    original_open = r.os.open
    def open_owned(path, *args, **kwargs):
        if mode == 'writer_failure' and str(path).endswith('.json') and 'event-' in str(path):
            raise OSError('synthetic disk failure')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(r.os, 'open', open_owned)
    with s.capture_scope(c), r.selection_scope('source'):
        try:
            with r.provider_scope('compression', NS(base_url=httpx.URL('http://offline.invalid/v1/'))):
                saved = copy_context()
                delta = NS(choices=[NS(delta=NS(content='ü'*128), finish_reason=None)])
                for _ in range(2400 if mode == 'byte_overflow' else 1):
                    r.observe_sdk(delta, delta=True)
                charged = c.bytes
                assert charged <= 1048576
                if mode == 'expiry':
                    c.deadline = time.monotonic()-1
                if mode == 'cancel':
                    raise marker
        except RuntimeError as error:
            assert mode == 'cancel' and error is marker
        if mode == 'late':
            before = set(c.root.glob('event-*.json'))
            saved.run(r.observe_sdk, delta, delta=True)
            assert set(c.root.glob('event-*.json')) == before
    assert c.bytes >= charged and c.bytes <= 1048576
    if mode == 'writer_failure':
        assert c.errors > 0 and not list(c.root.glob('event-*.json'))
    elif mode in ('byte_overflow', 'expiry', 'late'):
        assert c.omitted > 0
    else:
        assert c.omitted == c.errors == 0
    status = json.loads((c.root/'status.json').read_text())
    assert status['authorizes_execution'] is False
    if mode != 'cancel':
        assert status['attribution'] == 'incomplete'


def test_retry_flushes_under_previous_transport_ordinal(tmp_path):
    c = r.ResponseCapture(tmp_path / 'capture', time.time()+60, identity={}, campaign_id='retry')
    with s.capture_scope(c), r.selection_scope('source'):
        with r.provider_scope('compression', NS(base_url=httpx.URL('http://offline.invalid/v1/'))) as hook:
            request = httpx.Request('POST', 'http://offline.invalid/v1/chat/completions',
                json={'model': 'fixture', 'messages': [{'role': 'user', 'content': 'source'}]})
            hook(request)
            r.observe_sdk(NS(choices=[NS(delta=NS(content='first'), finish_reason=None)]), delta=True)
            hook(request)
            r.observe_sdk(NS(choices=[NS(delta=NS(content='second'), finish_reason='stop')]), delta=True)
    events = [json.loads(p.read_text()) for p in sorted(c.root.glob('event-*.json'))]
    assert [(e['kind'], e['transport_ordinal']) for e in events] == [
        ('transport_request', 1), ('sdk_delta_batch', 1),
        ('transport_request', 2), ('sdk_delta_batch', 2)]
    assert c.omitted == c.errors == 0
