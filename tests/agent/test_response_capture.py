"""Finite v2 capture, ownership lifetime and original extraction behavior."""
from contextvars import copy_context
import json
import threading
import time
from types import SimpleNamespace as NS

import httpx
import pytest

from agent import response_capture as r, selection_capture as s
from agent.auxiliary_client import extract_content_or_reasoning


def campaign(tmp_path):
    return r.ResponseCapture(tmp_path / 'capture', time.time()+60,
        identity=dict(profile='synthetic', subject='owner', chat='chat', root_session_id='root'),
        campaign_id='synthetic')


def records(c):
    return [json.loads(p.read_text()) for p in sorted(c.root.glob('event-*.json'))]


def response(**fields):
    return NS(choices=[NS(message=NS(**fields), finish_reason='stop')])


@pytest.mark.parametrize('gate', ['expired', 'bytes', 'events', 'closed_call', 'closed_request'])
def test_shared_budget_and_dead_scope(tmp_path, gate):
    c = campaign(tmp_path)
    with s.capture_scope(c), r.selection_scope('synthetic source'):
        ctx = copy_context()
        if gate == 'expired':
            c.deadline = 0
        elif gate == 'bytes':
            c.bytes = 1048576
        elif gate == 'events':
            c.events = 128
        if gate not in ('closed_call', 'closed_request'):
            r.observe_sdk(response(content='synthetic'))
            assert c.omitted == 1 and records(c) == []
            return
    ctx.run(r.observe_sdk, response(content='late'))
    assert c.omitted == 1 and records(c) == []
    assert c.closed_requests == 1
    assert r._call.get() is None and s._current.get() is None


@pytest.mark.parametrize('operation', ['open', 'fsync'])
@pytest.mark.parametrize('outcome', ['return', 'error'])
def test_writer_failure_keeps_original_outcome_and_charge(tmp_path, monkeypatch, operation, outcome):
    c = campaign(tmp_path)
    original = RuntimeError('original')
    def fail(*a, **kw):
        raise OSError('injected')
    with s.capture_scope(c), r.selection_scope('synthetic'):
        with monkeypatch.context() as m:
            m.setattr(r.os, operation, fail)
            if outcome == 'return':
                value = extract_content_or_reasoning(response(content=' synthetic '), _observe=r.extraction_observer())
                assert value == 'synthetic'
            else:
                try:
                    raise original
                except RuntimeError as caught:
                    r.observe_error(caught)
                    assert caught is original
        assert c.bytes > 0 and c.errors > 0
    assert len(records(c)) == (1 if operation == 'fsync' else 0)


@pytest.mark.parametrize('content,reasoning,branch,expected,limit,truncated', [
    ('synthetic', None, 'content', 'synthetic', 8000, False),
    ('<think>hidden</think>final', None, 'inline_think_content', 'final', 8000, False),
    ('', 'fallback', 'reasoning_fallback', 'fallback', 8000, False),
    ('', 'fallback', 'reasoning_fallback', 'fall', 4, True),
    ('', None, 'empty', '', 8000, False)])
def test_actual_extraction_branches_are_not_inferred(tmp_path, content, reasoning, branch, expected, limit, truncated):
    c = campaign(tmp_path)
    with s.capture_scope(c), r.selection_scope('synthetic'):
        value = response(content=content, reasoning_content=reasoning)
        before = extract_content_or_reasoning(value, max_reasoning_chars=limit)
        after = extract_content_or_reasoning(value, max_reasoning_chars=limit, _observe=r.extraction_observer())
    assert before == after == expected
    assert records(c)[0]['data'] == dict(selected=expected, branch=branch, truncated=truncated)


def test_one_aggregate_limit_across_multiple_kinds_and_scopes(tmp_path):
    c = campaign(tmp_path)
    for _ in range(2):
        with s.capture_scope(c), r.selection_scope('source'):
            r.observe_sdk(response(content='synthetic'))
            extract_content_or_reasoning(response(content='synthetic'), _observe=r.extraction_observer())
            s.observe_selection('source', '{"excerpts":["source"]}')
    assert c.events == 6
    assert c.bytes == sum(p.stat().st_size for p in c.root.glob('event-*.json'))
    assert c.closed_requests == 2
    assert len({x['request_id'] for x in records(c)}) == 2
    assert c.root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in c.root.iterdir())
    with pytest.raises(FileExistsError):
        campaign(tmp_path)


def test_transport_allowlist_identity_and_ordinal(tmp_path):
    c = campaign(tmp_path)
    url = httpx.URL('http://offline.invalid/v1/')
    body = dict(model='synthetic', messages=[dict(role='user', content='never persist this prompt')],
        api_key='credential-sentinel', unknown='credential-sentinel', response_format={'type': 'json_object'},
        chat_template_kwargs={'enable_thinking': False}, max_tokens=12000)
    request = httpx.Request('POST', str(url) + 'chat/completions', json=body,
        headers={'authorization': 'Bearer credential-sentinel'})
    with s.capture_scope(c), r.selection_scope('synthetic'):
        with r.provider_scope('compression', NS(base_url=url)) as hook:
            assert hook is not None
            hook(request)
            hook(request)
        # Closed HTTP callback must not persist another event.
        hook(request)
        r.observe_sdk(response(content='synthetic', private_key='credential-sentinel'))
    values = records(c)
    assert [v['transport_ordinal'] for v in values[:2]] == [1, 2]
    assert values[0]['provider_attempt_id'] == values[1]['provider_attempt_id']
    assert values[-1]['kind'] == 'summary_response'
    assert c.omitted == 1
    text = json.dumps(values)
    assert 'credential-sentinel' not in text and 'never persist this prompt' not in text
    assert values[0]['data']['messages'][0]['bytes'] > 0


def test_foreign_tasks_and_unscoped_calls_never_capture(tmp_path):
    c = campaign(tmp_path)
    r.observe_sdk(response(content='outside'))
    with s.capture_scope(c):
        r.observe_sdk(response(content='ordinary main/title'))
        with r.selection_scope('synthetic'):
            with r.provider_scope('title_generation', NS()) as hook:
                assert hook is None
                r.observe_sdk(NS(choices=[]), delta=True)
                r.unsupported_transport('title_generation')
    assert c.events == 0 and records(c) == []


def test_copied_context_after_request_closure_drops_in_provider_thread(tmp_path):
    c = campaign(tmp_path)
    with s.capture_scope(c), r.selection_scope('synthetic'):
        context = copy_context()
    thread = threading.Thread(target=context.run, args=(r.observe_sdk, response(content='late')))
    thread.start()
    thread.join(2)
    assert not thread.is_alive() and c.omitted == 1 and records(c) == []


def test_other_thread_without_context_is_not_selected(tmp_path):
    c = campaign(tmp_path)
    with s.capture_scope(c), r.selection_scope('synthetic'):
        thread = threading.Thread(target=r.observe_sdk, args=(response(content='foreign'),))
        thread.start()
        thread.join(2)
        r.observe_sdk(response(content='selected'))
    assert c.events == 1
    assert records(c)[0]['data']['content'] == 'selected'


def test_unsupported_body_destination_and_raw_shape_are_observation_errors(tmp_path):
    c = campaign(tmp_path)
    with s.capture_scope(c), r.selection_scope('synthetic'):
        with r.provider_scope('compression', NS(base_url=httpx.URL('http://offline.invalid/v1/'))) as hook:
            assert hook is not None
            hook(httpx.Request('POST', 'http://foreign.invalid/v1/chat/completions', json={'messages': []}))
            hook(httpx.Request('POST', 'http://offline.invalid/v1/chat/completions', content=iter([b'{}'])))
        r.observe_sdk(response(content={'not': 'a raw string'}))
    assert not records(c) and c.errors == 3


def test_observer_callback_exception_cannot_change_extraction():
    def fail(*a):
        raise OSError('injected callback')
    assert extract_content_or_reasoning(response(content='synthetic'), _observe=fail) == 'synthetic'


def test_closed_chunk_drops_late_context_while_request_stays_open(tmp_path):
    c = campaign(tmp_path)
    with s.capture_scope(c):
        with r.selection_scope('first chunk'):
            old = copy_context()
        old.run(r.observe_sdk, response(content='late first chunk'))
        assert c.omitted == 1 and c.closed_requests == 0 and records(c) == []
        with r.selection_scope('second chunk'):
            r.observe_sdk(response(content='current second chunk'))
        assert records(c)[0]['data']['content'] == 'current second chunk'
    assert c.closed_requests == 1
