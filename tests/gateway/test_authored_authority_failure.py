"""GXTD-755: durable producer settlement outlives its HTTP observer.

Only the external agent execution is scripted; admission, SQLite, drain,
result capture and settlement are the actual runtime owners.
"""
import asyncio

import pytest

from gateway.platforms.api_server_terminal import FailureDispositionStore
from gateway.session_results import execution_result
from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until

ORIGIN = {'subject_id': 'owner', 'chat_id': 'chat', 'message_id': 'reply',
          'user_message_id': 'input'}


@pytest.mark.asyncio
async def test_authority_failure_survives_departed_observer(api, owner, tmp_path):
    entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    result = {'failed': True, 'completed': False, 'failure_reason': 'dispatch_refused',
              'failure_retryable': False, 'final_response': ''}

    async def producer(event):
        entered.set()
        try:
            await release.wait()
            execution_result.get()['result'] = result
            return ''
        finally:
            cleaned.set()

    owner.runner._handle_message = producer
    task = asyncio.create_task(api._run_agent('fixture', [], session_id='authored', authored_origin=ORIGIN))
    try:
        assert await async_wait_until(lambda: entered.is_set() or task.done(), timeout=5)
        if task.done():
            await task
        assert entered.is_set()
        row, = list_session_admissions(owner.db, session_id='authored')
        assert row['status'] == 'started'
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not cleaned.is_set()
        assert not (tmp_path / 'authored-failures.sqlite3').exists()
    finally:
        release.set()
        if not task.done():
            await asyncio.wait_for(task, timeout=5)
        assert await async_wait_until(
            lambda: all(live.task is None or live.task.done() for live in owner.sessions.values()), timeout=5)
    assert cleaned.is_set()
    store = FailureDispositionStore(tmp_path / 'authored-failures.sqlite3')
    disposition = store.get(ORIGIN)
    assert disposition is not None
    assert disposition['reason'] == 'dispatch_refused'
    assert disposition['producer_settled'] is True
    assert disposition['native_retirement'] == 'unproven'
    assert store.get({**ORIGIN, 'subject_id': 'foreign'}) is None
    assert not (tmp_path / 'authored-receipts.sqlite3').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('flags', [
    {'completed': True, 'final_response': 'dispatch_refused'},
    {'failed': True, 'completed': False, 'failure_reason': 'other', 'failure_retryable': False},
    {'failed': True, 'completed': False, 'failure_reason': 'dispatch_refused', 'failure_retryable': True},
    {'failed': True, 'completed': False, 'failure_reason': 'dispatch_refused',
     'failure_retryable': False, 'partial': True},
])
async def test_authority_does_not_infer_failure_from_other_outcomes(api, owner, tmp_path, flags):
    async def producer(event):
        execution_result.get()['result'] = flags
        return ''
    owner.runner._handle_message = producer
    await api._run_agent('fixture', [], session_id='other', authored_origin=ORIGIN)
    assert not (tmp_path / 'authored-failures.sqlite3').exists()


def test_durable_origin_is_validated_and_not_session_settings(api, owner):
    from gateway.session_api_turn import admit_api_turn, api_settings, check_api_turn
    from hermes_state_runtime import RuntimeStoreError
    _, ref, row = admit_api_turn(api, user_message='fixture', conversation_history=[],
                                session_id='bound', authored_origin=ORIGIN)
    assert row['payload']['api_turn_v1']['authored_origin'] == ORIGIN
    assert 'authored_origin' not in api_settings(owner, ref)
    row['payload']['api_turn_v1']['authored_origin'] = {**ORIGIN, 'profile': 'foreign'}
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        check_api_turn(owner, ref, row['payload'])
