"""GXTD-848: admitted machinery must not become an automatic memory turn."""
import asyncio
from functools import partial
from types import SimpleNamespace

import pytest

from tests.agent.test_memory_sync_redaction import _manager
from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner


from tests.gateway.session_claims import claim


def _admissions(authority, session_id):
    """The session's admission rows in seq order, as its store lists them."""
    from gateway.session_authority import admission_row
    return [admission_row(record) for record in authority.persistence.admissions.list(
        session_id=session_id, pending_only=False)]


@pytest.mark.asyncio
async def test_completion_does_not_sync_but_identical_human_quote_does(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission
    from tests.gateway.test_completion_admission import pending

    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(
        text='human', source=source, message_id='human'))
    sid = receipt.ref.session_id
    key = runner.session_store._generate_session_key(source)
    human_row, = _admissions(authority, sid)
    manager, provider = _manager()
    seen = []

    async def turn(event):
        from gateway.run_turn_runner import TurnRunner
        from gateway.turn_context import TurnContext
        seen.append(event.internal)
        def conversation(message, *, turn_author=None, **kwargs):
            assert turn_author['is_bot'] is False
            manager.sync_all(message, 'response', turn_author=turn_author)
            return {'final_response': 'ok', 'messages': []}
        turn_runner = TurnRunner(runner, TurnContext(
            message=event.text, session_id=sid, session_key=key, source=event.source))
        await runner._run_in_executor_with_context(partial(
            turn_runner._run_conversation_with_approval,
            SimpleNamespace(run_conversation=conversation), [], None, None, None))
        assert await asyncio.to_thread(manager.flush_pending, 5)
        return 'ok'

    monkeypatch.setattr(runner, '_handle_message', turn)
    try:
        # FIFO: the claim only starts the queue's head, so the human turn runs before the completion.
        human_claim = claim(authority, receipt.ref, human_row)
        await execute_admission(authority, receipt.ref, human_claim)
        assert seen == [False]
        # No drain runs in this test: settle the executed row so the next admission can be claimed.
        authority.persistence.admissions.settle(
            epoch=authority.epoch, admission_id=human_claim['admission_id'],
            generation=human_claim['generation'], outcome='completed')
        assert provider.seen[0][0] == 'human', 'human authorship did not sync'
        assert await runner._deliver_async_delegation_group([
            dict(pending(key, 'memory-completion'), parent_session_id=sid)]) is True
        row = _admissions(authority, sid)[-1]
        completion_claim = claim(authority, SessionRef(authority.profile_id, sid), row)
        await execute_admission(authority, SessionRef(authority.profile_id, sid), completion_claim)
        assert seen == [False, True]
        assert len(provider.seen) == 1, 'runtime completion was automatically retained'
        authority.persistence.admissions.settle(
            epoch=authority.epoch, admission_id=completion_claim['admission_id'],
            generation=completion_claim['generation'], outcome='completed')
        human = await _admit_human(runner, authority, MessageEvent(
            text=row['payload']['text'], source=source, message_id='quoted-completion'))
        human_row = next(r for r in _admissions(authority, sid)
                         if r['admission_id'] == human.admission_id)
        await execute_admission(authority, human.ref, claim(authority, human.ref, human_row))
        assert seen == [False, True, False]
        assert len(provider.seen) == 2
        assert provider.seen[1][0] == row['payload']['text']
    finally:
        manager.shutdown_all()
        await runner._cancel_process_completion_batch_tasks()


@pytest.mark.asyncio
async def test_api_dispatch_internal_flag_does_not_suppress_human_memory(api, owner):
    from gateway.session_api_turn import admit_api_turn
    from gateway.session_ingress import execute_admission
    from gateway.session_results import execution_result

    _, ref, row = admit_api_turn(api, session_id='memory-api', user_message='human API input',
                                 conversation_history=[])
    manager, provider = _manager()
    async def handle(event):
        assert event.internal is True
        await asyncio.to_thread(manager.sync_all, event.text, 'response')
        assert await asyncio.to_thread(manager.flush_pending, 5)
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    owner.runner._handle_message = handle
    try:
        await execute_admission(owner, ref, claim(owner, ref, row))
        assert len(provider.seen) == 1
        assert provider.seen[0][0] == 'human API input'
    finally:
        manager.shutdown_all()


@pytest.mark.asyncio
async def test_local_completion_scope_restores_after_handler_failure(tmp_path, monkeypatch):
    from gateway.session_ingress import execute_admission
    from tests.gateway.test_session_end_delegations import _local, _deliver, _DELEGATION

    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    authority.sessions[ref.session_id].subscribers['watching'] = actor
    await _deliver(runner, authority, ref, _DELEGATION)
    row = _admissions(authority, ref.session_id)[-1]
    assert 'local_automation_v1' in row['payload']
    manager, provider = _manager()

    async def handle(event):
        assert event.internal is True
        await runner._run_in_executor_with_context(partial(manager.sync_all, event.text, 'response'))
        assert await asyncio.to_thread(manager.flush_pending, 5)
        assert provider.seen == []
        raise RuntimeError('fixture handler failure')

    monkeypatch.setattr(runner, '_handle_message', handle)
    try:
        with pytest.raises(RuntimeError, match='fixture handler failure'):
            await execute_admission(authority, ref, claim(authority, ref, row))
        manager.sync_all('ordinary input after failed notification', 'response')
        assert manager.flush_pending(5)
        assert len(provider.seen) == 1
    finally:
        manager.shutdown_all()
        await runner._cancel_process_completion_batch_tasks()
        authority.persistence.close()
