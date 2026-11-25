"""External maintenance must fence canonical admissions, not only bot ingress."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.session_authority import admission_row
from persistence.conversation.sessions import NewSession, SessionId
from persistence.core.errors import AuthorityRefusal
from persistence_assembly import open_persistence, profile_scope


@pytest.mark.asyncio
async def test_external_drain_refuses_canonical_submit_without_durable_input(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionActor, SessionRef, Submission
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = open_persistence(profile_scope(home=tmp_path))
    store.sessions.create(NewSession(SessionId('s'), 'telegram'))

    def admissions():
        """The session's admission rows, as the store lists them."""
        return [admission_row(row) for row in store.admissions.list(session_id='s')]

    ran = []

    async def handle(event):
        ran.append(event)
        return 'fixture response'

    runner = SimpleNamespace(_draining=False, _external_drain_active=True, _running=True,
                             config=SimpleNamespace(multiplex_profiles=False),
                             session_runtime_descriptor={'state': 'draining'},
                             _adapter_for_source=lambda source: None, _handle_message=handle)
    authority = SessionAuthority(runner, profile_id='p', instance_id='owner', persistence=store,
                                 epoch=store.runtime.begin_epoch(instance_id='owner'))
    authority.sessions['s'] = LiveSession(
        SessionSource(platform=Platform.TELEGRAM, chat_id='c', user_id='human'), 'route')
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')
    try:
        with pytest.raises(AuthorityRefusal, match='runtime_draining'):
            await authority.submit(actor, Submission('r1', SessionRef('p', 's'), {'text': 'next'}, 'queue'),
                                   queue_while_restarting=True)
        assert admissions() == []
        assert ran == []
        connection = AuthorityConnection(authority, object(), {'user_id': 'human'}, operator=True)
        try:
            resumed = await connection.dispatch({'id': 1, 'method': 'session.resume',
                                                 'params': {'session_id': 's'}})
            assert 'result' in resumed
            request = {'id': 2, 'method': 'prompt.submit', 'params': {
                'session_id': 's', 'submission_id': 'r2', 'text': 'after maintenance'}}
            refused = await connection.dispatch(request)
            assert refused['error']['message'] == 'runtime_draining'
            assert admissions() == []
            runner._external_drain_active = False
            accepted = await connection.dispatch(request)
            assert 'result' in accepted, accepted
            assert len(admissions()) == 1
        finally:
            await connection.close()
    finally:
        task = authority.sessions['s'].task
        if task is not None:
            await asyncio.wait_for(task, 10)
        store.close()
