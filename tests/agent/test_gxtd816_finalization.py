"""Finalization must own its failure, not an ambient caller's exception."""
from types import SimpleNamespace

import pytest

from agent.turn_facade_lease import DurableTurnLease
from run_agent import AIAgent
from tests.agent.test_cross_process_turn_lease import _DB, _agent_with_db


def test_successful_turn_inside_except_surfaces_stop_failure(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    failure = ValueError('stop-failure')
    calls = []

    def stop(self):
        calls.append('stop')
        if len(calls) == 1:
            raise failure

    monkeypatch.setattr(DurableTurnLease, 'stop_refresher', stop)
    monkeypatch.setattr('agent.conversation_loop.run_conversation',
                        lambda *a, **k: {'final_response': 'ok'})
    try:
        raise RuntimeError('unrelated-caller-error')
    except RuntimeError as ambient:
        with pytest.raises(ValueError) as caught:
            AIAgent.run_conversation(agent, 'hello', conversation_history=[])
        assert caught.value is failure
        assert not getattr(ambient, '__notes__', [])
    assert db.events[-1][0] == 'release'


def test_timer_cancellation_attempts_remaining_handles():
    calls = []
    first = KeyboardInterrupt('first-cancel')
    second = ValueError('second-cancel')

    def cancel(name, failure=None):
        def run(**kwargs):
            calls.append(name)
            if failure is not None:
                raise failure
        return SimpleNamespace(cancel=run)

    lease = DurableTurnLease(SimpleNamespace(), None, 'session', 'holder')
    lease.timer_handles = [cancel('first', first), cancel('second', second), cancel('third')]
    with pytest.raises(KeyboardInterrupt) as caught:
        lease.join_threads()
    assert calls == ['first', 'second', 'third']
    assert caught.value is first
    assert any('second-cancel' in note for note in first.__notes__)
