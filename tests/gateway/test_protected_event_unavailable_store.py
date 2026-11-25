"""B2-R1: real owning-store failure must not authorize a warm history cut."""
import asyncio
import copy
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_protected_event_ingress import ingress, post, submission, persist_history, load_history
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView


@pytest.mark.parametrize('protected', [True, False], ids=['protected', 'ordinary'])
@pytest.mark.parametrize('warm', [False, True], ids=['empty-cut', 'warm-cut'])
def test_unavailable_owner_refuses_cut_before_admission_and_survives_recovery(tmp_path, monkeypatch, protected, warm):
    async def run():
        async with ingress(tmp_path) as channel:
            db = channel[2]
            if protected:
                assert (await post(channel, submission())).status == 201
            else:
                db.messages.append('source-session', NewMessage('user', 'ordinary first ask'))
            persist_history(db, 'source-session', [
                {'role': 'assistant', 'content': 'acknowledged'},
                {'role': 'user', 'content': 'next ask'},
                {'role': 'assistant', 'content': 'next answer'}])
            home = tmp_path
            loaded = load_history(db, 'source-session', include_row_ids=True, repair_alternation=True)
            before = db.messages.read_transcript('source-session', TranscriptRead(view=TranscriptView.ALL))
            session_before = db.sessions.get('source-session')
            from tui_gateway import server
            from hermes_cli import backend_retirement
            import persistence_assembly
            monkeypatch.setattr(backend_retirement, 'retirement', backend_retirement.RetirementFence())
            monkeypatch.setattr(server, '_db', db)
            owner_activity = []
            require_owner = server._require_transcript_owner
            def observed_owner(record):
                require_owner(record)
                owner_activity.append(record['last_active'])
            monkeypatch.setattr(server, '_require_transcript_owner', observed_owner)
            agent = SimpleNamespace(_session_messages=copy.deepcopy(loaded))
            session = {'session_key': 'source-session', 'profile_home': str(home),
                       'history': copy.deepcopy(loaded), 'history_version': 7,
                       'history_lock': threading.RLock(), 'agent': agent, 'running': False}
            target = [row for row in loaded if row['role'] == 'user'][1 if warm else 0]
            params = {'truncate_before_row_id': target['_row_id'], 'confirm_truncate': True,
                      'confirm_empty_truncate': True}
            warm_before = session['history']
            with monkeypatch.context() as fault:
                def unavailable(*args, **kwargs):
                    raise OSError('disposable owning DB open failure')
                fault.setattr(persistence_assembly, 'open_persistence', unavailable)
                error, fields = server._lock_in_submit_turn(
                    'request', 'source-session', session, 'replacement', params, True, None, None, None)
            assert error is not None, 'unavailable owning store admitted a destructive edit'
            assert error['error']['code'] == 5008
            assert fields == {}
            assert session['history'] is warm_before
            assert session['history'] == loaded
            assert session['history_version'] == 7
            assert session['running'] is False
            assert 'inflight_turn' not in session
            assert '_turn_cancel_requested' not in session
            # Upstream owner reclamation refreshes activity even on refusal to
            # prevent idle lease release. No submit-time activity may replace it.
            assert owner_activity and session['last_active'] == owner_activity[-1]
            assert agent._session_messages == loaded
            server._adopt_out_of_band_turns(session)
            assert session['history'] == loaded
            assert db.messages.read_transcript('source-session', TranscriptRead(view=TranscriptView.ALL)) == before
            assert db.sessions.get('source-session') == session_before
    asyncio.run(run())
