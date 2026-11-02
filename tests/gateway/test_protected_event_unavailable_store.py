"""B2-R1: real owning-store failure must not authorize a warm history cut."""
import asyncio
import copy
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_protected_event_ingress import ingress, post, submission


@pytest.mark.parametrize('protected', [True, False], ids=['protected', 'ordinary'])
@pytest.mark.parametrize('warm', [False, True], ids=['empty-cut', 'warm-cut'])
def test_unavailable_owner_refuses_cut_before_admission_and_survives_recovery(tmp_path, monkeypatch, protected, warm):
    async def run():
        async with ingress(tmp_path) as channel:
            db = channel[2]
            if protected:
                assert (await post(channel, submission())).status == 201
            else:
                db.append_message('source-session', 'user', 'ordinary first ask')
            db.append_messages_batch('source-session', [
                {'role': 'assistant', 'content': 'acknowledged'},
                {'role': 'user', 'content': 'next ask'},
                {'role': 'assistant', 'content': 'next answer'}])
            home = tmp_path / 'owner'
            home.mkdir()
            (home / 'state.db').symlink_to(tmp_path / 'events.db')
            assert (home / 'state.db').resolve() == (tmp_path / 'events.db').resolve()
            loaded = db.get_messages_as_conversation('source-session', include_row_ids=True, repair_alternation=True)
            before = db.get_messages('source-session', include_inactive=True)
            session_before = db.get_session('source-session')
            from tui_gateway import server
            from hermes_cli import backend_retirement
            import hermes_state_registry
            monkeypatch.setattr(backend_retirement, 'retirement', backend_retirement.RetirementFence())
            monkeypatch.setattr(server, '_db', db)
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
                fault.setattr(hermes_state_registry, 'acquire', unavailable)
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
            assert 'last_active' not in session
            assert agent._session_messages == loaded
            server._adopt_out_of_band_turns(session)
            assert session['history'] == loaded
            assert db.get_messages('source-session', include_inactive=True) == before
            assert db.get_session('source-session') == session_before
    asyncio.run(run())
