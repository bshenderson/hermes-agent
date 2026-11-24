"""GXTD-900: unresolved launch models must not leave unrestorable receipts."""
from types import SimpleNamespace
import pytest


@pytest.mark.asyncio
async def test_missing_model_refuses_before_persisting_session(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection


    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text('platform_toolsets:\n  cli: []\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: None)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
    reply = await connection.dispatch({'id': 1, 'method': 'session.create', 'params': {
        'request_id': 'missing-model', 'source': 'cli', 'cwd': str(tmp_path)}})
    assert 'error' in reply, reply
    assert 'model' in str(reply['error']).lower()
    with authority.db._read_ctx() as conn:
        assert conn.execute("SELECT count(*) FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM sessions WHERE source='cli'").fetchone()[0] == 0
