"""Existing maintenance markers must precede startup recovery/connect work."""
from types import SimpleNamespace
import json
from unittest.mock import AsyncMock, Mock
import pytest
from gateway.run_startup import GatewayStartupMixin
from gateway.drain_control import write_drain_request
from gateway.session_authority import SessionAuthority
from persistence.core.errors import AuthorityRefusal


@pytest.mark.asyncio
@pytest.mark.parametrize('marker', ['active', 'absent', 'stale'])
async def test_startup_admission_before_recovery(tmp_path, monkeypatch, marker):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    if marker != 'absent':
        write_drain_request(home=tmp_path, principal='offline-fixture')
    if marker == 'stale':
        path = tmp_path / '.drain_request.json'
        body = json.loads(path.read_text())
        body['epoch'] = 'different-boot'
        path.write_text(json.dumps(body))
    runner = SimpleNamespace(
        _external_drain_active=False, _draining=False, _startup_parked_platforms=False,
        _start_install_faulthandler=Mock(),
        _start_log_startup_environment=AsyncMock(),
        _abort_startup_if_shutdown_requested=AsyncMock(return_value=False),
        _start_check_access_policy=Mock(return_value=False),
    )
    class ObservedBoundary(Exception):
        pass
    async def recovery():
        authority = SimpleNamespace(retiring=False, runner=runner)
        assert GatewayStartupMixin._serving_state(runner) == ('draining' if marker == 'active' else 'running')
        if marker == 'active':
            with pytest.raises(AuthorityRefusal, match='runtime_draining'):
                SessionAuthority._require_admission_open(authority)
        else:
            SessionAuthority._require_admission_open(authority)
        raise ObservedBoundary
    runner._start_recover_previous_run = recovery
    with pytest.raises(ObservedBoundary):
        await GatewayStartupMixin._start_impl(runner)
