"""Explicit reload must recover cold-deferred servers, not weaken startup/TTL."""
from contextlib import nullcontext
from unittest.mock import Mock


def test_explicit_activation_only_wakes_enabled_cold_deferred(monkeypatch):
    from tools import mcp_tool_discovery as d, mcp_tool as core
    monkeypatch.setattr(d, '_owner_secret_scope', nullcontext)
    monkeypatch.setattr(d._config, '_load_mcp_config', lambda: {
        'mail': {'lazy': True}, 'disabled': {'enabled': False}, 'foreign': {'lazy': True}})
    monkeypatch.setattr(d, '_server_key', lambda n: ('owner', n))
    monkeypatch.setattr(core, '_lazy_server_configs', {
        ('owner', 'mail'): {}, ('owner', 'disabled'): {}, ('other', 'foreign'): {}})
    monkeypatch.setattr(core, '_servers', {})
    monkeypatch.setattr(core, '_server_connecting', set())
    monkeypatch.setattr(core, '_server_connect_errors', {
        ('owner', 'mail'): 'deferred: schema cache unavailable; reconnect explicitly',
        ('owner', 'disabled'): 'deferred: schema cache unavailable; reconnect explicitly'})
    wake = Mock(return_value=True)
    monkeypatch.setattr(d, '_ensure_lazy_server_connected', wake)
    assert d.activate_deferred_mcp_servers() == ['mail']
    wake.assert_called_once_with('mail')


def test_explicit_activation_preserves_failure_and_live_servers(monkeypatch):
    from tools import mcp_tool_discovery as d, mcp_tool as core
    monkeypatch.setattr(d, '_owner_secret_scope', nullcontext)
    monkeypatch.setattr(d._config, '_load_mcp_config', lambda: {'mail': {}, 'failed': {}})
    monkeypatch.setattr(d, '_server_key', lambda n: n)
    monkeypatch.setattr(core, '_lazy_server_configs', {'mail': {}, 'failed': {}})
    monkeypatch.setattr(core, '_servers', {'mail': object()})
    monkeypatch.setattr(core, '_server_connecting', set())
    monkeypatch.setattr(core, '_server_connect_errors', {'failed': 'connection refused'})
    wake = Mock()
    monkeypatch.setattr(d, '_ensure_lazy_server_connected', wake)
    assert d.activate_deferred_mcp_servers() == []
    wake.assert_not_called()


def test_zero_ttl_still_expires(tmp_path, monkeypatch):
    from tools import mcp_schema_cache as c
    monkeypatch.setattr(c, '_cache_path', lambda: tmp_path / 'cache.json')
    c.write_cache_entry('mail', 'fixture', tools=[], ttl_ms=0)
    assert c.get_cached_entry('mail', 'fixture') is None


def test_gateway_reload_activates_before_reporting_unavailable(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from gateway import session_ancillary as gateway
    from tools import mcp_tool_discovery as d
    ready = []
    monkeypatch.setattr(d, 'reconcile_mcp_servers_with_config',
                        lambda: {'added': [], 'removed': [], 'pending': []})
    def activate():
        ready.append('mail')
        return ['mail']
    monkeypatch.setattr(d, 'activate_deferred_mcp_servers', activate)
    monkeypatch.setattr(d, 'mcp_servers_not_connected', lambda: [] if ready else ['mail'])
    monkeypatch.setattr(d, 'mcp_servers_ready', lambda: ready)
    monkeypatch.setattr('gateway.session_authorities.served_profile_name', lambda _: 'default')
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False),
                             _mcp_reload_refresh_cached_agents=lambda *args: None)
    connection = SimpleNamespace(actor=SimpleNamespace(capabilities={'session:control'}),
                                 authority=SimpleNamespace(runner=runner, profile_id='fixture'))
    result = asyncio.run(gateway.reload_mcp(connection, None, {'confirm': True}))
    assert result['status'] == 'reloaded'
    assert 'added: mail' in result['message']
    assert 'unavailable' not in result


def test_gateway_confirmation_does_not_activate(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from gateway import session_ancillary as gateway
    from tools import mcp_tool_discovery as d
    wake = Mock()
    monkeypatch.setattr(d, 'activate_deferred_mcp_servers', wake)
    connection = SimpleNamespace(actor=SimpleNamespace(capabilities={'session:control'}))
    result = asyncio.run(gateway.reload_mcp(connection, None, {}))
    assert result['status'] == 'confirm_required'
    wake.assert_not_called()
