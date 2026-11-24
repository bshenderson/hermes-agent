"""Independent-review failure branches; synthetic credentials and transports only."""
import builtins
from contextlib import asynccontextmanager
import sys
import pytest


@pytest.fixture
def lazy_home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('''secrets:
  lazy: true
  startup_names:
    gateway: [API_SERVER_KEY]
  onepassword:
    enabled: true
    override_existing: true
    env:
      API_SERVER_KEY: op://fixture/api/key
''')
    from hermes_cli import env_loader
    env_loader.reset_secret_source_cache()
    return tmp_path


@pytest.mark.parametrize('failure', ['import', 'apply'])
def test_required_source_errors_refuse(lazy_home, monkeypatch, failure):
    from hermes_cli import env_loader
    from agent.secret_sources import registry
    from agent.lazy_secrets import SecretUnavailable
    monkeypatch.setattr(sys, 'argv', ['hermes', 'gateway', 'run'])
    monkeypatch.setenv('API_SERVER_KEY', 'synthetic-stale')
    if failure == 'import':
        original = builtins.__import__
        def reject(name, *args, **kwargs):
            if name == 'agent.secret_sources.registry':
                raise ImportError('synthetic unavailable module')
            return original(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', reject)
    else:
        def reject_apply(*args, **kwargs):
            raise RuntimeError('synthetic source failure')
        monkeypatch.setattr(registry, 'apply_all', reject_apply)
    with pytest.raises(SecretUnavailable, match='no credential fallback'):
        env_loader._apply_external_secret_sources(lazy_home)


def test_malformed_secret_config_refuses_stale_credentials(lazy_home, monkeypatch):
    from hermes_cli import env_loader
    from agent.secret_scope import get_secret
    from agent.lazy_secrets import SecretUnavailable
    monkeypatch.setenv('API_SERVER_KEY', 'synthetic-stale')
    (lazy_home / 'config.yaml').write_text('secrets: [broken:')
    with pytest.raises(SecretUnavailable):
        env_loader._apply_external_secret_sources(lazy_home)
    with pytest.raises(SecretUnavailable):
        get_secret('API_SERVER_KEY')


def test_managed_credential_precedes_lazy_user_source(lazy_home, monkeypatch):
    from agent.secret_scope import get_secret, set_secret_scope, reset_secret_scope
    from hermes_cli import managed_scope, env_loader
    monkeypatch.setattr(managed_scope, 'load_managed_env', lambda: {'API_SERVER_KEY': 'synthetic-admin'})
    def forbidden(*args, **kwargs):
        pytest.fail('user source ran ahead of managed authority')
    monkeypatch.setattr(env_loader, 'resolve_profile_secret_names', forbidden)
    token = set_secret_scope({'API_SERVER_KEY': 'synthetic-admin'}, profile_home=str(lazy_home))
    try:
        assert get_secret('API_SERVER_KEY') == 'synthetic-admin'
    finally:
        reset_secret_scope(token)


def test_deferred_credential_does_not_skip_dependency(lazy_home):
    from agent.lazy_secrets import probe_requirements
    from agent.secret_scope import get_secret
    visited = []
    def probe():
        return bool(get_secret('API_SERVER_KEY')) and (visited.append(True) or False)
    assert not probe_requirements(probe)
    assert visited == [True]


@pytest.mark.parametrize('failure', ['import', 'lookup', 'register', 'empty'])
def test_mcp_cache_failure_never_overrides_cold_defer(lazy_home, monkeypatch, failure):
    from tools import mcp_tool_discovery as discovery, mcp_schema_cache, mcp_tool as core
    for name in ('_lazy_server_configs', '_server_connect_errors'):
        monkeypatch.setattr(core, name, {})
    monkeypatch.setattr(core, '_server_connecting', set())
    def reject(*args, **kwargs):
        raise RuntimeError('synthetic cache failure')
    if failure == 'import':
        original = builtins.__import__
        def reject_import(name, *args, **kwargs):
            if name == 'tools.mcp_schema_cache':
                raise ImportError('synthetic unavailable cache')
            return original(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', reject_import)
    elif failure == 'lookup':
        monkeypatch.setattr(mcp_schema_cache, 'get_cached_entry', reject)
    else:
        monkeypatch.setattr(mcp_schema_cache, 'get_cached_entry', lambda *a: {'tools': []})
        monkeypatch.setattr(discovery._registration, '_register_from_cache_sync',
                            (lambda *a: []) if failure == 'empty' else reject)
    cfg = {'url': 'https://example.invalid/mcp', 'lazy': True, 'lazy_cold': 'defer'}
    eager, tools, servers = discovery._register_lazy_from_cache({'fixture': cfg})
    assert eager == {}
    assert tools == 0 and servers == 1
    assert list(core._lazy_server_configs.values()) == [cfg]


def test_tui_preflight_uses_shared_explicit_route_connector(monkeypatch):
    from hermes_cli import main_tui_launch, startup_budget, gateway_client
    monkeypatch.setenv('HERMES_TUI_GATEWAY_URL', 'wss://example.invalid/gateway')
    monkeypatch.setattr(startup_budget, 'active', lambda: True)
    seen = []
    @asynccontextmanager
    async def connect():
        seen.append('selected-connector')
        yield object()
    monkeypatch.setattr(gateway_client, 'connect_gateway', connect)
    monkeypatch.setattr(startup_budget, 'ready', lambda: seen.append('ready'))
    main_tui_launch._prepare_tui_gateway()
    assert seen == ['selected-connector', 'ready']
