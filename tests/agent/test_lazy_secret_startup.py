"""GXTD-900: actual loader and scoped lookup, synthetic vault only."""
from pathlib import Path
import os
import pytest
import sys


def test_lazy_startup_does_not_fetch_optional_secret(tmp_path, monkeypatch):
    from hermes_cli import env_loader
    from agent.secret_sources import onepassword
    from agent.secret_scope import get_secret
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('AGENTMAIL_API_KEY', 'stale-process-key')
    (tmp_path / 'config.yaml').write_text('''secrets:
  lazy: true
  onepassword:
    enabled: true
    override_existing: true
    env:
      AGENTMAIL_API_KEY: op://fixture/mail/key
      OPENAI_API_KEY: op://fixture/provider/key
''')
    calls = []
    monkeypatch.setattr(onepassword, 'find_op', lambda *a, **kw: Path('/synthetic/op'))
    def fetch(**kwargs):
        calls.append(kwargs['references'])
        return {name: 'synthetic-resolved' for name in kwargs['references']}, []
    monkeypatch.setattr(onepassword, 'fetch_onepassword_secrets', fetch)
    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)
    assert calls == [], 'optional integrations were eagerly resolved'
    assert get_secret('OPENAI_API_KEY') == 'synthetic-resolved'
    assert calls == [{'OPENAI_API_KEY': 'op://fixture/provider/key'}]
    assert os.environ['AGENTMAIL_API_KEY'] == 'stale-process-key'
    assert get_secret('AGENTMAIL_API_KEY') == 'synthetic-resolved'
    assert calls[-1] == {'AGENTMAIL_API_KEY': 'op://fixture/mail/key'}


def test_tool_discovery_defers_vault_until_real_dispatch(tmp_path, monkeypatch):
    from hermes_cli import env_loader
    from agent.secret_sources import onepassword
    from agent.secret_scope import get_secret
    from tools.registry import ToolRegistry, _run_check_fn_uncached
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('''secrets:
  lazy: true
  onepassword:
    enabled: true
    env:
      AGENTMAIL_API_KEY: op://fixture/mail/key
''')
    calls = []
    monkeypatch.setattr(onepassword, 'find_op', lambda *a, **kw: Path('/synthetic/op'))
    def fetch(**kwargs):
        calls.append(kwargs['references'])
        return {'AGENTMAIL_API_KEY': 'synthetic'}, []
    monkeypatch.setattr(onepassword, 'fetch_onepassword_secrets', fetch)
    probe = lambda: bool(get_secret('AGENTMAIL_API_KEY'))
    assert _run_check_fn_uncached(probe)
    assert calls == [], 'schema discovery must not read optional credentials'
    tools = ToolRegistry()
    tools.register(name='fixture_mail', toolset='fixture',
                   schema={'name': 'fixture_mail', 'parameters': {'type': 'object'}},
                   handler=lambda args: 'used' if get_secret('AGENTMAIL_API_KEY') else 'missing')
    assert tools.dispatch('fixture_mail', {}) == 'used'
    assert len(calls) == 1


def test_local_command_metadata_remains_eager(tmp_path, monkeypatch):
    from hermes_cli import env_loader
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('GXTD900_LOCAL_URL', raising=False)
    (tmp_path / 'config.yaml').write_text('''secrets:
  lazy: true
  command:
    enabled: true
    command: printf 'GXTD900_LOCAL_URL=http://127.0.0.1\\n'
    override_existing: true
''')
    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)
    assert os.environ.get('GXTD900_LOCAL_URL') == 'http://127.0.0.1'


def test_required_surface_secret_failure_never_uses_stale_env(tmp_path, monkeypatch):
    from hermes_cli import env_loader
    from agent.secret_sources import onepassword
    from agent.lazy_secrets import SecretUnavailable
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('API_SERVER_KEY', 'stale-not-authorized')
    monkeypatch.setattr(sys, 'argv', ['hermes', 'gateway', 'run'])
    (tmp_path / 'config.yaml').write_text('''secrets:
  lazy: true
  startup_names:
    gateway: [API_SERVER_KEY]
  onepassword:
    enabled: true
    override_existing: true
    env:
      API_SERVER_KEY: op://fixture/api/key
      AGENTMAIL_API_KEY: op://fixture/mail/key
''')
    calls = []
    monkeypatch.setattr(onepassword, 'find_op', lambda *a, **kw: Path('/synthetic/op'))
    def fetch(**kwargs):
        calls.append(kwargs['references'])
        return {}, ['synthetic auth failure']
    monkeypatch.setattr(onepassword, 'fetch_onepassword_secrets', fetch)
    env_loader.reset_secret_source_cache()
    with pytest.raises(SecretUnavailable, match='no credential fallback'):
        env_loader._apply_external_secret_sources(tmp_path)
    assert calls == [{'API_SERVER_KEY': 'op://fixture/api/key'}]


def test_invalid_required_classification_refuses(tmp_path, monkeypatch):
    from hermes_cli import env_loader
    from agent.lazy_secrets import SecretUnavailable
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('secrets:\n  lazy: true\n  startup_names: invalid\n')
    env_loader.reset_secret_source_cache()
    with pytest.raises(SecretUnavailable):
        env_loader._apply_external_secret_sources(tmp_path)
