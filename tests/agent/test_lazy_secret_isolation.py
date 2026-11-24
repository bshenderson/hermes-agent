"""Failure closure and profile custody for deferred configured credentials."""
from pathlib import Path
import pytest


@pytest.mark.parametrize('name', ['OPENAI_API_KEY', 'AGENTMAIL_API_KEY'])
def test_configured_failure_rejects_process_and_dotenv_fallback(tmp_path, monkeypatch, name):
    from agent.secret_scope import get_secret
    from agent.secret_sources import onepassword
    from agent.lazy_secrets import SecretUnavailable
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv(name, 'stale-process')
    (tmp_path / '.env').write_text(f'{name}=stale-file\n')
    (tmp_path / 'config.yaml').write_text(f'''secrets:
  lazy: true
  onepassword:
    enabled: true
    override_existing: true
    env:
      {name}: op://fixture/item/key
''')
    monkeypatch.setattr(onepassword, 'find_op', lambda *a, **kw: Path('/synthetic/op'))
    monkeypatch.setattr(onepassword, 'fetch_onepassword_secrets', lambda **kw: ({}, ['synthetic auth refusal']))
    with pytest.raises(SecretUnavailable, match='no credential fallback'):
        get_secret(name, 'also-not-a-fallback')


def test_scoped_home_wins_over_launch_identity(tmp_path, monkeypatch):
    from agent.secret_scope import get_secret, set_secret_scope, reset_secret_scope
    from agent.secret_sources import onepassword
    launch, foreign = tmp_path / 'launch', tmp_path / 'foreign'
    for home in (launch, foreign):
        home.mkdir()
        (home / 'config.yaml').write_text(f'''secrets:
  lazy: true
  onepassword:
    enabled: true
    env:
      AGENTMAIL_API_KEY: op://fixture/{home.name}/key
''')
    monkeypatch.setenv('HERMES_HOME', str(launch))
    monkeypatch.setenv('AGENTMAIL_API_KEY', 'launch-residue')
    calls = []
    monkeypatch.setattr(onepassword, 'find_op', lambda *a, **kw: Path('/synthetic/op'))
    def fetch(**kwargs):
        calls.append(kwargs['references'])
        return {'AGENTMAIL_API_KEY': kwargs['references']['AGENTMAIL_API_KEY']}, []
    monkeypatch.setattr(onepassword, 'fetch_onepassword_secrets', fetch)
    token = set_secret_scope({}, profile_home=str(foreign))
    try:
        assert get_secret('AGENTMAIL_API_KEY') == 'op://fixture/foreign/key'
    finally:
        reset_secret_scope(token)
    assert calls == [{'AGENTMAIL_API_KEY': 'op://fixture/foreign/key'}]
    assert get_secret('AGENTMAIL_API_KEY') == 'op://fixture/launch/key'


@pytest.mark.parametrize('token_name', ['OP_SERVICE_ACCOUNT_TOKEN', 'CUSTOM_VAULT_TOKEN'])
def test_lazy_launch_auth_matches_eager_without_foreign_inheritance(tmp_path, monkeypatch, token_name):
    from agent.secret_sources import onepassword
    from agent.secret_sources.base import get_source_environment
    from hermes_cli import env_loader
    launch, foreign = tmp_path / 'launch', tmp_path / 'foreign'
    names = [token_name, 'OP_ACCOUNT', 'OP_CONNECT_HOST', 'OP_CONNECT_TOKEN',
             'OP_SESSION_fixture', 'OP_CONFIG_DIR', 'OP_LOAD_DESKTOP_APP_SETTINGS']
    for name in names:
        monkeypatch.setenv(name, 'synthetic-process-' + name)
    monkeypatch.setenv('UNRELATED_PRIVATE_KEY', 'never-copy')
    for home in (launch, foreign):
        home.mkdir()
        (home / '.op.env').write_text(''.join(f'{name}=synthetic-file-{name}\n' for name in names))
        (home / 'config.yaml').write_text(f'''secrets:
  lazy: true
  onepassword:
    enabled: true
    service_account_token_env: {token_name}
    env:
      AGENTMAIL_API_KEY: op://fixture/mail/key
''')
    monkeypatch.setenv('HERMES_HOME', str(launch))
    monkeypatch.setattr(env_loader, '_process_hermes_home', lambda: launch)
    monkeypatch.setattr(onepassword, 'find_op', lambda *a, **kw: Path('/synthetic/op'))
    seen = []
    def fetch(**kwargs):
        seen.append(dict(get_source_environment()))
        return {'AGENTMAIL_API_KEY': 'synthetic-resolved'}, []
    monkeypatch.setattr(onepassword, 'fetch_onepassword_secrets', fetch)
    for home in (launch, foreign):
        assert env_loader.resolve_profile_secret_names(home, ['AGENTMAIL_API_KEY'], mapped_only=True)
    assert {n: seen[0].get(n) for n in names} == {n: 'synthetic-process-' + n for n in names}
    assert {n: seen[1].get(n) for n in names} == {n: 'synthetic-file-' + n for n in names}
    assert all('UNRELATED_PRIVATE_KEY' not in env for env in seen)


@pytest.mark.parametrize('token_name', ['OP_SERVICE_ACCOUNT_TOKEN', 'CUSTOM_VAULT_TOKEN'])
@pytest.mark.parametrize('process_default', [True, False])
def test_actual_eager_and_lazy_helper_auth_match(tmp_path, monkeypatch, token_name, process_default):
    from agent.secret_sources import onepassword
    from agent.secret_sources.base import get_source_environment
    from hermes_cli import env_loader
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(env_loader, '_process_hermes_home', lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    for name in ['OP_SERVICE_ACCOUNT_TOKEN', 'CUSTOM_VAULT_TOKEN', 'OP_CONNECT_HOST',
                 'OP_CONNECT_TOKEN', 'OP_ACCOUNT', 'OP_SESSION_fileonly', 'OP_CONFIG_DIR']:
        monkeypatch.delenv(name, raising=False)
    if process_default:
        monkeypatch.setenv('OP_SERVICE_ACCOUNT_TOKEN', 'synthetic-selected-token')
    if token_name != 'OP_SERVICE_ACCOUNT_TOKEN':
        monkeypatch.setenv(token_name, 'synthetic-selected-custom')
    (tmp_path / '.op.env').write_text('''OP_SERVICE_ACCOUNT_TOKEN=synthetic-file-token
OP_CONNECT_HOST=https://example.invalid
OP_CONNECT_TOKEN=synthetic-file-connect
OP_ACCOUNT=synthetic-file-account
OP_SESSION_fileonly=synthetic-file-session
OP_CONFIG_DIR=/synthetic/file-only
''')
    config = f'''secrets:
  lazy: false
  onepassword:
    enabled: true
    service_account_token_env: {token_name}
    env:
      AGENTMAIL_API_KEY: op://fixture/mail/key
'''
    (tmp_path / 'config.yaml').write_text(config)
    monkeypatch.setattr(onepassword, 'find_op', lambda *a, **kw: Path('/synthetic/op'))
    seen = []
    def fetch(**kwargs):
        env = get_source_environment()
        name = kwargs['token_env']
        seen.append((onepassword._auth_fingerprint(name), onepassword._op_child_env(env.get(name, ''))))
        return {'AGENTMAIL_API_KEY': 'synthetic-result'}, []
    monkeypatch.setattr(onepassword, 'fetch_onepassword_secrets', fetch)
    env_loader.reset_secret_source_cache()
    env_loader.load_hermes_dotenv(hermes_home=tmp_path)
    assert len(seen) == 1
    (tmp_path / 'config.yaml').write_text(config.replace('lazy: false', 'lazy: true'))
    env_loader.resolve_profile_secret_names(tmp_path, ['AGENTMAIL_API_KEY'], mapped_only=True)
    assert len(seen) == 2
    assert seen[0] == seen[1], 'lazy mode changed actual helper auth inputs or their absence'
    assert ('OP_CONNECT_TOKEN' in seen[0][1]) is (not process_default)
