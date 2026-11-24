"""GXTD-900: general config reads must not hydrate deferred MCP credentials."""
from pathlib import Path


def test_repeated_config_read_leaves_lazy_mcp_secret_unresolved(tmp_path, monkeypatch):
    from hermes_cli import config
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from agent.secret_sources import onepassword
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('''secrets:
  lazy: true
  onepassword:
    enabled: true
    env:
      AGENTMAIL_API_KEY: op://fixture/mail/key
mcp_servers:
  agentmail:
    lazy: true
    command: fixture
    env:
      AGENTMAIL_API_KEY: ${AGENTMAIL_API_KEY}
''')
    calls = []
    monkeypatch.setattr(onepassword, 'find_op', lambda *a, **kw: Path('/synthetic/op'))
    def fetch(**kwargs):
        calls.append(kwargs['references'])
        return {}, ['synthetic offline refusal']
    monkeypatch.setattr(onepassword, 'fetch_onepassword_secrets', fetch)
    token = set_secret_scope({}, profile_home=str(tmp_path))
    try:
        for _ in range(2):
            loaded = config.load_config()
            assert loaded['mcp_servers']['agentmail']['env']['AGENTMAIL_API_KEY'] == '${AGENTMAIL_API_KEY}'
    finally:
        reset_secret_scope(token)
    assert calls == []
