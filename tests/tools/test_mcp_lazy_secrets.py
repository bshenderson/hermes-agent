"""Lazy MCP uses native schema-cache/reconnect owners, no eager credentials."""


def test_lazy_config_retains_secret_reference_without_lookup(tmp_path, monkeypatch):
    from tools.mcp_tool_config import _load_mcp_config
    from agent.secret_sources import onepassword
    from pathlib import Path
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('AGENTMAIL_API_KEY', 'stale-process-not-authorized')
    (tmp_path / 'config.yaml').write_text('''secrets:
  lazy: true
  onepassword:
    enabled: true
    env:
      AGENTMAIL_API_KEY: op://fixture/mail/key
mcp_servers:
  fixture:
    url: https://example.invalid/mcp
    lazy: true
    headers:
      Authorization: ${env:AGENTMAIL_API_KEY}
''')
    calls = []
    def lookup(**kwargs):
        calls.append(kwargs['references'])
        return {'AGENTMAIL_API_KEY': 'synthetic-key'}, []
    monkeypatch.setattr(onepassword, 'find_op', lambda *a, **kw: Path('/synthetic/op'))
    monkeypatch.setattr(onepassword, 'fetch_onepassword_secrets', lookup)
    config = _load_mcp_config()
    assert config['fixture']['headers']['Authorization'] == '${env:AGENTMAIL_API_KEY}'
    assert calls == []


def test_cold_defer_does_not_fall_back_to_eager_connect(tmp_path, monkeypatch):
    from tools import mcp_tool_discovery as discovery
    from tools import mcp_schema_cache
    from tools import mcp_tool as core
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(mcp_schema_cache, 'get_cached_entry', lambda *a: None)
    # Per-fixture mutable runtime state; no live server transport is started.
    for name in ('_lazy_server_configs', '_server_connect_errors'):
        monkeypatch.setattr(core, name, {})
    monkeypatch.setattr(core, '_server_connecting', set())
    config = {'url': 'https://example.invalid/mcp', 'lazy': True, 'lazy_cold': 'defer'}
    eager, tools, servers = discovery._register_lazy_from_cache({'fixture': config})
    assert eager == {}, 'cold optional MCP fell back to eager network initialization'
    assert tools == 0
    assert servers == 1
    assert list(core._lazy_server_configs.values()) == [config]
