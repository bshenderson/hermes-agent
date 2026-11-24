"""GXTD-900: provider presence must not construct a deferred credential."""
from pathlib import Path
import pytest


def test_firecrawl_presence_defers_key_but_dispatch_still_refuses(tmp_path, monkeypatch):
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from agent.secret_sources import onepassword
    from agent.lazy_secrets import SecretUnavailable
    from plugins.web.firecrawl import provider
    from tools.web_tools import check_web_api_key
    from tools.registry import _run_check_fn_uncached
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('''web:
  backend: firecrawl
secrets:
  lazy: true
  onepassword:
    enabled: true
    env:
      FIRECRAWL_API_KEY: op://fixture/firecrawl/key
''')
    calls = []
    monkeypatch.setattr(onepassword, 'find_op', lambda *a, **kw: Path('/synthetic/op'))
    def fetch(**kwargs):
        calls.append(kwargs['references'])
        return {}, ['synthetic offline refusal']
    monkeypatch.setattr(onepassword, 'fetch_onepassword_secrets', fetch)
    token = set_secret_scope({}, profile_home=str(tmp_path))
    try:
        assert _run_check_fn_uncached(provider.check_firecrawl_api_key)
        assert _run_check_fn_uncached(check_web_api_key)
        assert calls == []
        with pytest.raises(SecretUnavailable, match='no credential fallback'):
            provider._get_direct_firecrawl_config()
    finally:
        reset_secret_scope(token)
    assert calls == [{'FIRECRAWL_API_KEY': 'op://fixture/firecrawl/key'}]
