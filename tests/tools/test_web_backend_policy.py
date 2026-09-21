"""GXTD-603: pin effective endpoints before native provider/cache access."""
import copy

import pytest


def config():
    return {
        "web": {"search_backend": "searxng", "extract_backend": "firecrawl",
                "use_gateway": False, "keyless_rescue": False, "keyless_fallback": False,
                "cache_enabled": False,
                "required_endpoints": {"search": "http://search.test/searxng", "extract": "https://extract.test:16443"}},
        "plugins": {"enabled": ["web/searxng"], "disabled": []},
    }


def test_missing_effective_extract_url_never_unlocks_cloud():
    from tools.web_backend_policy import backend_policy_error
    error = backend_policy_error("extract", config(), lambda name: None)
    assert error["code"] == "web_backend_unavailable"
    assert error["reason"] == "endpoint_mismatch"


def test_matching_effective_endpoints_allowed_without_mutating_config():
    from tools.web_backend_policy import backend_policy_error
    cfg = config()
    original = copy.deepcopy(cfg)
    env = {"SEARXNG_URL": "http://search.test/searxng", "FIRECRAWL_API_URL": "https://extract.test:16443"}
    for capability in ("search", "extract"):
        assert backend_policy_error(capability, cfg, env.get) is None
    assert cfg == original


def test_unconfigured_profiles_keep_ordinary_native_behavior():
    from tools.web_backend_policy import backend_policy_error
    assert backend_policy_error("extract", {"web": {"backend": "firecrawl"}}, lambda _: None) is None


@pytest.mark.parametrize("disabled", ["web-searxng", "web/searxng"])
def test_selected_but_disabled_search_refused(disabled):
    from tools.web_backend_policy import backend_policy_error
    cfg = config()
    cfg["plugins"]["disabled"] = [disabled]
    assert backend_policy_error("search", cfg, lambda _: "http://search.test/searxng")["reason"] == "plugin_disabled"


@pytest.mark.parametrize("field,value", [("keyless_rescue", True), ("keyless_fallback", None), ("cache_enabled", True), ("use_gateway", True), ("extract_backend", "nous")])
def test_native_fallback_and_unscoped_cache_fail_closed(field, value):
    from tools.web_backend_policy import backend_policy_error
    cfg = config()
    cfg["web"][field] = value
    assert backend_policy_error("extract", cfg, lambda _: "https://extract.test:16443") is not None


def test_native_tools_refuse_before_any_provider_or_cache(monkeypatch):
    import asyncio
    from tools import web_tools
    from hermes_cli import config as config_module
    import json
    monkeypatch.setattr(config_module, "load_config_readonly", config)
    monkeypatch.setattr(config_module, "get_env_value", lambda _: None)
    calls = []
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: calls.append("provider"))
    search = json.loads(web_tools.web_search_tool("fixture"))
    extract = json.loads(asyncio.run(web_tools.web_extract_tool(["https://example.com"])))
    assert search.get("code") == "web_backend_unavailable"
    assert extract.get("code") == "web_backend_unavailable"
    assert calls == []


def test_pinned_firecrawl_client_is_request_local(monkeypatch):
    from plugins.web.firecrawl import provider
    from hermes_cli import config as config_module
    from tools import web_tools
    from types import SimpleNamespace
    monkeypatch.setattr(config_module, "load_config_readonly", config)
    monkeypatch.setattr(config_module, "get_env_value", lambda key: "https://extract.test:16443" if key == "FIRECRAWL_API_URL" else None)
    monkeypatch.setattr(provider, "_get_direct_firecrawl_config", lambda: ("sdk", {"api_url": "https://extract.test:16443"}, ("direct", "https://extract.test:16443", None)))
    monkeypatch.setattr(provider, "Firecrawl", lambda **kw: SimpleNamespace(**kw))
    cached = object()
    monkeypatch.setattr(web_tools, "_firecrawl_client", cached)
    monkeypatch.setattr(web_tools, "_firecrawl_client_config", ("direct", "https://extract.test:16443", None))
    result = provider._get_firecrawl_client()
    assert result is not cached
    assert result.api_url == "https://extract.test:16443"
    assert web_tools._firecrawl_client is cached
