"""Opt-in native backend pins. This is not a tool or an egress sandbox."""
from __future__ import annotations

import json
from urllib.parse import urlsplit


def _failure(capability, reason):
    return {"success": False, "code": "web_backend_unavailable", "reason": reason,
            "capability": capability,
            "error": "The approved web backend is unavailable. Stop this operation; do not use an alternate backend."}


def backend_policy_error(capability, config, env_value):
    """Compare effective profile endpoints before plugin selection or cache lookup.

    No process environment writes, network calls, credential values in diagnostics,
    or changed behavior for profiles without the explicit policy field.
    """
    web = config.get("web") or {}
    if "required_endpoints" not in web:
        return None
    pins = web["required_endpoints"]
    if not isinstance(pins, dict) or capability not in ("search", "extract"):
        return _failure(capability, "policy_missing")
    if any(web.get(key) is not False for key in (
        "use_gateway", "keyless_rescue", "keyless_fallback", "cache_enabled",
    )):
        # Existing search memo keys omit profile and endpoint; do not trust it in a pinned lane.
        return _failure(capability, "fallback_or_cache_enabled")
    backend, env_key = {"search": ("searxng", "SEARXNG_URL"),
                        "extract": ("firecrawl", "FIRECRAWL_API_URL")}[capability]
    if web.get(capability + "_backend") != backend:
        return _failure(capability, "provider_mismatch")
    plugins = config.get("plugins") or {}
    aliases = {"web/" + backend, "web-" + backend}
    if aliases.intersection(plugins.get("disabled") or []):
        return _failure(capability, "plugin_disabled")
    if capability == "search" and not aliases.intersection(plugins.get("enabled") or []):
        return _failure(capability, "plugin_not_enabled")
    expected = pins.get(capability)
    if not isinstance(expected, str) or not expected:
        return _failure(capability, "policy_missing")
    parsed = urlsplit(expected)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        return _failure(capability, "invalid_endpoint_pin")
    actual = env_value(env_key)
    if not isinstance(actual, str) or actual.rstrip("/") != expected.rstrip("/"):
        return _failure(capability, "endpoint_mismatch")
    return None


def native_backend_guard(capability):
    """Return a typed JSON refusal, or None. Config/secret-loader errors fail closed."""
    try:
        from hermes_cli.config import load_config_readonly, get_env_value
        error = backend_policy_error(capability, load_config_readonly(), get_env_value)
    except Exception:
        error = _failure(capability, "policy_resolution_failed")
    return json.dumps(error) if error else None
