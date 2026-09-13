"""An explicitly empty compression chain is a routing boundary, not auto fallback."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import yaml

from agent import auxiliary_client as ac


@pytest.fixture
def compression_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    def write(chain):
        cfg = {"auxiliary": {"compression": {"fallback_chain": chain}}}
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    write([])
    return write


@pytest.mark.parametrize("provider", ["custom:gpu0", "auto"])
def test_provider_recovery_cannot_escape_empty_chain(compression_config, monkeypatch, provider):
    route = SimpleNamespace(task="compression", tag="", resolved_provider=provider,
                            final_model="compressor", route_info={}, client=None)
    candidates = []
    for name in ("_try_configured_fallback_chain", "_try_main_fallback_chain",
                 "_try_payment_fallback", "_try_main_agent_model_fallback"):
        candidate = Mock(return_value=(None, None, ""))
        monkeypatch.setattr(ac, name, candidate)
        candidates.append(candidate)
    monkeypatch.setattr(ac, "_FALLBACK_REASONS", [(lambda error: True, "capacity")])
    assert list(ac._ladder_provider_fallback(RuntimeError("capacity"), cast(Any, route))) == []
    for candidate in candidates:
        candidate.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
def test_missing_client_cannot_auto_discover(compression_config, monkeypatch, async_mode):
    cached = Mock(return_value=(None, None))
    monkeypatch.setattr(ac, "_get_cached_client", cached)
    with pytest.raises(RuntimeError):
        ac._resolve_call_client(
            "compression", provider="custom", model="compressor", base_url=None,
            api_key=None, resolved_provider="custom", resolved_model="compressor",
            resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None,
            main_runtime={}, async_mode=async_mode,
        )
    assert cached.call_count == 1


@pytest.mark.parametrize("chain", [None, [{"provider": "custom:approved", "model": "backup"}]])
def test_missing_or_nonempty_chain_keeps_existing_recovery(compression_config, monkeypatch, chain):
    compression_config(chain)
    route = SimpleNamespace(task="compression", tag="", resolved_provider="custom:gpu0",
                            final_model="compressor", route_info={}, client=None)
    for name in ("_try_configured_fallback_chain", "_try_main_fallback_chain", "_try_payment_fallback"):
        monkeypatch.setattr(ac, name, Mock(return_value=(None, None, "")))
    main = Mock(return_value=(None, None, ""))
    monkeypatch.setattr(ac, "_try_main_agent_model_fallback", main)
    monkeypatch.setattr(ac, "_FALLBACK_REASONS", [(lambda error: True, "capacity")])
    assert list(ac._ladder_provider_fallback(RuntimeError("capacity"), cast(Any, route))) == []
    main.assert_called_once()


@pytest.mark.parametrize("async_mode", [False, True])
def test_real_http_error_never_contacts_main_model(tmp_path, monkeypatch, async_mode):
    """Real config + provider resolution + HTTP transport, with only retry delay suppressed."""
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload["model"])
            if payload["model"] == "compressor":
                self.send_response(429)
                body = {"error": {"message": "capacity exceeded", "type": "rate_limit_error"}}
            else:
                self.send_response(200)
                body = {"id": "test", "object": "chat.completion", "created": 0,
                        "model": "main-not-allowed", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "wrong route"},
                        "finish_reason": "stop"}]}
            data = json.dumps(body).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {"model": {"provider": "custom", "default": "main-not-allowed",
                     "base_url": base, "api_key": "local-test"},
           "auxiliary": {"compression": {"provider": "custom", "model": "compressor",
                         "base_url": base, "api_key": "local-test", "fallback_chain": []}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(ac, "_transient_retry_count", lambda: 0)
    try:
        kwargs: dict[str, Any] = dict(task="compression", messages=[{"role": "user", "content": "summarize"}],
                      provider="custom", model="compressor", base_url=base,
                      api_key="local-test", timeout=3, max_tokens=16)
        with pytest.raises(Exception, match="capacity|429"):
            if async_mode:
                asyncio.run(ac.async_call_llm(**kwargs))
            else:
                ac.call_llm(**kwargs)
        assert seen and set(seen) == {"compressor"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
