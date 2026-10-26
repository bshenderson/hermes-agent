"""GXTD-758: fail-closed auxiliary calls must never escape to primary recovery."""
import asyncio
import copy
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import hermes_yaml as yaml
from agent import auxiliary_client as aux


@pytest.fixture
def peer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    state = {"status": 429}

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            if state["status"] == 0:
                self.close_connection = True
                return
            primary = self.path.startswith("/primary/")
            status = 200 if primary else state["status"]
            if status == 200:
                payload = {"id": "fixture", "object": "chat.completion", "created": 1,
                           "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                           "message": {"role": "assistant", "content": "PRIMARY" if primary else "AUX"}}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
            else:
                reason = {401: "fixture unauthorized", 409: "output_tokens_required",
                          429: "scheduler_rejected_concurrency", 503: "fixture service unavailable"}[status]
                payload = {"error": {"message": reason, "type": "fixture_failure", "code": reason}}
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    connect = socket.socket.connect
    blocked = []

    def local_only(sock, address):
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1") or address[1] != port:
            blocked.append(address)
            raise OSError("fixture forbids external network")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    monkeypatch.setenv("FIXTURE_KEY", "fixture-only")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"model": {"provider": "custom:fixture-primary", "default": "fixture-primary"},
              "providers": {
                  "fixture-aux": {"api": f"http://127.0.0.1:{port}/aux/v1", "key_env": "FIXTURE_KEY", "default_model": "fixture-aux"},
                  "fixture-primary": {"api": f"http://127.0.0.1:{port}/primary/v1", "key_env": "FIXTURE_KEY", "default_model": "fixture-primary"}},
              "auxiliary": {"transient_retries": 0, "profile_describer": {
                  "provider": "fixture-aux", "model": "fixture-aux", "recovery_policy": "fail_closed",
                  "fallback_chain": [], "extra_body": {"max_tokens": 64}}}}

    def configure(task="profile_describer", **fields):
        entry = config["auxiliary"].pop("profile_describer", None)
        if entry is not None:
            config["auxiliary"][task] = entry
        config["auxiliary"][task].update(fields)
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))

    configure()
    try:
        yield calls, state, configure, config
    finally:
        aux.shutdown_cached_clients()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert blocked == []


def invoke(mode, task="profile_describer"):
    messages = [{"role": "user", "content": "Synthetic policy input"}]
    original = copy.deepcopy(messages)
    try:
        if mode == "sync":
            return aux.call_llm(task=task, messages=messages, max_tokens=64, timeout=2)
        return asyncio.run(aux.async_call_llm(task=task, messages=messages, max_tokens=64, timeout=2))
    finally:
        assert messages == original


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("status, exception", [(401, "AuthenticationError"), (409, "ConflictError"),
                                              (429, "RateLimitError"), (503, "InternalServerError")])
@pytest.mark.parametrize("task", ["profile_describer", "compression", "vision"])
def test_fail_closed_errors_do_not_reach_primary(peer, mode, status, exception, task):
    calls, state, configure, _ = peer
    state["status"] = status
    configure(task)
    with pytest.raises(Exception) as error:
        invoke(mode, task)
    assert type(error.value).__name__ == exception, repr(error.value)
    assert len(calls) == 1
    assert all(path.startswith("/aux/") for path, _ in calls)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("task", ["profile_describer", "compression", "vision"])
def test_fail_closed_success(peer, mode, task):
    calls, state, configure, _ = peer
    state["status"] = 200
    configure(task)
    assert invoke(mode, task).choices[0].message.content == "AUX"
    assert len(calls) == 1
    assert calls[0][1]["model"] == "fixture-aux"
    assert calls[0][1]["max_tokens"] == 64


@pytest.mark.parametrize("status", [200, 401, 409, 429, 503])
def test_profile_describer_consumer_preserves_state_on_route_failure(peer, tmp_path, status):
    from hermes_cli import profile_describer, profiles

    calls, state, configure, _ = peer
    state["status"] = status
    configure()
    profiles.write_profile_meta(tmp_path, description="previous", description_auto=True)
    before = (tmp_path / "profile.yaml").read_bytes()
    outcome = profile_describer.describe_profile("default", timeout=2)
    assert outcome.ok is (status == 200), outcome.reason
    assert len(calls) == 1
    path, body = calls[0]
    assert path.startswith("/aux/")
    assert body["model"] == "fixture-aux"
    assert body["max_tokens"] == 64
    if status == 200:
        assert outcome.description == "AUX"
        assert profiles.read_profile_meta(tmp_path)["description"] == "AUX"
    else:
        assert (tmp_path / "profile.yaml").read_bytes() == before


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("policy", ["absent", None, "legacy"])
def test_legacy_empty_chain_still_recovers_to_primary(peer, mode, policy):
    calls, _, configure, config = peer
    if policy == "absent":
        config["auxiliary"]["profile_describer"].pop("recovery_policy")
        configure()
    else:
        configure(recovery_policy=policy)
    assert invoke(mode).choices[0].message.content == "PRIMARY"
    assert any(path.startswith("/primary/") for path, _ in calls)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("status", [401, 409, 429, 503])
def test_legacy_compression_empty_chain_remains_isolated(peer, mode, status):
    calls, state, configure, _ = peer
    state["status"] = status
    configure("compression", recovery_policy=None)
    with pytest.raises(Exception):
        invoke(mode, "compression")
    assert calls
    assert all(path.startswith("/aux/") for path, _ in calls)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("policy", [False, True, "", "typo", [], {}, 1])
def test_invalid_policy_refuses_before_client_resolution(peer, monkeypatch, mode, policy):
    calls, _, configure, _ = peer
    configure(recovery_policy=policy)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid policy reached provider resolution")
    monkeypatch.setattr(aux, "_resolve_task_provider_model", forbidden)
    with pytest.raises(ValueError, match="recovery_policy"):
        invoke(mode)
    assert calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("fields", [{"provider": "auto"}, {"provider": "moa"}, {"provider": "main"},
                                    {"provider": "custom:auto"}, {"provider": "custom:main"},
                                    {"provider": "custom:"}, {"provider": "custom"},
                                    {"model": "auto"}, {"model": ""}, {"model": "  "}])
def test_ambiguous_route_refused_without_client(peer, monkeypatch, mode, fields):
    calls, _, configure, _ = peer
    configure(**fields)
    def forbidden(*args, **kwargs):
        pytest.fail("ambiguous route reached client construction")
    monkeypatch.setattr(aux, "_get_cached_client", forbidden)
    with pytest.raises(ValueError, match="concrete provider"):
        invoke(mode)
    assert calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("task", ["profile_describer", "vision"])
def test_unavailable_selected_client_cannot_execute_configured_chain(peer, monkeypatch, mode, task):
    calls, _, configure, _ = peer
    configure(task, fallback_chain=[{"provider": "fixture-primary", "model": "fixture-primary"}])
    selected = []
    def unavailable(provider, *args, **kwargs):
        selected.append(provider)
        return None, None
    monkeypatch.setattr(aux, "_get_cached_client", unavailable)
    with pytest.raises(aux.AuxiliaryClientUnavailable, match="fail-closed"):
        invoke(mode, task)
    assert selected == ["fixture-aux"]
    assert calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_failed_request_ignores_configured_chain_and_late_policy_drift(peer, monkeypatch, mode):
    calls, _, configure, _ = peer
    configure(fallback_chain=[{"provider": "fixture-primary", "model": "fixture-primary"}])
    original_resolver = aux._resolve_call_client
    def resolve_then_widen(*args, **kwargs):
        result = original_resolver(*args, **kwargs)
        configure(recovery_policy="legacy")
        return result
    monkeypatch.setattr(aux, "_resolve_call_client", resolve_then_widen)
    with pytest.raises(Exception) as error:
        invoke(mode)
    assert type(error.value).__name__ == "RateLimitError"
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_transport_failure_preserves_original_error_without_recovery(peer, monkeypatch, mode):
    calls, _, _, _ = peer
    original = RuntimeError("synthetic connection boundary")
    def fail(*args, **kwargs):
        raise original
    async def async_fail(*args, **kwargs):
        raise original
    monkeypatch.setattr(aux, "_relay_sync_completion", fail)
    monkeypatch.setattr(aux, "_relay_async_completion", async_fail)
    with pytest.raises(RuntimeError) as error:
        invoke(mode)
    assert error.value is original
    assert calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("policy", ["fail_closed", ""])
def test_plugin_inheritance_carries_policy_and_invalid_empty_value(peer, monkeypatch, mode, policy):
    calls, _, configure, config = peer
    from hermes_cli import plugins
    monkeypatch.setattr(plugins, "get_plugin_auxiliary_tasks", lambda: [
        {"key": "fixture_plugin", "inherit_from": "profile_describer", "defaults": {}}])
    config["auxiliary"]["fixture_plugin"] = {"recovery_policy": policy}
    configure()
    with pytest.raises(Exception) as error:
        invoke(mode, "fixture_plugin")
    if policy == "":
        assert isinstance(error.value, ValueError)
        assert calls == []
    else:
        assert type(error.value).__name__ == "RateLimitError"
        assert len(calls) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_disconnected_http_peer_cannot_trigger_replay(peer, mode):
    calls, state, _, _ = peer
    state["status"] = 0
    with pytest.raises(Exception) as error:
        invoke(mode)
    assert type(error.value).__name__ == "APIConnectionError"
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_real_named_provider_without_endpoint_cannot_use_chain(peer, mode):
    calls, _, configure, config = peer
    config["providers"]["fixture-aux"].pop("api")
    configure(fallback_chain=[{"provider": "fixture-primary", "model": "fixture-primary"}])
    with pytest.raises(aux.AuxiliaryClientUnavailable):
        invoke(mode)
    assert calls == []
