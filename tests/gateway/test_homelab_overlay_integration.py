"""GXTD-647: exercise carried policies through real HTTP and executor paths."""
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import chat_location
from gateway.config import PlatformConfig
from gateway.platforms import api_server, api_server_runtime_metrics


@pytest.mark.asyncio
async def test_unknown_location_short_circuits_before_agent_creation(monkeypatch):
    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True))
    create = Mock(side_effect=AssertionError("must not create an agent"))
    monkeypatch.setattr(adapter, "_create_agent", create)
    chunks = []
    location = chat_location.TurnLocation("synthetic-session", {"status": "unknown"}, 0)
    try:
        result, usage = await adapter._run_agent(
            "What is the weather here today?", [], chat_location_turn=location,
            stream_delta_callback=chunks.append)
        assert chunks == [result["final_response"]]
        assert "city" in result["final_response"]
        assert usage["total_tokens"] == 0
        assert location._location == {}
        create.assert_not_called()
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_signed_location_and_research_survive_http_route_merge(monkeypatch):
    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "synthetic-execution-key"}))
    run = AsyncMock(return_value=({"final_response": "bounded answer", "completed": True},
                                 {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}))
    monkeypatch.setattr(adapter, "_run_agent", run)
    monkeypatch.setattr(adapter, "_select_request_route", lambda *a, **kw: (None, {}, None))
    monkeypatch.setattr(adapter, "_declared_conversation_session", lambda *a: None)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"web": {"preferred_research_mode": True}})

    @web.middleware
    async def profile_scope(request, handler):
        from agent.secret_scope import set_secret_scope, reset_secret_scope

        token = api_server._api_request_profile.set("homelab-coordinator")
        secrets = set_secret_scope({"API_SERVER_KEY": "synthetic-execution-key"})
        try:
            return await handler(request)
        finally:
            reset_secret_scope(secrets)
            api_server._api_request_profile.reset(token)

    app = web.Application(middlewares=[profile_scope])
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    headers = {"Authorization": "Bearer synthetic-execution-key"}
    metadata = {"chat_id": "11111111-1111-4111-8111-111111111111",
                "message_id": "22222222-2222-4222-8222-222222222222",
                "user_message_id": "33333333-3333-4333-8333-333333333333",
                "variables": {"{{USER_LOCATION}}": "42.360, -71.058 (lat, long)"}}
    try:
        async with TestClient(TestServer(app)) as client:
            body = {"messages": [{"role": "user", "content": "weather here?"}]}
            context = chat_location.native_context(user_id="synthetic-user", metadata=metadata)
            signed = chat_location.sign_context(context, body=body, user_id="synthetic-user",
                                               profile="homelab-coordinator", key="synthetic-execution-key")
            body["hermes_chat_context"] = signed
            response = await client.post("/v1/chat/completions", json=body, headers=headers)
            assert response.status == 200, await response.text()
            location = run.call_args.kwargs["chat_location_turn"]
            assert "42.36" in location.prompt_context()
            assert run.call_args.kwargs["bind_declared_conversation"] is True
            assert run.call_args.kwargs["session_history_delivery"] == ""

            run.reset_mock()
            body["hermes_chat_context"] = {**signed, "subject_id": "tampered"}
            response = await client.post("/v1/chat/completions", json=body, headers=headers)
            assert response.status == 400
            run.assert_not_called()

            body = {"messages": [{"role": "user", "content": "/research compare https://example.com/a"}]}
            context = chat_location.native_context(user_id="synthetic-user", metadata=metadata)
            body["hermes_chat_context"] = chat_location.sign_context(
                context, body=body, user_id="synthetic-user", profile="homelab-coordinator",
                key="synthetic-execution-key")
            response = await client.post("/v1/chat/completions", json=body, headers=headers)
            assert response.status == 200, await response.text()
            assert run.call_args.kwargs["preferred_web_policy"].allowed == {"https://example.com/a": 0}
            assert "42.36" not in run.call_args.kwargs["chat_location_turn"].prompt_context()
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_private_metrics_auth_and_read_only_database(tmp_path, monkeypatch):
    from hermes_cli import kanban_db

    path = tmp_path / "board.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE tasks (id INTEGER PRIMARY KEY, status TEXT, created_at REAL);
            CREATE TABLE task_runs (id INTEGER PRIMARY KEY, task_id INTEGER, outcome TEXT, ended_at REAL);
            CREATE TABLE task_events (task_id INTEGER, run_id INTEGER, kind TEXT);
            INSERT INTO tasks VALUES (1, 'ready', 1000);
        """)
    before = path.read_bytes()
    monkeypatch.setattr(kanban_db, "kanban_db_path", lambda board: path)
    adapter = SimpleNamespace(config=SimpleNamespace(extra={"runtime_metrics": {
        "enabled": True, "boards": ["homelab"], "api_lifecycle": True}}), _api_key="execution-key")
    app = web.Application()
    for method, route, handler in api_server_runtime_metrics.http_routes(adapter, token="metrics-key"):
        app.router.add_route(method, route, handler)
        app.router.add_route(method, "/p/private" + route, handler)
    async with TestClient(TestServer(app)) as client:
        assert (await client.get("/metrics/runtime")).status == 401
        assert (await client.get("/metrics/runtime", headers={"Authorization": "Bearer execution-key"})).status == 401
        headers = {"Authorization": "Bearer metrics-key"}
        assert (await client.get("/p/private/metrics/runtime", headers=headers)).status == 404
        response = await client.get("/metrics/runtime", headers=headers)
        assert response.status == 200
        text = await response.text()
        assert 'hermes_kanban_board_tasks_retained{board="homelab",state="ready"} 1' in text
        assert 'hermes_api_executors_active{family="request"} 0' in text
        assert "execution-key" not in text and "metrics-key" not in text
    assert path.read_bytes() == before
