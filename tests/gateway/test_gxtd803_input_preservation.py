"""GXTD-803 complete-input preservation across API ingress normalization.

Root cause retained by ``docs/reports/gxtd803-output-budget-8192-result-20261107.md``:
a 300372-char coordinator submission persisted as exactly the first 65536 characters,
losing the final reply instruction. The silent slice lived in
``gateway/platforms/api_server.py`` (``MAX_NORMALIZED_TEXT_LENGTH`` + ``_cap_text``),
applied by ``_normalize_multimodal_content`` / ``_normalize_chat_content`` before the
Chat Completions route forwarded input to dispatch and context selection.

Contract under test:
  * complete valid submitted input is preserved end-to-end within the existing
    authenticated request-byte bound (``MAX_REQUEST_BYTES``),
  * genuinely excessive input receives an explicit typed refusal, never a silent
    slice,
  * only the external agent dispatch is mocked; no native calls, no service state.
"""

import hashlib
from contextlib import suppress

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig


def _module_origin_guard():
    """Fail closed unless the imported product modules come from THIS checkout —
    the same source tree this test file lives in (candidate worktree or baseline
    export each exercise their own bytes)."""
    from gateway.platforms import api_server
    import gateway.platforms.api_server_openai_routes as routes
    from pathlib import Path
    own_root = str(Path(__file__).resolve().parents[2])
    for mod in (api_server, routes):
        origin = str(mod.__file__)
        assert origin.startswith(own_root + "/gateway/"), \
            f"module {mod.__name__} loaded from foreign origin {origin} (expected under {own_root})"
    return api_server


api_server = _module_origin_guard()

from gateway.platforms.api_server import (
    APIServerAdapter,
    MAX_REQUEST_BYTES,
    _normalize_chat_content,
    _normalize_multimodal_content,
    _session_chat_user_message,
    cors_middleware,
    security_headers_middleware,
)


def _make_app(adapter: APIServerAdapter, *, body_bound: bool = False) -> web.Application:
    from gateway.platforms.api_server import body_limit_middleware
    mws = [mw for mw in (cors_middleware, body_limit_middleware, security_headers_middleware)
           if mw is not None] if body_bound else [
        mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws, client_max_size=MAX_REQUEST_BYTES)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


def _capturing_run_agent():
    """MagicMock for ``_run_agent`` that records the dispatch kwargs and returns a
    minimal complete turn. Stands in for the external agent/provider only."""
    from unittest.mock import MagicMock

    async def _stub(**kwargs):
        mock_run.captured = kwargs  # type: ignore[attr-defined]
        return ({"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    mock_run = MagicMock(side_effect=_stub)
    return mock_run


async def _post_chat(cli, messages):
    return await cli.post("/v1/chat/completions",
                          json={"model": "hermes-agent", "messages": messages})


# 300372 mirrors the retained actual submission length; the tail mirrors the
# instruction that was lost from the persisted session row.
_ACTUAL_SUBMISSION_CHARS = 300372


def _long_text(total_chars: int, tail: str) -> str:
    body = hashlib.sha256(tail.encode()).hexdigest() * ((total_chars // 64) + 1)
    text = (body[: total_chars - len(tail)] + tail)
    assert len(text) == total_chars
    return text


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


class TestChatCompletionsActualIngress:

    @pytest.mark.asyncio
    async def test_actual_length_user_message_tail_reaches_dispatch(self, adapter):
        """RED tracer: the exact tail of a 300372-char user message reaches the
        dispatch/selection seam unchanged (the retained defect persisted only
        the first 65536 chars)."""
        tail = "REPLY_MARKER_EXACT_TAIL_DO_NOT_DROP"
        text = _long_text(_ACTUAL_SUBMISSION_CHARS, tail)
        mock_run = _capturing_run_agent()
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            from unittest.mock import patch
            with patch.object(adapter, "_run_agent", new=mock_run):
                resp = await _post_chat(cli, [{"role": "user", "content": text}])
        assert resp.status == 200, await _body(resp)
        captured = mock_run.captured["user_message"]
        assert captured == text, (
            f"input not preserved end-to-end: captured {len(captured)} chars, "
            f"tail present={captured.endswith(tail)}")

    @pytest.mark.asyncio
    async def test_text_parts_beyond_64k_flatten_complete(self, adapter):
        """Typed text-part content flattened to one string keeps every part's tail."""
        marker_a = "PART_A_TAIL_" + "a" * 70_000
        marker_b = "PART_B_SHORT_TAIL"
        content = [{"type": "text", "text": marker_a}, {"type": "text", "text": marker_b}]
        mock_run = _capturing_run_agent()
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            from unittest.mock import patch
            with patch.object(adapter, "_run_agent", new=mock_run):
                resp = await _post_chat(cli, [{"role": "user", "content": content}])
        assert resp.status == 200, await _body(resp)
        captured = mock_run.captured["user_message"]
        assert captured == marker_a + "\n" + marker_b

    @pytest.mark.asyncio
    async def test_system_message_beyond_64k_reaches_dispatch(self, adapter):
        tail = "SYSTEM_TAIL_" + "s" * 66_000
        mock_run = _capturing_run_agent()
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            from unittest.mock import patch
            with patch.object(adapter, "_run_agent", new=mock_run):
                resp = await _post_chat(
                    cli,
                    [{"role": "system", "content": tail},
                     {"role": "user", "content": "hi"}])
        assert resp.status == 200, await _body(resp)
        assert mock_run.captured["ephemeral_system_prompt"] == tail

    @pytest.mark.asyncio
    async def test_history_messages_beyond_64k_reach_dispatch(self, adapter):
        tail = "HISTORY_TAIL_" + "h" * 66_000
        mock_run = _capturing_run_agent()
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            from unittest.mock import patch
            with patch.object(adapter, "_run_agent", new=mock_run):
                resp = await _post_chat(
                    cli,
                    [{"role": "user", "content": tail},
                     {"role": "assistant", "content": "ack"},
                     {"role": "user", "content": "follow-up"}])
        assert resp.status == 200, await _body(resp)
        history = mock_run.captured["conversation_history"]
        assert history[0]["content"] == tail


class TestResponsesSiblingIngress:

    @pytest.mark.asyncio
    async def test_responses_long_input_item_tail_reaches_dispatch(self, adapter):
        tail = "RESPONSES_TAIL_" + "r" * 66_000
        mock_run = _capturing_run_agent()
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            from unittest.mock import patch
            with patch.object(adapter, "_run_agent", new=mock_run):
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "store": False,
                    "input": [{"role": "user", "content": tail}]})
        assert resp.status == 200, await _body(resp)
        assert mock_run.captured["user_message"] == tail


class TestExplicitOversizedRefusal:
    """Genuinely excessive input is refused with a typed error, never sliced."""

    def test_plain_string_beyond_bound_refused_not_sliced(self):
        oversized = "x" * (api_server.MAX_INPUT_TEXT_LENGTH + 1)
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content(oversized)
        assert str(exc.value).startswith("input_too_large:")

    def test_long_plain_string_within_bound_preserved(self):
        text = "x" * api_server.MAX_INPUT_TEXT_LENGTH
        assert _normalize_multimodal_content(text) == text

    def test_long_text_part_within_bound_preserved(self):
        text = "x" * api_server.MAX_INPUT_TEXT_LENGTH
        parts = [{"type": "text", "text": text}]
        assert _normalize_multimodal_content(parts) == text

    def test_long_text_part_beyond_bound_refused(self):
        oversized = "x" * (api_server.MAX_INPUT_TEXT_LENGTH + 1)
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "text", "text": oversized}])
        assert str(exc.value).startswith("input_too_large:")

    def test_content_list_beyond_documented_size_refused(self):
        parts = ["p" for _ in range(api_server.MAX_CONTENT_LIST_SIZE + 1)]
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content(parts)
        assert str(exc.value).startswith("input_too_large:")

    def test_system_normalization_beyond_bound_refused(self):
        oversized = "x" * (api_server.MAX_INPUT_TEXT_LENGTH + 1)
        with pytest.raises(ValueError) as exc:
            _normalize_chat_content(oversized)
        assert str(exc.value).startswith("input_too_large:")

    def test_native_session_ingestion_preserves_long_message(self, tmp_path, monkeypatch):
        """Sibling native session route (`/api/sessions/{id}/chat` prologue) shares the
        same normalization helper; a long message must be preserved, not sliced."""
        tail = "NATIVE_TAIL_" + "n" * 66_000
        message, err = _session_chat_user_message({"message": tail})
        assert err is None
        assert message == tail

    @pytest.mark.asyncio
    async def test_chat_route_maps_typed_refusal_to_400(self, adapter):
        """A single text part beyond the bound cannot exceed the request-byte bound
        through the wire, so exercise the route mapping directly through the
        normalizer seam the route calls, preserving the route's error contract."""
        from gateway.platforms.api_server import _multimodal_validation_error
        oversized = "x" * (api_server.MAX_INPUT_TEXT_LENGTH + 1)
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content(oversized)
        response = _multimodal_validation_error(exc.value, param="messages[0].content")
        assert response.status == 400
        import json
        payload = json.loads(response.body)
        assert payload["error"]["code"] == "input_too_large"
        assert "messages[0].content" in payload["error"]["param"]

    @pytest.mark.asyncio
    async def test_chat_route_oversized_part_list_is_typed_400(self, adapter):
        """Wire-level: a content-part array beyond the documented list bound is a
        request-achievable oversized input; it earns the typed 400 and never dispatches.
        (A single text field beyond MAX_INPUT_TEXT_LENGTH can only arrive as a body beyond
        the request-byte bound, so the 413 case below covers that wire leg.)"""
        parts = [{"type": "text", "text": f"p{i}"} for i in range(api_server.MAX_CONTENT_LIST_SIZE + 1)]
        mock_run = _capturing_run_agent()
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            from unittest.mock import patch
            with patch.object(adapter, "_run_agent", new=mock_run):
                resp = await _post_chat(cli, [{"role": "user", "content": parts}])
                assert resp.status == 400, await _body(resp)
                payload = await resp.json()
        assert payload["error"]["code"] == "input_too_large"
        assert payload["error"]["param"] == "messages[0].content"
        assert mock_run.call_count == 0

    @pytest.mark.asyncio
    async def test_chat_route_oversized_system_part_list_is_typed_400(self, adapter):
        """System content shares the normalizer; its refusal must map to the same typed
        400 rather than escaping as an unhandled exception."""
        parts = ["s" for _ in range(api_server.MAX_CONTENT_LIST_SIZE + 1)]
        mock_run = _capturing_run_agent()
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            from unittest.mock import patch
            with patch.object(adapter, "_run_agent", new=mock_run):
                resp = await _post_chat(
                    cli, [{"role": "system", "content": parts},
                          {"role": "user", "content": "hi"}])
                assert resp.status == 400, await _body(resp)
                payload = await resp.json()
        assert payload["error"]["code"] == "input_too_large"
        assert mock_run.call_count == 0

    @pytest.mark.asyncio
    async def test_responses_route_oversized_part_list_is_typed_400(self, adapter):
        """Sibling Responses route shares the normalizer and maps the same typed 400."""
        parts = [{"type": "text", "text": f"p{i}"} for i in range(api_server.MAX_CONTENT_LIST_SIZE + 1)]
        mock_run = _capturing_run_agent()
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            from unittest.mock import patch
            with patch.object(adapter, "_run_agent", new=mock_run):
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "store": False,
                    "input": [{"role": "user", "content": parts}]})
                assert resp.status == 400, await _body(resp)
                payload = await resp.json()
        assert payload["error"]["code"] == "input_too_large"
        assert mock_run.call_count == 0

    @pytest.mark.asyncio
    async def test_chat_route_body_over_request_bound_is_typed_413(self, adapter):
        """The authenticated request-byte bound stays authoritative: a declared body above
        MAX_REQUEST_BYTES earns the existing typed 413 before the body is even read. Raw
        socket: an ordinary client that keeps writing while the server answers early can
        see the early response close the connection, which is transport behavior, not the
        product contract under test."""
        import asyncio
        import json as _json
        server = TestServer(_make_app(adapter, body_bound=True))
        await server.start_server()
        try:
            reader, writer = await asyncio.open_connection(server.host, server.port)
            declared = MAX_REQUEST_BYTES + 10_000
            head = (b"POST /v1/chat/completions HTTP/1.1\r\n"
                    b"Host: localhost\r\nContent-Type: application/json\r\n"
                    b"Content-Length: " + str(declared).encode() + b"\r\n\r\n")
            writer.write(head + b"{")
            await writer.drain()
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=15)
            status_line = raw.split(b"\r\n", 1)[0]
            assert b"413" in status_line, status_line
            # body_too_large is the documented code; the JSON body follows the headers
            body = await asyncio.wait_for(reader.read(4096), timeout=15)
            payload = _json.loads(body)
            assert payload["error"]["code"] == "body_too_large"
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
        finally:
            await server.close()


async def _body(resp):
    return await resp.text()
