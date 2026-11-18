"""GXTD-850: retain requested versus effective route identity, never credentials."""

import json
from types import SimpleNamespace
import pytest

from agent.delegation_registry import get_active_provider
from tests.tools.test_delegation_route_fields import LOCAL, _construct, _spawn
from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry, _fabricated_entry
from tools.delegation_live_log import _manifest_path, create_live_transcripts, update_manifest_statuses


def test_default_route_manifest_survives_fallback_result_and_persistence(tmp_path):
    def construct(**kwargs):
        child = _construct(**kwargs)
        child.context_compressor = SimpleNamespace(context_length=65536)
        return child
    _, _, (child,) = _spawn([None], cfg=LOCAL, provider=get_active_provider({}), construct=construct)
    child.model, child.provider = "fallback-model", "fallback-provider"
    child._fallback_activated = True
    entry = _build_result_entry(child, {"final_response": "partial", "completed": False}, 0, 1.0,
                                _SchemaOutcome(None, None, [], 0))
    manifest = entry["execution_manifest"]
    assert manifest["requested"]["model"] == LOCAL["model"]
    assert manifest["resolved"]["model"] == LOCAL["model"]
    assert manifest["effective"]["model"] == "fallback-model"
    assert manifest["effective"]["provider"] == "fallback-provider"
    assert manifest["effective"]["context_length"] == 65536
    assert manifest["effective"]["fallback_activated"] is True
    assert manifest["native_model"] is None and manifest["route_epoch"] is None
    assert "api_key" not in json.dumps(manifest) and "base_url" not in json.dumps(manifest)
    assert _fabricated_entry(0, "error", "failed", child)["execution_manifest"] == manifest
    identifier, _, _ = create_live_transcripts(
        [{"goal": "test"}], delegation_id="manifest-fixture", home=tmp_path)
    assert identifier == "manifest-fixture"
    update_manifest_statuses("manifest-fixture", [entry], home=tmp_path)
    saved = json.loads(_manifest_path("manifest-fixture", tmp_path).read_text())
    assert saved["tasks"][0]["execution_manifest"] == manifest


@pytest.mark.parametrize("isolated", [False, True])
def test_manifest_records_seeded_workspace_without_leaking_to_isolated_child(tmp_path, monkeypatch, isolated):
    from agent.runtime_cwd import reset_session_cwd, set_session_cwd
    from tools.delegate_execution_manifest import capture, report
    from tools.delegate_tool_child_run import _ChildRun
    from tools.delegate_tool_routes import TaskRoute
    from tools.terminal_tool import clear_session_cwd
    monkeypatch.setattr("tools.delegate_tool_child_run._create_isolated_worktree", lambda *args: None)
    monkeypatch.setattr("tools.terminal_tool.register_container_alias", lambda *args: None)
    child = SimpleNamespace(_delegation_context_only=isolated)
    capture(child, TaskRoute({}, {}))
    task_id = f"manifest-workspace-{isolated}"
    token = set_session_cwd(str(tmp_path))
    try:
        run = _ChildRun(child, SimpleNamespace(_current_task_id="manifest-parent"),
                        0, "workspace fixture", task_id, None)
        run.seed_workspace()
        workspace = report(child)["execution_manifest"]["workspace"]
        assert workspace == {"mode": "isolated" if isolated else "shared",
                             "initial_cwd": None if isolated else str(tmp_path)}
    finally:
        reset_session_cwd(token)
        clear_session_cwd(task_id)
