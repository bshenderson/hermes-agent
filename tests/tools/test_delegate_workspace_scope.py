"""Delegation workspace hints honor the owning session (GXTD-850)."""
from types import SimpleNamespace

from agent.runtime_cwd import reset_session_cwd, set_session_cwd
from tools.delegate_tool_progress import _resolve_workspace_hint


def test_session_workspace_precedes_profile_terminal_default(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    session = tmp_path / "session"
    profile.mkdir()
    session.mkdir()
    monkeypatch.setattr("agent.runtime_cwd.scope_terminal_cwd", lambda: str(profile))
    token = set_session_cwd(str(session))
    try:
        assert _resolve_workspace_hint(SimpleNamespace()) == str(session)
    finally:
        reset_session_cwd(token)


def test_seed_workspace_without_parent_terminal_uses_session_directory(tmp_path, monkeypatch):
    from tools.delegate_tool_child_run import _ChildRun
    from tools.terminal_tool import clear_session_cwd, get_session_cwd

    session = tmp_path / "session"
    session.mkdir()
    monkeypatch.setattr("tools.delegate_tool_child_run._create_isolated_worktree", lambda *args: None)
    monkeypatch.setattr("tools.terminal_tool.register_container_alias", lambda *args: None)
    parent_id, child_id = "g850-parent-no-terminal", "g850-child-no-terminal"
    clear_session_cwd(parent_id)
    clear_session_cwd(child_id)
    token = set_session_cwd(str(session))
    try:
        run = _ChildRun(SimpleNamespace(), SimpleNamespace(_current_task_id=parent_id),
                        0, "offline workspace fixture", child_id, None)
        run.seed_workspace()
        assert get_session_cwd(child_id) == str(session)
        assert get_session_cwd(parent_id) is None
    finally:
        reset_session_cwd(token)
        clear_session_cwd(child_id)


def test_seed_workspace_preserves_parent_cd_and_child_isolation(tmp_path, monkeypatch):
    from tools.delegate_tool_child_run import _ChildRun
    from tools.terminal_tool import clear_session_cwd, get_session_cwd, record_session_cwd

    monkeypatch.setattr("tools.delegate_tool_child_run._create_isolated_worktree", lambda *args: None)
    monkeypatch.setattr("tools.terminal_tool.register_container_alias", lambda *args: None)
    parent_id, child_id = "g850-parent-cd", "g850-child-cd"
    record_session_cwd(parent_id, str(tmp_path))
    token = set_session_cwd(str(tmp_path.parent))
    try:
        run = _ChildRun(SimpleNamespace(), SimpleNamespace(_current_task_id=parent_id),
                        0, "offline workspace fixture", child_id, None)
        run.seed_workspace()
        assert get_session_cwd(child_id) == str(tmp_path)
        record_session_cwd(child_id, str(tmp_path.parent))
        assert get_session_cwd(parent_id) == str(tmp_path)
    finally:
        reset_session_cwd(token)
        clear_session_cwd(parent_id)
        clear_session_cwd(child_id)


def test_context_only_child_does_not_inherit_workspace(tmp_path, monkeypatch):
    from tools.delegate_tool_child_run import _ChildRun
    from tools.terminal_tool import clear_session_cwd, get_session_cwd

    def forbidden(*args):
        raise AssertionError("isolated child must not resolve parent workspace")

    monkeypatch.setattr("tools.delegate_tool_progress._resolve_workspace_hint", forbidden)
    child_id = "g850-context-only"
    clear_session_cwd(child_id)
    token = set_session_cwd(str(tmp_path))
    try:
        run = _ChildRun(SimpleNamespace(_delegation_context_only=True), SimpleNamespace(),
                        0, "offline isolated fixture", child_id, None)
        run.seed_workspace()
        assert get_session_cwd(child_id) is None
    finally:
        reset_session_cwd(token)
        clear_session_cwd(child_id)
