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
