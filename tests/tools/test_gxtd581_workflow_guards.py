"""Real Qwen workflow regressions, without relaxing shell background rules."""
from tools.terminal_tool_guards import _foreground_background_guidance


def test_literal_cd_prefix_preserves_inert_python_heredoc():
    command = "cd /tmp && python3 - <<'PY'\nprint({1, 2} & {2, 3})\nPY\n"
    assert _foreground_background_guidance(command) is None
    assert _foreground_background_guidance(command + 'sleep 30 &\n') is not None
    assert _foreground_background_guidance("cd /tmp && bash <<'PY'\nsleep 30 &\nPY\n") is not None
    assert _foreground_background_guidance("cd $(pwd) && python3 - <<'PY'\nx = a & b\nPY\n") is not None


def test_literal_cd_prefix_does_not_mask_shell_lists_or_expansion():
    for command in [
        "cd /tmp; python3 - <<'PY'\nx = a & b\nPY\n",
        "cd /tmp && python3 - <<PY\n$(sleep 30 &)\nPY\n",
        "cd /tmp && python3 - <<'PY' | bash\nsleep 30 &\nPY\n",
    ]:
        assert _foreground_background_guidance(command) is not None
