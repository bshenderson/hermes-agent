from types import SimpleNamespace

import hermes_cli.cli_tui_mixin as tui_module
from cli import HermesCLI


def _command_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = object()
    cli._agent_running = False
    cli._command_running = True
    cli._last_ctrl_c_time = 0.0
    cli._should_exit = False
    cli._attached_images = []
    cli._tui_cancel_voice_recording = lambda event: False
    cli._tui_cancel_foreground_ui = lambda event, closers: False
    cli._tui_clear_blocking_overlays = lambda event: False
    app = SimpleNamespace(
        current_buffer=SimpleNamespace(text="", reset=lambda: None),
        exit=lambda: (_ for _ in ()).throw(AssertionError("first Ctrl-C must not exit")),
        invalidate=lambda: None,
    )
    return cli, SimpleNamespace(app=app)


def test_ctrl_c_hard_cancels_running_slash_command(monkeypatch):
    cli, event = _command_cli()
    calls = []
    monkeypatch.setattr(tui_module, "request_hard_interrupt", lambda agent: calls.append(agent))

    cli._tui_handle_ctrl_c(event)

    assert calls == [cli.agent]
    assert cli._should_exit is False


def test_ctrl_q_hard_cancels_running_slash_command(monkeypatch):
    cli, event = _command_cli()
    calls = []
    monkeypatch.setattr(tui_module, "request_hard_interrupt", lambda agent: calls.append(agent))

    cli._tui_handle_ctrl_q(event)

    assert calls == [cli.agent]
    assert cli._should_exit is False
