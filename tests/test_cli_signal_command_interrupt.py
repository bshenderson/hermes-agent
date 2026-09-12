import signal
from types import SimpleNamespace

import pytest

import cli as cli_module
import hermes_cli.cli_tui_mixin as tui_module
from cli import HermesCLI


@pytest.mark.parametrize("agent_running,command_running", [(True, False), (False, True)])
def test_tui_shutdown_signal_hard_cancels_active_work(monkeypatch, agent_running, command_running):
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = object()
    cli._agent_running = agent_running
    cli._command_running = command_running
    calls = []

    monkeypatch.setattr(cli_module, "_arm_exit_watchdog_on_shutdown_signal", lambda: None)
    monkeypatch.setattr(
        cli_module, "_interrupt_agent_for_signal", lambda agent, signum: calls.append((agent, signum)))

    with pytest.raises(KeyboardInterrupt):
        cli._tui_signal_handler(signal.SIGTERM, None)

    assert calls == [(cli.agent, signal.SIGTERM)]


def test_tui_shutdown_signal_does_not_interrupt_idle_agent(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = object()
    cli._agent_running = False
    cli._command_running = False
    calls = []

    monkeypatch.setattr(cli_module, "_arm_exit_watchdog_on_shutdown_signal", lambda: None)
    monkeypatch.setattr(
        cli_module, "_interrupt_agent_for_signal", lambda agent, signum: calls.append((agent, signum)))

    with pytest.raises(KeyboardInterrupt):
        cli._tui_signal_handler(signal.SIGTERM, None)

    assert calls == []


def test_ctrl_c_hard_cancels_running_slash_command(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = object()
    cli._agent_running = False
    cli._command_running = True
    cli._last_ctrl_c_time = 0

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
    calls = []
    monkeypatch.setattr(tui_module, "request_hard_interrupt", lambda agent: calls.append(agent))

    cli._tui_handle_ctrl_c(SimpleNamespace(app=app))

    assert calls == [cli.agent]
    assert cli._should_exit is False
