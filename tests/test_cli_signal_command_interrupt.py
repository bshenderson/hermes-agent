import signal

import pytest

import cli as cli_module
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
