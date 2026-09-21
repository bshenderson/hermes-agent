"""Manual commands must start a new cancellation lifetime after a stopped command."""
import threading

from agent.interrupt_control import InterruptControlMixin
from tests.cli.test_cli_init import _make_cli
from tests.cli.test_manual_compress import _make_history


class InterruptAgent(InterruptControlMixin):
    def __init__(self, session_id):
        self.session_id = session_id
        self._hard_interrupt_requested = threading.Event()
        self._execution_thread_id = None
        self._active_children_lock = threading.Lock()
        self._active_children = []
        self.quiet_mode = True
        self.tools = None
        self._cached_system_prompt = ''
        self.attempts = 0
        self.recovery_dispatched = False

    def _compress_context(self, messages, *args, **kwargs):
        self.attempts += 1
        if self.attempts == 1:
            raise KeyboardInterrupt
        if self._hard_interrupt_requested.is_set():
            return messages, ''
        self.recovery_dispatched = True
        return messages[:2], ''


def test_manual_compress_can_recover_after_keyboard_interrupt():
    shell = _make_cli()
    shell.conversation_history = _make_history()
    shell.agent = InterruptAgent(shell.session_id)
    shell._manual_compress('/compress')
    assert shell.agent._hard_interrupt_requested.is_set()
    assert len(shell.conversation_history) == 4
    shell._manual_compress('/compress')
    assert shell.agent.recovery_dispatched
    assert len(shell.conversation_history) == 2


def test_preview_does_not_reset_stopped_command():
    shell = _make_cli()
    shell.conversation_history = _make_history()
    shell.agent = InterruptAgent(shell.session_id)
    shell.agent.hard_interrupt()
    shell._manual_compress('/compress --preview')
    assert shell.agent._hard_interrupt_requested.is_set()
    assert shell.agent.attempts == 0


def test_manual_command_does_not_clear_active_turn_stop():
    shell = _make_cli()
    shell.conversation_history = _make_history()
    shell.agent = InterruptAgent(shell.session_id)
    shell.agent.hard_interrupt()
    shell._agent_running = True
    shell._manual_compress('/compress')
    assert shell.agent._hard_interrupt_requested.is_set()
    assert shell.agent.attempts == 0
