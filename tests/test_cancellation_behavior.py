
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from agent.interrupt_control import InterruptControlMixin
from gateway.session_api_turn import observe_api_turn

class MockAgent(InterruptControlMixin):
    def __init__(self, name, children=None):
        self.name = name
        self._active_children = children or []
        self._active_children_lock = asyncio.Lock()
        self._execution_thread_id = 123
        self._interrupt_requested = False
        self._tool_interrupt_reason = None
        self._interrupt_message = None
        self._turn_user_intervened = False
        self._pending_redirect_lock = asyncio.Lock()
        self._pending_redirect = None
        self._liveness_activity_lock = MagicMock()
        self._liveness_activity_lock.return_value = asyncio.Lock()

    def _liveness_activity_lock(self):
        return self._liveness_activity_lock()

    def __repr__(self):
        return f"<MockAgent {self.name}>"

class MockAuthority:
    def __init__(self):
        self.waiters = {}
        self.api_observers = {}
        self.db = MagicMock()

@pytest.mark.asyncio
async def test_canonical_observer_vs_explicit_stop():
    # Setup parent and child
    child = MockAgent("child")
    parent = MockAgent("parent", children=[child])
    
    authority = MockAuthority()
    ref = "ref1"
    row = {'admission_id': 'adm1', 'status': 'running'}
    admitted = (authority, ref, row)
    
    # Create a waiter that we can control
    waiter = asyncio.get_running_loop().create_future()
    authority.waiters['adm1'] = waiter
    
    # Call observe_api_turn in a task
    # We mock the internal get_session_admission to return the row
    with patch('gateway.session_api_turn.get_session_admission', return_value=row):
        observer_task = asyncio.create_task(observe_api_turn(admitted, stream_delta_callback=lambda x: None))
        
        # Give it a moment to register and reach the shield(waiter)
        await asyncio.sleep(0.1)
        
        # 1. Verify that cancelling the observer task (canonical observer cancellation)
        # does NOT interrupt the agent.
        observer_task.cancel()
        try:
            await observer_task
        except asyncio.CancelledError:
            pass
        
        assert parent._interrupt_requested is False, "Canonical observer cancellation should not stop the agent"
        assert child._interrupt_requested is False, "Canonical observer cancellation should not stop the child"

        # 2. Verify that an explicit stop (hard_interrupt) DOES stop both parent and child.
        parent.hard_interrupt(message="Stop now")
        
        assert parent._interrupt_requested is True, "Explicit stop should interrupt the parent"
        assert parent._tool_interrupt_reason == "explicit stop requested"
        assert child._interrupt_requested is True, "Explicit stop should propagate to children"
        assert child._tool_interrupt_reason == "explicit stop requested"

if __name__ == "__main__":
    # Simple wrapper to run the test logic if not using pytest
    async def run():
        try:
            await test_canonical_observer_vs_explicit_stop()
            print("Test Passed: Behavioral qualification successful.")
        except Exception as e:
            print(f"Test Failed: {e}")
            raise e
    asyncio.run(run())
