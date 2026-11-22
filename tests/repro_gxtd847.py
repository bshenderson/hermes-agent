
import asyncio
import sys
import os
import threading
from unittest.mock import MagicMock, AsyncMock

# Add workspace to PYTHONPATH
sys.path.append('/home/barry/.hermes/worktrees/gxtd847-cancellation')

from run_agent import AIAgent
from agent.interrupt_compat import request_hard_interrupt

class LongRunningModel:
    async def generate(self, *args, **kwargs):
        # Simulate a long running model call that doesn't check interrupt flags immediately
        # but will eventually finish.
        await asyncio.sleep(2)
        return MagicMock(content="Child response")

async def test_repro_gxtd847():
    print("Starting repro...")
    
    # Setup parent and child
    parent = AIAgent(user_name="parent", model="mock")
    child = AIAgent(user_name="child", model="mock")
    child.model = LongRunningModel() # Give child a slow model
    
    # Setup delegation
    parent._active_children = [child]
    parent._active_children_lock = threading.Lock()
    
    # Simulate the parent turn loop waiting for the child
    async def parent_turn_loop():
        print("Parent: waiting for child...")
        try:
            # This simulates 'await child.run_turn()'
            await child.run_turn() 
            print("Parent: child completed")
        except asyncio.CancelledError:
            print("Parent: task cancelled")
            raise

    # Start parent turn in a task
    parent_task = asyncio.create_task(parent_turn_loop())
    
    # Let it start
    await asyncio.sleep(0.1)
    
    # Simulate gateway disconnect
    print("Gateway: SSE client disconnected. Calling _abandon_agent_task...")
    request_hard_interrupt(parent, "SSE client disconnected")
    parent_task.cancel()
    
    try:
        await parent_task
    except asyncio.CancelledError:
        print("Parent turn loop caught cancellation.")

    # Now we check if the child is still running or was interrupted
    # In a real scenario, the child's run_turn() loop would check self._interrupt_requested
    print(f"Child _interrupt_requested: {child._interrupt_requested}")
    hard_set = child._hard_interrupt_requested.is_set() if hasattr(child, '_hard_interrupt_requested') else False
    print(f"Child _hard_interrupt_requested set: {hard_set}")
    
    # The child should be interrupted. If it's not, it will "complete normally" after its sleep.
    assert child._interrupt_requested is True, "Child should have been interrupted"
    assert hard_set is True, "Child should have received a HARD interrupt"
    print("Repro test passed (child was interrupted)!")

if __name__ == "__main__":
    try:
        asyncio.run(test_repro_gxtd847())
    except Exception as e:
        print(f"Repro failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
