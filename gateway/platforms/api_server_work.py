"""Pending API work ownership across handler and detached-task lifetimes."""
from contextlib import contextmanager


@contextmanager
def reserve_pending_api_work(adapter):
    """Keep externally-triggered work visible until its current owner retires it."""
    from gateway.platforms.api_server import _release_pending_api_work

    reservation = {"active": True, "detached": False}
    adapter._pending_agent_requests += 1
    try:
        yield reservation
    finally:
        if not reservation["detached"]:
            _release_pending_api_work(adapter, reservation)
