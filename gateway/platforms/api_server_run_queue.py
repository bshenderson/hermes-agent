"""Bounded FIFO API-turn waiters; native generation leases stay downstream."""
from __future__ import annotations

import asyncio
import time
from functools import wraps


def queued_count(adapter) -> int:
    return len(getattr(adapter, '_api_run_waiters', ()))


async def wait_for_capacity(adapter, request):
    from gateway.platforms import api_server as api
    limit = getattr(adapter, '_max_concurrent_runs', 0)
    bound = getattr(adapter, '_max_pending_runs', 0)
    if type(limit) is not int or limit <= 0 or type(bound) is not int or bound <= 0:
        return None  # Preserve legacy immediate-refusal behavior unless opted in.
    waiters = adapter._api_run_waiters
    # The caller's admitted-request reservation is already counted but has
    # not joined the queue. Other queued reservations do not consume run slots.
    if not waiters and adapter.active_agent_work_count() - 1 < limit:
        return None
    if len(waiters) >= bound:
        return api._error_response('API pending queue is full', 429,
                                   code='run_queue_full', headers={'Retry-After': '1'})
    ticket = object()
    waiters.append(ticket)
    deadline = time.monotonic() + adapter._pending_run_timeout
    try:
        while True:
            draining = adapter._draining_response()
            if draining is not None:
                return draining
            transport = getattr(request, 'transport', None)
            if ((isinstance(request, api.web.Request) and transport is None)
                    or (transport is not None and transport.is_closing())):
                return api._error_response('Client disconnected while queued', 499, code='run_queue_disconnected')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return api._error_response('API pending queue wait expired', 429,
                                           code='run_queue_timeout', headers={'Retry-After': '1'})
            if waiters[0] is ticket and adapter.active_agent_work_count() - len(waiters) < limit:
                return None
            await asyncio.sleep(min(.05, remaining))
    finally:
        waiters.remove(ticket)


def admit_request(handler):
    """Authenticate, reserve drain-visible work, then wait before agent work."""
    @wraps(handler)
    async def wrapped(adapter, request, *args, **kwargs):
        from gateway.platforms import api_server as api
        auth_error = (adapter._check_run_auth(request, permission='dispatch')
                      if api._api_runs._uses_room_run_auth(adapter, request)
                      else adapter._check_auth(request))
        if auth_error:
            return auth_error
        draining = adapter._draining_response()
        if draining is not None:
            return draining
        reservation = {'active': True}
        token = api._api_agent_request_reservation.set(reservation)
        adapter._pending_agent_requests += 1
        try:
            limited = await wait_for_capacity(adapter, request)
            if limited is not None:
                return limited
            return await handler(adapter, request, *args, **kwargs)
        finally:
            api._release_pending_api_work(adapter, reservation)
            api._api_agent_request_reservation.reset(token)
    return wrapped
