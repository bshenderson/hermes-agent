"""Attempt-scoped sync OpenAI transport ownership for protected auxiliary work."""
from contextlib import contextmanager
import threading

from agent.process_bootstrap import build_keepalive_http_client


@contextmanager
def cancellable_openai_attempt(client, cancel_check, *, verify=True):
    """Keep the cached SDK client alive; abort only a fresh transport owner's I/O.

    Shared keepalive pools stamp each view's in-flight requests with its owner id.
    The existing socket walker honors that id, including before response headers.
    Only the provider thread closes descriptors; the watcher uses shutdown only.
    """
    from agent.agent_runtime_helpers import force_close_tcp_sockets
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    http = build_keepalive_http_client(str(client.base_url), verify=verify)
    if http is None:
        raise RuntimeError('Could not create owned auxiliary HTTP transport')
    done = threading.Event()
    owned = None
    watcher = None

    def reject_cancelled_retry(request):
        # SDK retries retain their ordinary policy, but a stopped attempt must
        # never send another request even if shutdown surfaced as a read error.
        if cancel_check():
            raise AuxiliaryExplicitCancellation()

    def watch():
        while not done.wait(.02):
            if cancel_check():
                # A connect can finish after the first sweep; keep observing
                # until the provider has actually unwound its owned request.
                force_close_tcp_sockets(owned)

    try:
        http.event_hooks['request'].append(reject_cancelled_retry)
        owned = client.with_options(http_client=http)
        if cancel_check():
            raise AuxiliaryExplicitCancellation()
        watcher = threading.Thread(target=watch, name='hermes-aux-socket-cancel', daemon=True)
        watcher.start()
        yield owned
    finally:
        done.set()
        if watcher is not None:
            watcher.join()
        # This runs on the I/O-owning provider thread, never the interrupt thread.
        try:
            http.close()
        finally:
            if cancel_check():
                raise AuxiliaryExplicitCancellation()
