"""Run SSE helpers recovered from merge parent 5141a32be2^2.

This existing bounded contract was dropped during the fork merge, while its
callers remained. Event replay is separate from pending agent-run admission.
"""
import asyncio
import json
from collections import deque
from typing import Any, Callable, Optional

_TOOL_COMPLETED_PREVIEW_MAX_CHARS = 500


def _tool_completed_preview(result: Any, redact_sensitive_text: Callable[..., str]) -> str:
    """Redact before truncation so a cut never leaves a secret prefix on wire."""
    if result is None:
        return ""
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    preview = redact_sensitive_text(text, force=True)
    limit = _TOOL_COMPLETED_PREVIEW_MAX_CHARS
    return preview if len(preview) <= limit else preview[: limit - 3] + "..."


_RUN_STREAM_SUBSCRIBER_OVERFLOW = object()
_RUN_STREAM_WRITE_TIMEOUT = 5.0


class _RunStream:
    """Sequence and retain one run's events while fanning out to SSE clients."""

    BACKLOG_LIMIT = 1000
    SUBSCRIBER_QUEUE_LIMIT = 256

    def __init__(self) -> None:
        self.subscribers: set[asyncio.Queue] = set()
        self.backlog: deque[tuple[int, Optional[dict[str, Any]]]] = deque(maxlen=self.BACKLOG_LIMIT)
        self.next_seq = 0
        self.terminal = False

    def put_nowait(self, event: Optional[dict[str, Any]]) -> None:
        if self.terminal:
            return
        seq = self.next_seq
        self.next_seq += 1
        self.backlog.append((seq, event))
        if event is None:
            self.terminal = True
        for queue in list(self.subscribers):
            try:
                queue.put_nowait((seq, event))
            except asyncio.QueueFull:
                self.detach(queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait((seq, _RUN_STREAM_SUBSCRIBER_OVERFLOW))

    def attach(self, last_seq: int = -1) -> tuple[asyncio.Queue, list[tuple[int, Optional[dict[str, Any]]]]]:
        replay = [(seq, event) for seq, event in self.backlog if seq > last_seq]
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.SUBSCRIBER_QUEUE_LIMIT + len(replay))
        self.subscribers.add(queue)
        return queue, replay

    def detach(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)
