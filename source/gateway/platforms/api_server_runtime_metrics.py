"""Private gateway-owned durable Kanban aggregates (GXTD-594).

Retained records are gauges, not lifetime counters or proof of live workers.
Submission HTTP traffic is deliberately absent. No payloads/IDs leave this owner.
"""
from __future__ import annotations

import sqlite3
import time
import asyncio
import hmac
import re
from pathlib import Path

TASK_STATES = ('triage', 'todo', 'scheduled', 'ready', 'running', 'blocked', 'review', 'done', 'archived', 'other')
QUEUE_STATES = ('triage', 'todo', 'scheduled', 'ready', 'blocked', 'review')
OUTCOMES = ('completed', 'blocked', 'crashed', 'timed_out', 'spawn_failed', 'gave_up',
            'reclaimed', 'released', 'review_requested', 'changes_requested', 'scheduled', 'rate_limited', 'other')
EXECUTION = ('spawn_recorded', 'unobserved')


def collect_board(path: Path, *, now=None) -> dict:
    """Bounded read-only snapshot; never initialize a missing board on scrape."""
    now = time.time() if now is None else now
    conn = sqlite3.connect(Path(path).absolute().as_uri() + '?mode=ro', uri=True, timeout=0.2)
    deadline = time.monotonic() + 0.2
    conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    tasks = dict.fromkeys(TASK_STATES, 0)
    oldest = dict.fromkeys(QUEUE_STATES, 0)
    unknown_age = dict.fromkeys(QUEUE_STATES, 0)
    open_runs = dict.fromkeys(('launch_unobserved', 'spawn_recorded'), 0)
    outcomes = {(outcome, execution): 0 for outcome in OUTCOMES for execution in EXECUTION}
    try:
        conn.execute('PRAGMA query_only=ON')
        conn.execute('BEGIN')
        placeholders = ','.join('?' for _ in TASK_STATES[:-1])
        rows = conn.execute(f"""
            SELECT CASE WHEN status IN ({placeholders}) THEN status ELSE 'other' END AS state,
                   COUNT(*), MIN(CASE WHEN created_at > 0 AND created_at <= ? THEN created_at END),
                   SUM(CASE WHEN created_at IS NULL OR created_at <= 0 OR created_at > ? THEN 1 ELSE 0 END)
            FROM tasks GROUP BY state
        """, (*TASK_STATES[:-1], now, now))
        for state, count, created, unknown in rows:
            tasks[state] = count
            if state in oldest:
                oldest[state] = created or 0
                unknown_age[state] = unknown
        placeholders = ','.join('?' for _ in OUTCOMES[:-1])
        rows = conn.execute(f"""
            SELECT r.ended_at IS NULL AS is_open,
                   CASE WHEN r.outcome IN ({placeholders}) THEN r.outcome ELSE 'other' END AS outcome,
                   EXISTS(SELECT 1 FROM task_events e WHERE e.task_id = r.task_id
                          AND e.run_id = r.id AND e.kind = 'spawned') AS spawned,
                   COUNT(*)
            FROM task_runs r GROUP BY is_open, outcome, spawned
        """, OUTCOMES[:-1])
        for is_open, outcome, spawned, count in rows:
            if is_open:
                open_runs['spawn_recorded' if spawned else 'launch_unobserved'] += count
            else:
                outcomes[(outcome, 'spawn_recorded' if spawned else 'unobserved')] += count
        return {'tasks': tasks, 'oldest_created': oldest, 'unknown_age': unknown_age,
                'open_runs': open_runs, 'outcomes': outcomes}
    finally:
        conn.close()


def render(snapshots: list[dict], boards: list[str] | None = None) -> str:
    """Keep compatibility totals; optionally add bounded board-scoped families."""
    if boards is not None and (
            len(boards) != len(snapshots) or not 1 <= len(boards) <= 8
            or any(not isinstance(b, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', b) for b in boards)
            or len(set(boards)) != len(boards)):
        raise ValueError('Board labels must be distinct configured safe slugs')
    lines = []
    def family(name, help_text, samples):
        name = 'hermes_kanban_' + name
        lines.extend([f'# HELP {name} {help_text}', f'# TYPE {name} gauge'])
        lines.extend(f'{name}{labels} {value}' for labels, value in samples)

    family('boards_observed', 'Configured durable boards read successfully, not execution health.', [('', len(snapshots))])
    family('tasks_retained', 'Retained task state; running includes claims before worker spawn.',
           [(f'{{state="{state}"}}', sum(s['tasks'][state] for s in snapshots)) for state in TASK_STATES])
    family('queue_tasks', 'Retained waiting work by state; not active agent execution.',
           [(f'{{state="{state}"}}', sum(s['tasks'][state] for s in snapshots)) for state in QUEUE_STATES])
    family('queue_unknown_created_at', 'Queued tasks with unknown or invalid creation timestamp.',
           [(f'{{state="{state}"}}', sum(s['unknown_age'][state] for s in snapshots)) for state in QUEUE_STATES])
    ages = []
    for state in QUEUE_STATES:
        if any(s['unknown_age'][state] for s in snapshots):
            continue
        stamps = [s['oldest_created'][state] for s in snapshots if s['oldest_created'][state] > 0]
        ages.append((f'{{state="{state}"}}', min(stamps) if stamps else 0))
    family('queue_oldest_created_timestamp_seconds',
           'Oldest task creation timestamp, NOT queue-entry time; zero only for empty state.', ages)
    family('runs_open', 'Unended durable attempts; spawn_recorded is historical launch, NOT worker liveness.',
           [(f'{{stage="{stage}"}}', sum(s['open_runs'][stage] for s in snapshots))
            for stage in ('launch_unobserved', 'spawn_recorded')])
    family('run_outcomes_retained', 'Terminal attempt records, not task success rate; unobserved includes synthetic runs and pruned spawn events.',
           [(f'{{outcome="{outcome}",execution="{execution}"}}', sum(s['outcomes'][(outcome, execution)] for s in snapshots))
            for outcome in OUTCOMES for execution in EXECUTION])
    if boards is not None:
        families = {}
        for board, snapshot in zip(boards, snapshots):
            for line in render([snapshot]).splitlines():
                if 'hermes_kanban_boards_observed' in line:
                    continue
                line = line.replace('hermes_kanban_', 'hermes_kanban_board_')
                if line.startswith('#'):
                    name = line.split()[2]
                    group = families.setdefault(name, [])
                    if line not in group:
                        group.append(line)
                else:
                    name, sample = line.split(' ', 1)
                    name = name.replace('{', '{board="' + board + '",', 1)
                    families[name.split('{', 1)[0]].append(name + ' ' + sample)
        for group in families.values():
            lines.extend(group)
    return '\n'.join(lines) + '\n'


def http_routes(adapter, *, token: str) -> list:
    """Opt-in private base route, never a profile-mirrored execution authority."""
    options = (adapter.config.extra or {}).get('runtime_metrics') or {}
    if options.get('enabled') is not True:
        return []
    boards = options.get('boards')
    if (not isinstance(boards, list) or not 1 <= len(boards) <= 8
            or any(not isinstance(b, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', b) for b in boards)
            or len(set(boards)) != len(boards)):
        raise ValueError('runtime_metrics requires 1..8 distinct explicit board slugs')
    from aiohttp import web
    from hermes_cli import kanban_db as kb
    # Resolve trusted configured boards once, not from request/profile parameters.
    paths = [kb.kanban_db_path(board=board) for board in boards]
    if len(set(paths)) != len(paths):
        raise ValueError('runtime_metrics boards must resolve to distinct databases')
    lock = asyncio.Lock()
    delivery_home = None
    if options.get('delivery_ledgers') is True:
        from hermes_constants import get_hermes_home
        delivery_home = get_hermes_home().resolve()
    lifecycle = None
    if options.get('api_lifecycle') is True:
        from gateway.platforms.api_server_lifecycle_metrics import Collector
        lifecycle = getattr(adapter, '_api_lifecycle_metrics', None)
        if not isinstance(lifecycle, Collector):
            lifecycle = adapter._api_lifecycle_metrics = Collector()

    async def handle(request):
        if request.path != '/metrics/runtime':
            return web.Response(status=404)
        auth = request.headers.get('Authorization', '').strip()
        if (not token or token == adapter._api_key
                or not hmac.compare_digest(auth.encode(), f'Bearer {token}'.encode())):
            return web.Response(status=401)
        if lock.locked():
            return web.Response(status=503)
        async with lock:
            try:
                text = await asyncio.to_thread(lambda: render([collect_board(path) for path in paths], boards))
                if lifecycle is not None:
                    text += lifecycle.render()
                if delivery_home is not None:
                    from gateway.platforms.api_server_delivery_metrics import render as delivery_render
                    text += await asyncio.to_thread(delivery_render, delivery_home)
            except (sqlite3.Error, OSError):
                return web.Response(status=503)
        return web.Response(text=text, headers={'Content-Type': 'text/plain; version=0.0.4; charset=utf-8',
                                                'Cache-Control': 'no-store'})
    return [('GET', '/metrics/runtime', handle)]
