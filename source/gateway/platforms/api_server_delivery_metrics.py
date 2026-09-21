"""Bounded read-only delivery snapshots; no payloads, recovery or profile discovery."""
from contextlib import closing
from datetime import datetime
from pathlib import Path
import math
import sqlite3
import time

SOURCES = {
    'gateway_reply': ('state.db', 'delivery_obligations', 'state',
                      ('pending', 'attempting', 'failed', 'delivered', 'abandoned'),
                      ('pending', 'attempting', 'failed')),
    'cron': ('cron/deliveries.db', 'deliveries', 'status',
             ('pending', 'delivering', 'delivered', 'failed', 'unknown'),
             ('pending', 'delivering')),
}


def timestamp(value, source, now):
    try:
        if source == 'cron':
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                return None
            value = dt.timestamp()
        value = float(value)
        return value if math.isfinite(value) and 0 < value <= now else None
    except (ValueError, TypeError, OverflowError):
        return None


def collect(home: Path, source: str, now=None):
    relative, table, column, states, unresolved = SOURCES[source]
    now = time.time() if now is None else now
    path = (home / relative).resolve()
    deadline = time.monotonic() + 0.2
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0.2)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA busy_timeout=200')
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        db.create_function('safe_created', 1, lambda value: timestamp(value, source, now))
        db.execute('BEGIN')
        placeholders = ','.join('?' for _ in states)
        rows = db.execute(
            f'SELECT CASE WHEN {column} IN ({placeholders}) THEN {column} ELSE \'other\' END, COUNT(*) '
            f'FROM {table} GROUP BY 1', states).fetchall()
        counts = dict.fromkeys((*states, 'other'), 0)
        counts.update(rows)
        placeholders = ','.join('?' for _ in unresolved)
        pending, unknown, oldest = db.execute(
            f'SELECT COUNT(*), COALESCE(SUM(safe_created(created_at) IS NULL),0), '
            f'MIN(safe_created(created_at)) FROM {table} WHERE {column} IN ({placeholders})',
            unresolved).fetchone()
        return {'states': counts, 'unresolved': pending, 'unknown': unknown,
                'oldest': None if unknown else oldest if pending else 0}


def render(home: Path) -> str:
    # Collect every configured source before emitting anything; failures fail the scrape.
    snapshots = {source: collect(home, source) for source in SOURCES}
    lines = []
    families = (
        ('states_retained', 'Retained owner delivery rows, not event rates or client receipt.'),
        ('unresolved', 'Gateway pending/attempting/failed; cron pending/delivering only.'),
        ('unknown_created_at', 'Unresolved rows with unknown original record creation age.'),
        ('oldest_created_timestamp_seconds', 'Oldest unresolved record creation; omitted if any age unknown.'),
    )
    for name, description in families:
        metric = 'hermes_delivery_' + name
        lines.extend([f'# HELP {metric} {description}', f'# TYPE {metric} gauge'])
        for source, data in snapshots.items():
            if name == 'states_retained':
                for state, count in data['states'].items():
                    lines.append(f'{metric}{{source="{source}",state="{state}"}} {count}')
            else:
                key = {'unresolved': 'unresolved', 'unknown_created_at': 'unknown',
                       'oldest_created_timestamp_seconds': 'oldest'}[name]
                if data[key] is not None:
                    lines.append(f'{metric}{{source="{source}"}} {data[key]}')
    return '\n'.join(lines) + '\n'
