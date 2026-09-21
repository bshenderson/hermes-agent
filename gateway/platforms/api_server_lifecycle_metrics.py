"""Opt-in, process-local API executor telemetry; never session/task identity.

Completion means the agent's return flags, not task quality or client receipt.
The executor owns its lifetime even after cancellation of its asyncio waiter.
"""
from __future__ import annotations

import threading
import time

FAMILIES = ('request', 'durable_run')
OUTCOMES = ('completed', 'incomplete', 'failed', 'interrupted', 'exception', 'unknown')


class Collector:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = {}
        self._outcomes = {(family, outcome): 0 for family in FAMILIES for outcome in OUTCOMES}
        self.started_at = time.time()

    def begin(self, turn):
        with self._lock:
            self._active[turn] = {'family': turn.family, 'tracked': turn.tracks_progress, 'last': None}

    def progress(self, turn):
        with self._lock:
            if turn in self._active:
                self._active[turn]['last'] = time.time()

    def finish(self, turn, outcome):
        with self._lock:
            self._active.pop(turn)
            self._outcomes[(turn.family, outcome)] += 1

    def render(self):
        with self._lock:
            active = [dict(row) for row in self._active.values()]
            outcomes = dict(self._outcomes)
        lines = [
            '# HELP hermes_api_executors_active Agent conversation calls currently executing; not admitted HTTP requests.',
            '# TYPE hermes_api_executors_active gauge',
        ]
        for family in FAMILIES:
            count = sum(row['family'] == family for row in active)
            lines.append(f'hermes_api_executors_active{{family="{family}"}} {count}')
        lines.extend([
            '# HELP hermes_api_executor_outcomes_total Agent return outcomes since this collector started; not delivery or task quality.',
            '# TYPE hermes_api_executor_outcomes_total counter',
        ])
        for (family, outcome), value in outcomes.items():
            lines.append(f'hermes_api_executor_outcomes_total{{family="{family}",outcome="{outcome}"}} {value}')
        lines.extend([
            '# HELP hermes_api_lifecycle_started_timestamp_seconds Collector epoch; counters reset on adapter restart.',
            '# TYPE hermes_api_lifecycle_started_timestamp_seconds gauge',
            f'hermes_api_lifecycle_started_timestamp_seconds {self.started_at}',
        ])
        for name, help_text in (
            ('tracked_active', 'Active executors with an existing callback instrumented; does not enable streaming.'),
            ('unobserved', 'Tracked active executors without an observed nonempty text or tool callback.'),
            ('oldest_timestamp_seconds', 'Oldest latest callback among tracked executors; omitted for unknown age, zero if empty. Not client receipt.'),
        ):
            lines.extend([f'# HELP hermes_api_progress_{name} {help_text}',
                          f'# TYPE hermes_api_progress_{name} gauge'])
            for family in FAMILIES:
                tracked = [row for row in active if row['family'] == family and row['tracked']]
                unknown = sum(row['last'] is None for row in tracked)
                if name == 'oldest_timestamp_seconds' and unknown:
                    continue
                value = (len(tracked) if name == 'tracked_active' else unknown if name == 'unobserved'
                         else min((row['last'] for row in tracked), default=0))
                lines.append(f'hermes_api_progress_{name}{{family="{family}"}} {value}')
        return '\n'.join(lines) + '\n'


class Turn:
    def __init__(self, collector, family):
        if family not in FAMILIES:
            raise ValueError('Unknown API executor family')
        self.collector = collector
        self.family = family
        self.tracks_progress = False

    def wrap_callback(self, kind, callback):
        if self.collector is None or callback is None:
            return callback
        self.tracks_progress = True

        def observed(*args, **kwargs):
            result = callback(*args, **kwargs)
            if kind != 'text' or (args and isinstance(args[0], str) and args[0]):
                self.collector.progress(self)
            return result
        return observed

    def run(self, callback, **kwargs):
        if self.collector is None:
            return callback(**kwargs)
        self.collector.begin(self)
        outcome = 'exception'
        try:
            result = callback(**kwargs)
            if not isinstance(result, dict):
                outcome = 'unknown'
            elif result.get('interrupted'):
                outcome = 'interrupted'
            elif result.get('failed'):
                outcome = 'failed'
            elif result.get('partial') or not result.get('completed', True):
                outcome = 'incomplete'
            else:
                outcome = 'completed'
            return result
        finally:
            self.collector.finish(self, outcome)


def new_turn(adapter, family):
    collector = getattr(adapter, '_api_lifecycle_metrics', None)
    return Turn(collector if isinstance(collector, Collector) else None, family)
