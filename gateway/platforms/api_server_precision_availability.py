"""One profile-authorized route choice before agent construction; never a retry.

The admission snapshot is advisory. A late admission refusal does not authorize
another provider attempt. General runtime and child fallback chains stay empty.
"""
from __future__ import annotations

import http.client
import json
from urllib.parse import quote

PROFILES = frozenset({'homelab-coordinator', 'homelab-coding', 'homelab-ops',
                      'ops-admin', 'ops-engineer', 'ops-monitor'})
MODELS = frozenset({'operator-default', 'precision-default'})
CLOUD_PROVIDER = 'openai-codex'
CLOUD_MODEL = 'gpt-6.1-sol'


class AvailabilityRefused(RuntimeError):
    """No model or tools were dispatched by this selection."""


def probe_local(base_url, model):
    """Read a fixed loopback origin only: no bearer, redirect or prompt transport."""
    if str(base_url).rstrip('/') != 'http://127.0.0.1:8090/v1' or model not in MODELS:
        raise ValueError('unsupported admission destination')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('duplicate availability field')
            out[key] = value
        return out
    connection = http.client.HTTPConnection('127.0.0.1', 8090, timeout=2)
    try:
        connection.request('GET', '/routing/precision-availability?model=' + quote(model, safe=''))
        response = connection.getresponse()
        raw = response.read(4097)
        if response.status != 200 or len(raw) > 4096:
            raise ValueError('availability response refused')
        observed = json.loads(raw, object_pairs_hook=unique)
        if (not isinstance(observed, dict) or observed.get('schema') != 'homelab.precision-availability.v1'
                or observed.get('model') != model or type(observed.get('available')) is not bool):
            raise ValueError('invalid availability observation')
        return observed
    finally:
        connection.close()


def select_runtime(*, enabled, profile, model, explicit_selection, probe, resolve):
    """Select once using server-owned scope; callbacks are readiness/auth only."""
    if enabled is not True or profile not in PROFILES or model not in MODELS or explicit_selection:
        return None
    try:
        observed = probe(model)
    except Exception:
        observed = {'available': False, 'reason': 'observation_unavailable'}
    if isinstance(observed, dict) and observed.get('available') is True:
        return {'model': model, 'runtime': None, 'notice': '', 'local_no_wait': True}
    reason = observed.get('reason') if isinstance(observed, dict) else None
    if reason not in {'backend_fenced', 'backend_unavailable', 'backend_cooldown',
                      'capacity_busy', 'observation_unavailable', 'unconfigured_model'}:
        reason = 'observation_unavailable'
    try:
        runtime = resolve(CLOUD_PROVIDER, CLOUD_MODEL)
    except Exception as exc:
        # Resolver exception messages may contain credentials; never render them.
        raise AvailabilityRefused('Precision service unavailable; approved cloud credentials could not be resolved. No model or tools dispatched.') from exc
    if not isinstance(runtime, dict) or runtime.get('provider') != CLOUD_PROVIDER:
        raise AvailabilityRefused('Precision service unavailable; approved cloud identity could not be resolved. No model or tools dispatched.')
    return {'model': CLOUD_MODEL, 'runtime': runtime, 'local_no_wait': False,
            'notice': f'Precision route: Sparky unavailable ({reason}); using {CLOUD_PROVIDER}/{CLOUD_MODEL} for this turn. Profile, tools and memory are unchanged.'}
