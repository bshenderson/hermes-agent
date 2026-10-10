"""Exact custom-model output limits, re-resolved when the runtime route changes.

Caller/global limits retain precedence. A model-specific reservation must never
become the caller default for a later model, provider, or turn-scoped fallback.
"""
from __future__ import annotations


def positive_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    return value if type(value) is int and value > 0 else None


def resolve_output_limit(config, *, model, provider, requested_provider='', base_url='', explicit=None):
    caller = positive_int(explicit)
    if caller is not None:
        return caller
    model_section = config.get('model')
    global_limit = positive_int(model_section.get('max_tokens')) if isinstance(model_section, dict) else None
    if global_limit is not None:
        return global_limit
    identities = {str(p or '').lower().removeprefix('custom:')
                  for p in (provider, requested_provider)}
    # A built-in provider must not inherit a local route's limit from a matching URL/model.
    if provider not in {'custom', ''} and not any(
        name in identities for name in config.get('providers', {})
    ):
        return None
    candidates = []
    for name, entry in config.get('providers', {}).items():
        if not isinstance(entry, dict) or entry.get('enabled') is False:
            continue
        endpoint = entry.get('api') or entry.get('base_url') or entry.get('url') or ''
        if str(endpoint).rstrip('/') != str(base_url).rstrip('/'):
            continue
        if identities - {'custom', ''} and name not in identities:
            continue
        models = entry.get('models')
        model_config = models.get(model) if isinstance(models, dict) else None
        if not isinstance(model_config, dict):
            continue
        limit = positive_int(model_config.get('max_tokens'))
        if limit is not None:
            candidates.append(limit)
    # Ambiguous canonical custom routes fail closed rather than inheriting another consumer.
    return candidates[0] if len(candidates) == 1 else None


def apply_output_limit(agent, config=None):
    if config is None:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly() or {}
    agent.max_tokens = resolve_output_limit(
        config, model=agent.model, provider=agent.provider,
        requested_provider=getattr(agent, 'requested_provider', ''),
        base_url=agent.base_url, explicit=getattr(agent, '_explicit_max_tokens', None),
    )
    agent._ephemeral_max_output_tokens = None
    return agent.max_tokens


def update_compressor_output_limit(agent):
    compressor = getattr(agent, 'context_compressor', None)
    if compressor is not None:
        # update_model(max_tokens=None) deliberately means preserve; route changes instead
        # require clearing a previous route's reservation before recalculating its threshold.
        compressor.max_tokens = agent.max_tokens
