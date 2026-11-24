"""Opt-in mapped secret resolution at the credential's actual consumer.

No environment publication or new value cache: existing source/profile/identity
custody remains authoritative. A configured failure never falls back to process
credentials. Bulk sources are incompatible with strict lazy mode.
"""
from pathlib import Path
from contextvars import ContextVar
import sys

_PROBING = ContextVar('lazy_secret_probe', default=False)


class SecretDeferred(RuntimeError):
    """Configured capability whose credential is intentionally not fetched yet."""


class _DeferredCredential:
    """Truthful configuration presence, never a usable credential string."""

    def __bool__(self):
        return True

    def __str__(self):
        raise SecretDeferred('credential use is not an availability probe')

    def __getattr__(self, name):
        raise SecretDeferred('credential use is not an availability probe')


def probe_requirements(fn):
    token = _PROBING.set(True)
    try:
        return bool(fn())
    except SecretDeferred:
        # A predicate needing the actual credential cannot establish availability
        # offline. Never skip the rest of its non-credential prerequisites.
        return False
    finally:
        _PROBING.reset(token)


class SecretUnavailable(RuntimeError):
    """Sanitized, capability-local configured-source refusal."""


def startup_config(cfg: dict) -> tuple[dict, frozenset[str]]:
    if cfg.get('lazy') is not True:
        return cfg, frozenset()
    surfaces = cfg.get('startup_names', {})
    if not isinstance(surfaces, dict):
        raise SecretUnavailable('invalid startup secret classification')
    from hermes_cli._parser import command_argv
    command = command_argv(sys.argv[1:])
    surface = command[0] if command else 'chat'
    raw = surfaces.get(surface, [])
    if not isinstance(raw, list) or not all(isinstance(n, str) for n in raw):
        raise SecretUnavailable('invalid startup secret name list')
    names = frozenset(raw)
    result = dict(cfg)
    source = dict(cfg.get('onepassword', {}))
    source['env'] = {k: v for k, v in source.get('env', {}).items() if k in names}
    if names and (not source.get('enabled') or names - set(source['env'])):
        raise SecretUnavailable('required startup credential has no configured source; no credential fallback')
    if not source['env']:
        source['enabled'] = False
    result['onepassword'] = source
    return result, names


def resolve(home: Path, name: str) -> tuple[bool, str | _DeferredCredential | None]:
    from agent.safe_worker_policy import worker_config_snapshot
    if worker_config_snapshot() is not None:
        return False, None
    from hermes_cli.env_loader import _load_secrets_config, resolve_profile_secret_names
    cfg = _load_secrets_config(home, strict=True)
    if cfg.get('lazy') is not True:
        return False, None
    # Only mapped 1Password is deferred. Other explicitly enabled sources retain
    # their startup classification; a scoped read must never invoke those bulk sources.
    section = cfg.get('onepassword', {})
    if not isinstance(section, dict) or section.get('enabled') is not True:
        return False, None
    mapping = section.get('env', {})
    if not isinstance(mapping, dict) or name not in mapping:
        return False, None
    # Preserve the managed authority applied last by build_profile_secret_scope.
    # Frozen private workers already returned above and retain their own snapshot.
    from hermes_cli.managed_scope import load_managed_env
    managed = load_managed_env()
    if name in managed:
        return True, managed[name]
    if _PROBING.get():
        return True, _DeferredCredential()
    from agent.secret_sources.deadline import secret_budget
    with secret_budget(float(cfg.get('lazy_timeout_seconds', 5))):
        values = resolve_profile_secret_names(home, [name], mapped_only=True)
    value = values.get(name)
    if not value:
        raise SecretUnavailable(f'configured secret unavailable: {name}; no credential fallback')
    return True, value
