"""Opt-in supplied-context boundary for trusted profile-routed children.

This is structural non-inheritance, not semantic filtering of model-authored briefs.
Unenrolled routes retain the existing tool/prefill/workspace inheritance contract.
"""
from pathlib import Path


def _target_delegation(profile_home: Path | None) -> dict:
    if profile_home is None:
        return {}
    from hermes_cli.config import load_config_readonly
    from tools.delegate_tool_routes import delegated_profile_scope

    with delegated_profile_scope(profile_home):
        delegation = load_config_readonly().get('delegation') or {}
    if not isinstance(delegation, dict):
        raise ValueError('Invalid target delegation configuration')
    return delegation


def supplied_context_only(profile_home: Path | None) -> bool:
    delegation = _target_delegation(profile_home)
    if 'child_boundary' not in delegation:
        return False
    if delegation['child_boundary'] not in ('supplied_context_only', 'bounded_tools'):
        raise ValueError('Unsupported target delegation.child_boundary')
    return delegation['child_boundary'] == 'supplied_context_only'


def child_tool_ceiling(profile_home: Path | None, parent) -> frozenset[str] | None:
    """None preserves legacy inheritance; an empty ceiling grants nothing.

    Only native, currently carried session grants count, never requested toolsets
    or model-authored task text. Deferred capabilities must first be authorized
    on the parent surface. Snapshot before entering the target profile scope.
    """
    parent_names = frozenset(
        (entry.get('function') or {}).get('name')
        for entry in (getattr(parent, 'tools', None) or []))
    parent_names &= frozenset(getattr(parent, 'valid_tool_names', None) or ())
    delegation = _target_delegation(profile_home)
    if 'child_boundary' not in delegation:
        return None
    mode = delegation['child_boundary']
    if mode == 'supplied_context_only':
        return frozenset()
    if mode != 'bounded_tools':
        raise ValueError('Unsupported target delegation.child_boundary')
    ceiling = delegation.get('child_tool_ceiling', [])
    if not isinstance(ceiling, list) or any(not isinstance(n, str) for n in ceiling):
        raise ValueError('Invalid target delegation.child_tool_ceiling')
    supported = frozenset({'web_search', 'web_extract'})
    if not set(ceiling) <= supported:
        raise ValueError('Unsupported target delegation.child_tool_ceiling')
    return supported.intersection(ceiling, parent_names)


def child_tool_denied(child, name: str) -> bool:
    if getattr(child, '_delegation_context_only', False) is True:
        return True
    if not isolated_child(child):
        return False
    ceiling = getattr(child, '_delegation_tool_ceiling', frozenset())
    return not isinstance(ceiling, frozenset) or name not in ceiling


def isolated_child(child) -> bool:
    return (getattr(child, '_delegation_context_only', False) is True
            or getattr(child, '_delegation_isolated_context', False) is True)


def filter_child_tools(child, definitions: list) -> list:
    return [entry for entry in definitions
            if not child_tool_denied(child, (entry.get('function') or {}).get('name', ''))]


def constrain_child(child, ceiling: frozenset[str] = frozenset()) -> None:
    """Bind the immutable-in-practice native child policy before any generation."""
    child._delegation_context_only = not ceiling
    child._delegation_isolated_context = True
    child._delegation_tool_ceiling = frozenset(ceiling)
    child.tools = filter_child_tools(child, child.tools)
    child.valid_tool_names = {(entry.get('function') or {}).get('name') for entry in child.tools}
