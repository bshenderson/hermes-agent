"""Opt-in supplied-context boundary for trusted profile-routed children.

This is structural non-inheritance, not semantic filtering of model-authored briefs.
Unenrolled routes retain the existing tool/prefill/workspace inheritance contract.
"""
from pathlib import Path


def supplied_context_only(profile_home: Path | None) -> bool:
    if profile_home is None:
        return False
    from hermes_cli.config import load_config_readonly
    from tools.delegate_tool_routes import delegated_profile_scope

    with delegated_profile_scope(profile_home):
        delegation = load_config_readonly().get('delegation') or {}
    if not isinstance(delegation, dict):
        raise ValueError('Invalid target delegation configuration')
    if 'child_boundary' not in delegation:
        return False
    if delegation['child_boundary'] != 'supplied_context_only':
        raise ValueError('Unsupported target delegation.child_boundary')
    return True


def constrain_child(child) -> None:
    """Bind the immutable-in-practice native child policy before any generation."""
    child._delegation_context_only = True
    child.tools = []
    child.valid_tool_names = set()
