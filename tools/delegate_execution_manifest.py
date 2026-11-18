"""Allowlisted execution observations, separate from backend labels and authority."""

from copy import deepcopy
from typing import Any


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _effective(child: Any) -> dict[str, Any]:
    context = getattr(getattr(child, "context_compressor", None), "context_length", None)
    return {
        "provider": _text(getattr(child, "provider", None)),
        "model": _text(getattr(child, "model", None)),
        "context_length": context if type(context) is int and context > 0 else None,
        "fallback_activated": getattr(child, "_fallback_activated", False) is True,
    }


def capture(child: Any, route: Any) -> None:
    requested = route.requested if route.requested is not None else route.routing_cfg
    raw = route.routing_cfg.get("execution_identity")
    identity = raw if isinstance(raw, dict) and route.profile_home is None else {}
    child._delegate_execution_manifest = {
        "version": 1,
        "claim_type": "runtime_observation",
        "backend": route.backend,
        "requested": {key: _text(requested.get(key)) for key in ("model", "provider", "profile")},
        "resolved": _effective(child),
        # Client model names do not establish native weights or an admission epoch.
        "native_model": None,
        "route_epoch": None,
        "workspace": None,
    }
    snapshot = child._delegate_execution_manifest
    if identity:
        snapshot["requested"]["model"] = _text(identity.get("selector")) or snapshot["requested"]["model"]
        for key in ("native_model", "route_epoch", "epoch_kind"):
            snapshot[key] = _text(identity.get(key))
        context = identity.get("context_length")
        snapshot["admitted_context_length"] = context if type(context) is int and context > 0 else None
        snapshot["identity_claim_type"] = "provider_observation"


def report(child: Any) -> dict[str, Any]:
    snapshot = getattr(child, "_delegate_execution_manifest", None)
    if not isinstance(snapshot, dict):
        return {}
    return {"execution_manifest": {**deepcopy(snapshot), "effective": _effective(child)}}


def capture_workspace(child: Any, task_id: str, *, isolated: bool = False, worktree: bool = False) -> None:
    snapshot = getattr(child, "_delegate_execution_manifest", None)
    if not isinstance(snapshot, dict):
        return
    cwd = None
    if not isolated:
        from tools.delegate_tool_routes import child_profile_scope
        from tools.terminal_tool import get_session_cwd
        with child_profile_scope(child):
            cwd = _text(get_session_cwd(task_id))
    snapshot["workspace"] = {
        "mode": "isolated" if isolated else "worktree" if worktree else "shared",
        "initial_cwd": cwd,
    }
