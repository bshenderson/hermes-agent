"""Bounded tool-result projections for public API-server progress events.

The full tool result remains in the agent transcript.  Public progress channels
only get small, explicitly whitelisted projections needed by UI/broker bridges.
"""

from __future__ import annotations

import json
from typing import Any, Optional

_MEDIA_RESULT_TOOL_NAMES = frozenset({"media_match_workflow", "media_generate", "media_artifact_get"})
_MEDIA_RESULT_MAX_BYTES = 16_384
_MEDIA_RESULT_MAX_DEPTH = 5
_MEDIA_RESULT_MAX_LIST_ITEMS = 5
_MEDIA_RESULT_MAX_STRING_CHARS = 2_048

_MEDIA_ROOT_KEYS = frozenset({
    "success",
    "error",
    "prompt_id",
    "status_url",
    "submission_recovered",
    "submission",
    "status",
    "job",
    "media_artifacts",
    "artifacts",
    "artifact_id",
    "filename",
    "download_url",
    "preview_url",
    "inline_markdown",
    "auto_submitted_media_generate",
    "media_generate_result",
    "owui_media",
})
_MEDIA_NESTED_KEYS = _MEDIA_ROOT_KEYS | frozenset({
    "url",
    "content_type",
    "mime_type",
    "type",
    "width",
    "height",
    "bytes",
    "size",
    "state",
    "message",
})
_MEDIA_COMPACT_KEYS = (
    "success",
    "error",
    "prompt_id",
    "status_url",
    "media_artifacts",
    "artifacts",
    "media_generate_result",
    "submission",
    "status",
    "job",
    "owui_media",
)


def bounded_tool_result_projection(tool_name: Any, result: Any) -> Optional[dict[str, Any]]:
    """Return a small structured result payload for public tool-completion events.

    Media completion events need enough metadata for brokers/UI bridges to poll
    or render artifacts before the assistant's final prose arrives.  This is a
    transport projection only: it does not upload files, write Open WebUI rows,
    route work, or own artifact lifecycle.
    """

    name = str(tool_name or "")
    if name not in _MEDIA_RESULT_TOOL_NAMES:
        return None
    payload = _parse_payload(result)
    if not isinstance(payload, dict):
        return {"success": False, "parse_error": "non_object_media_tool_result"}

    projected = _copy_media_value(payload, context="root")
    if not isinstance(projected, dict):
        return None
    if _encoded_size(projected) <= _MEDIA_RESULT_MAX_BYTES:
        return projected

    compact = {key: projected[key] for key in _MEDIA_COMPACT_KEYS if key in projected}
    compact = _shrink_to_budget(compact)
    if isinstance(compact, dict):
        return compact
    return {"success": bool(projected.get("success")), "truncated": True}


def _parse_payload(result: Any) -> Any:
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return {"success": False, "parse_error": "non_json_media_tool_result"}
    return result


def _copy_media_value(value: Any, *, context: str, depth: int = 0) -> Any:
    if depth >= _MEDIA_RESULT_MAX_DEPTH:
        return None
    if isinstance(value, dict):
        allowed = _MEDIA_ROOT_KEYS if context == "root" else _MEDIA_NESTED_KEYS
        out: dict[str, Any] = {}
        for key, child in value.items():
            key_s = str(key)
            if key_s not in allowed:
                continue
            child_context = _child_context(key_s, context)
            copied = _copy_media_value(child, context=child_context, depth=depth + 1)
            if copied is not None or child is None:
                out[key_s] = copied
        return out
    if isinstance(value, list):
        return [_copy_media_value(item, context=context, depth=depth + 1) for item in value[:_MEDIA_RESULT_MAX_LIST_ITEMS]]
    if isinstance(value, str):
        return value[:_MEDIA_RESULT_MAX_STRING_CHARS]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:_MEDIA_RESULT_MAX_STRING_CHARS]


def _child_context(key: str, parent: str) -> str:
    if key in {"media_artifacts", "artifacts", "owui_media"}:
        return "artifact"
    if key in {"submission", "status", "job"}:
        return "job"
    return parent


def _encoded_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _shrink_to_budget(value: dict[str, Any]) -> dict[str, Any]:
    candidate = value
    while _encoded_size(candidate) > _MEDIA_RESULT_MAX_BYTES:
        changed = False
        for key in ("media_artifacts", "artifacts", "owui_media"):
            items = candidate.get(key)
            if isinstance(items, list) and len(items) > 1:
                candidate[key] = items[:1]
                changed = True
        if changed:
            continue
        for container in (candidate.get("status"), candidate.get("submission"), candidate.get("job")):
            if isinstance(container, dict):
                for key in ("media_artifacts", "artifacts", "owui_media"):
                    items = container.get(key)
                    if isinstance(items, list) and len(items) > 1:
                        container[key] = items[:1]
                        changed = True
        if changed:
            continue
        for key in ("inline_markdown", "download_url", "preview_url", "url", "status_url", "error"):
            if _truncate_strings(candidate, key, 512):
                changed = True
        if not changed:
            candidate = {key: candidate[key] for key in ("success", "prompt_id", "status_url", "error") if key in candidate}
            candidate["truncated"] = True
            break
    return candidate


def _truncate_strings(value: Any, key: str, limit: int) -> bool:
    changed = False
    if isinstance(value, dict):
        current = value.get(key)
        if isinstance(current, str) and len(current) > limit:
            value[key] = current[:limit]
            changed = True
        for child in value.values():
            if _truncate_strings(child, key, limit):
                changed = True
    elif isinstance(value, list):
        for item in value:
            if _truncate_strings(item, key, limit):
                changed = True
    return changed
