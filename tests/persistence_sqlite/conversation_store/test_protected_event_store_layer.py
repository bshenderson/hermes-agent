"""The operator journal belongs to the store layer, not to its callers.

The pre-migration wiring had the root-level ``hermes_state_protected_events`` module reach back
into ``persistence.sqlite.conversation._store.messages`` for the row, counter and guard helpers,
so a transport-layer module owned a transcript write it could not see end to end. The journal
now lives beside those helpers; these checks keep it there.
"""
from __future__ import annotations

import ast
from pathlib import Path

from persistence.sqlite.conversation._store import protected_events

REPO_ROOT = Path(__file__).resolve().parents[3]
PRIVATE_HELPERS = {"_check_transcript_write_guards", "_insert_message_rows", "_bump_session_counters"}
STORE_PACKAGE = REPO_ROOT / "persistence"


def _store_helper_imports(path: Path) -> set[str]:
    """Names *path* imports from the store's private messages module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("conversation._store.messages"):
                found.update(alias.name for alias in node.names)
    return found


def _product_modules() -> list[Path]:
    modules = []
    for root in ("agent", "gateway", "tools", "hermes_cli"):
        modules.extend(path for path in (REPO_ROOT / root).rglob("*.py") if "__pycache__" not in path.parts)
    modules.append(REPO_ROOT / "hermes_state_protected_events.py")
    return modules


def test_no_product_module_owns_the_transcript_helpers_the_journal_needs():
    offenders = []
    for path in _product_modules():
        reached = _store_helper_imports(path) & PRIVATE_HELPERS
        if reached:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {sorted(reached)}")
    assert offenders == [], offenders


def test_journal_module_is_the_store_layer_owner():
    assert protected_events.__name__.endswith("conversation._store.protected_events")
    assert Path(protected_events.__file__).resolve().is_relative_to(STORE_PACKAGE)


def test_compatibility_surface_re_exports_the_same_objects():
    import hermes_state_protected_events as legacy

    assert legacy.append_verified_event is protected_events.append_verified_event
    assert legacy.refuse_history_mutation is protected_events.refuse_history_mutation
    assert legacy.restore_stream is protected_events.restore_stream
