"""Compatibility surface for the operator protected-event journal.

The journal lives in the store layer (``persistence.sqlite.conversation._store.protected_events``),
so its write transaction and its transcript carrier row are the store's own. This module only
re-exports it for callers written against the pre-migration name; new code imports the store
module.
"""
from persistence.sqlite.conversation._store.protected_events import (
    append_verified_event, refuse_history_mutation, restore_stream,
)

__all__ = ['append_verified_event', 'refuse_history_mutation', 'restore_stream']
