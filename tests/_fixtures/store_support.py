"""Fixture steps on the public persistence interface for the fork's compression/citation tests.

The fork's pre-merge fixtures were written against ``hermes_state.SessionDB``'s private test surface
(``append_messages_batch``, ``archive_and_compact``, ``get_active_message_watermark``, ``_write_sql``,
``_read_ctx``). Those verbs are deliberately gone from product code: storage authority is the public
persistence handle. These helpers express the same fixture steps through the public repositories so
the assertions they guard stay intact, and give raw-SQL setup/observation its own fixture-owned
connection. Nothing here is a product surface.
"""
from contextlib import contextmanager
import sqlite3
from pathlib import Path

from persistence import MessageId, SessionId
from persistence.conversation.messages import ConversationRequest, HeldCoverage, NewMessage, Watermark
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, profile_scope


def open_store(home):
    """A writable public handle on *home*'s store, opened eagerly so a broken store fails here."""
    return open_persistence(profile_scope(home=Path(home)), eager=True)


def create_session(store, session_id, source='cli', *, profile_name=None, **kwargs):
    return store.sessions.create(NewSession(SessionId(session_id), source, profile_name=profile_name, **kwargs))


def _new_message(role, content, **kwargs):
    """``NewMessage`` with the fork's live-dict spelling of the summary flag accepted."""
    if '_compressed_summary' in kwargs:
        kwargs['compressed_summary'] = kwargs.pop('_compressed_summary')
    return NewMessage(role, content, **kwargs)


def append(store, session_id, role, content, **kwargs):
    return store.messages.append(session_id, _new_message(role, content, **kwargs))


def append_rows(store, session_id, rows):
    return store.messages.append_batch(session_id, tuple(NewMessage.from_live(row) for row in rows))


def watermark(store, session_id):
    """The newest active row id at compression start (``None`` when no active row exists)."""
    return store.messages.active_watermark(session_id).last_active


def active_ids(store, session_id):
    return store.messages.active_message_ids(session_id)


def session_profile(store, session_id):
    session = store.sessions.get(SessionId(session_id))
    return None if session is None else session.profile_name


def compact(store, session_id, rows, *, watermark=None, covered_ids=None, **kwargs):
    """Publish *rows* as the live transcript, archiving the active source in one transaction.

    Returns the new active row count, the shape the fork's ``archive_and_compact`` returned.
    """
    at = None if watermark is None else Watermark(MessageId(int(watermark)))
    if covered_ids is not None:
        kwargs['coverage'] = HeldCoverage(covered_ids=tuple(MessageId(int(i)) for i in covered_ids))
    result = store.messages.compact(session_id, tuple(NewMessage.from_live(row) for row in rows),
                                    watermark=at, **kwargs)
    return result.active_count


def conversation(store, session_id, *, repair_alternation=False):
    """The replay projection of *session_id* as the plain message dicts the old reader returned."""
    return [row.as_openai_message() for row in store.messages.read_conversation(
        session_id, ConversationRequest(repair_alternation=repair_alternation))]


def transcript(store, session_id, *, include_inactive=False):
    """The stored transcript rows: active only, or every row including compaction-archived ones."""
    from persistence.conversation.messages import TranscriptRead, TranscriptView
    view = TranscriptView.ALL if include_inactive else TranscriptView.ACTIVE
    return store.messages.read_transcript(session_id, TranscriptRead(view=view))


@contextmanager
def raw(home):
    """A fixture-owned connection to the same state.db, for setup and observation only."""
    connection = sqlite3.connect(Path(home) / 'state.db', timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def write_sql(home, sql, params=()):
    with raw(home) as connection:
        return connection.execute(sql, params)


def read_one(home, sql, params=()):
    with raw(home) as connection:
        return connection.execute(sql, params).fetchone()


def patch_model_config(store, session_id, patch):
    """Merge a model-config patch onto the session row (usage anchors, lineage markers)."""
    return store.sessions.patch_model_config(SessionId(session_id), patch)


def acquire_lease(store, session_id, holder):
    """Take the conversation's turn lease without waiting; False when another holder has it."""
    return store.coordination.try_acquire_turn_lease(session_id=SessionId(session_id), holder=holder)


def release_lease(store, session_id, holder):
    return store.coordination.release_turn_lease(session_id=SessionId(session_id), holder=holder)


def active_lease_holder(store, session_id):
    return store.coordination.active_turn_lease_holder(session_id=SessionId(session_id))


def publish_child(store, parent_session_id, child_session_id, *, source, messages,
                  profile_name=None, require_compression_lease=True):
    """Publish a compression continuation with its handoff rows, as the compressor does."""
    from persistence.conversation.lineage import CompressionChild
    return store.lineage.publish_compression_child(
        SessionId(parent_session_id),
        CompressionChild(SessionId(child_session_id), source, profile_name=profile_name),
        tuple(NewMessage.from_live(message) for message in messages),
        require_compression_lease=require_compression_lease,
    )


def write_sql_store(store, sql, params=()):
    """Raw-SQL setup on the home *store* was opened from (fixture-owned connection)."""
    return write_sql(store.scope.hermes_home_key, sql, params)


def read_one_store(store, sql, params=()):
    return read_one(store.scope.hermes_home_key, sql, params)
