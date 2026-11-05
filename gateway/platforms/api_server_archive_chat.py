"""Owned native history for archive-opted-in, verified UI chat origins.

Call only after artifact_binding verified the signed final request. A native row
is authority; raw browser history never repairs a missing/corrupt owned store.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time


class ArchiveChatError(ValueError):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def owned_session_id(origin, profile):
    subject, chat = origin.get('subject_id'), origin.get('chat_id')
    if any(not isinstance(v, str) or not 0 < len(v) <= 256 for v in (subject, chat, profile)):
        raise ArchiveChatError('Invalid authenticated archive chat identity', 400)
    value = json.dumps([profile, subject, chat], separators=(',', ':'), ensure_ascii=True)
    return 'owui-archive-' + hashlib.sha256(value.encode()).hexdigest()[:32]


def _claim(db, sid, origin, profile, history, enabled, explicit_header):
    def transaction(conn):
        row = conn.execute('SELECT profile_name,user_id,chat_id FROM sessions WHERE id=?', (sid,)).fetchone()
        if row is None and not enabled:
            return False  # Accepted legacy behavior; never claim from metadata alone.
        if explicit_header:
            raise ArchiveChatError('Authenticated archive chats use their own native history', 400)
        expected = (profile, origin['subject_id'], origin['chat_id'])
        if row is not None:
            if tuple(row) != expected:
                raise ArchiveChatError('Authenticated archive chat owner mismatch')
            return True
        if history:
            raise ArchiveChatError('Archive chat requires a fresh chat with native originals; import retained separately')
        conn.execute('INSERT INTO sessions(id,source,profile_name,user_id,chat_id,started_at) '
                     'VALUES(?,?,?,?,?,?)', (sid, 'api_server', *expected, time.time()))
        return True
    return db._execute_write(transaction)


def _validate_owner(db, sid, origin, profile):
    with db._read_ctx() as conn:
        row = conn.execute('SELECT profile_name,user_id,chat_id FROM sessions WHERE id=?', (sid,)).fetchone()
        if row is None or tuple(row) != (profile, origin['subject_id'], origin['chat_id']):
            raise ArchiveChatError('Authenticated archive chat continuation owner mismatch')
        lease_key = db._session_turn_lease_key_on_conn(conn, sid)
        active = conn.execute('SELECT COUNT(*) FROM session_turn_leases '
                              'WHERE conversation_id=? AND expires_at>?', (lease_key, time.time())).fetchone()[0]
        if active:
            raise ArchiveChatError('Authenticated archive chat is busy')


async def resolve_owned_history(adapter, origin, profile, client_history, explicit_header):
    """Return (native id, native history), or None for an unclaimed disabled chat.

    Existing ownership survives publication opt-out. Unknown store state refuses
    rather than treating an unavailable owned chat as an ordinary raw replay.
    The ordinary core still owns turn leases, compaction and all model routing.
    """
    from agent.conversation_compression import _archive_receipts_enabled


    sid = owned_session_id(origin, profile)
    try:
        db = await adapter._ensure_session_db_async()
        if db is None:
            raise ArchiveChatError('Authenticated archive chat store unavailable', 503)
        claimed = await asyncio.to_thread(_claim, db, sid, origin, profile, client_history,
                                          _archive_receipts_enabled(), explicit_header)
        if not claimed:
            return None
        # The generic API resume helpers intentionally swallow lookup errors.
        # An owned archive chat must never replay a stale parent on that path.
        tip = await asyncio.to_thread(db.get_compression_tip, sid)
        if not isinstance(tip, str) or not tip:
            raise ArchiveChatError('Authenticated archive chat lineage unavailable', 503)
        await asyncio.to_thread(_validate_owner, db, tip, origin, profile)
        history = await asyncio.to_thread(db.get_messages_as_conversation, tip)
        if not history and client_history:
            raise ArchiveChatError('Authenticated archive chat has no native originals; start a fresh chat')
        return tip, history
    except ArchiveChatError:
        raise
    except Exception as error:
        raise ArchiveChatError('Authenticated archive chat store unavailable', 503) from error
