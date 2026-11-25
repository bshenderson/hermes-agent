"""Owned native history for archive-opted-in, verified UI chat origins.

Call only after artifact_binding verified the signed final request. A native row
is authority; raw browser history never repairs a missing/corrupt owned store.
Ownership, the busy refusal and the replay all go through the public store handle.
"""
from __future__ import annotations

import asyncio
import hashlib
import json


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


def _owner(row):
    return (row.profile_name, row.user_id, row.chat_id)


def _claim(store, sid, origin, profile, history, enabled, explicit_header):
    """Create the owned row, confirm the existing owner, or answer False.

    A session id is derived from ``[profile, subject, chat]``, so an existing row must already
    carry this owner; anything else is an impostor or a collision and refuses. Creation is one
    store write whose upsert never rewrites an earlier writer's columns, and the owner is
    re-read afterwards, so a row that appeared between the read and the write still refuses.
    """
    from persistence import RetiredSessionError
    from persistence.conversation.sessions import NewSession, SessionId

    expected = (profile, origin['subject_id'], origin['chat_id'])
    row = store.sessions.get(SessionId(sid))
    if row is None and not enabled:
        return False  # Accepted legacy behavior; never claim from metadata alone.
    if explicit_header:
        raise ArchiveChatError('Authenticated archive chats use their own native history', 400)
    if row is not None:
        if _owner(row) != expected:
            raise ArchiveChatError('Authenticated archive chat owner mismatch')
        return True
    if history:
        raise ArchiveChatError('Archive chat requires a fresh chat with native originals; import retained separately')
    try:
        store.sessions.create(NewSession(
            SessionId(sid), 'api_server',
            profile_name=expected[0], user_id=expected[1], chat_id=expected[2],
        ))
    except RetiredSessionError as error:
        raise ArchiveChatError('Authenticated archive chat session id is retired') from error
    if _owner(store.sessions.get(SessionId(sid))) != expected:
        raise ArchiveChatError('Authenticated archive chat owner mismatch')
    return True


def _validate_owner(store, sid, origin, profile):
    from persistence.conversation.sessions import SessionId

    row = store.sessions.get(SessionId(sid))
    if row is None or _owner(row) != (profile, origin['subject_id'], origin['chat_id']):
        raise ArchiveChatError('Authenticated archive chat continuation owner mismatch')
    if store.coordination.active_turn_lease_holder(session_id=SessionId(sid)):
        raise ArchiveChatError('Authenticated archive chat is busy')


def _history(store, sid):
    from persistence.conversation.messages import ConversationRequest

    return [message.as_openai_message()
            for message in store.messages.read_conversation(sid, ConversationRequest(repair_alternation=True))]


async def resolve_owned_history(adapter, origin, profile, client_history, explicit_header):
    """Return (native id, native history), or None for an unclaimed disabled chat.

    Existing ownership survives publication opt-out. Unknown store state refuses
    rather than treating an unavailable owned chat as an ordinary raw replay.
    The ordinary core still owns turn leases, compaction and all model routing.
    """
    from agent.conversation_compression import _archive_receipts_enabled


    sid = owned_session_id(origin, profile)
    try:
        store = adapter._store_beside(await adapter._ensure_store_async())
        if store is None:
            raise ArchiveChatError('Authenticated archive chat store unavailable', 503)
        claimed = await asyncio.to_thread(_claim, store, sid, origin, profile, client_history,
                                          _archive_receipts_enabled(), explicit_header)
        if not claimed:
            return None
        # The generic API resume helpers intentionally swallow lookup errors.
        # An owned archive chat must never replay a stale parent on that path.
        tip = await asyncio.to_thread(store.lineage.tip, sid)
        if not isinstance(tip, str) or not tip:
            raise ArchiveChatError('Authenticated archive chat lineage unavailable', 503)
        await asyncio.to_thread(_validate_owner, store, tip, origin, profile)
        history = await asyncio.to_thread(_history, store, tip)
        if not history and client_history:
            raise ArchiveChatError('Authenticated archive chat has no native originals; start a fresh chat')
        return tip, history
    except ArchiveChatError:
        raise
    except Exception as error:
        raise ArchiveChatError('Authenticated archive chat store unavailable', 503) from error
