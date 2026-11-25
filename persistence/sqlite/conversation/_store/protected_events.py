"""Transactional journal for signature-verified operator ingress, owned by the store layer.

Not a public setter. The caller is the signature-verifying API transport; ordinary message
inserts, metadata, tools and imports never write this journal. The journal, its transcript
carrier row and the session counters commit in one write transaction, so a verified event is
durable exactly when its user row is.
"""

from __future__ import annotations

import json

from persistence.sqlite._holder import StoreHolder


def append_verified_event(holder: StoreHolder, body, issuer: str):
    """Journal one verified event and its user row in the caller's store transaction.

    Returns ``(event, inserted)``. An identical repeat of ``submission_id`` answers the original
    event without another transcript write; the same identity with different content refuses the
    whole transaction.
    """
    from agent.message_metadata import mint_uid
    from agent.protected_events import EventStream, canonical, validate_request, validate_stream
    from persistence.sqlite.conversation._store.messages import (
        _bump_session_counters, _check_transcript_write_guards, _insert_message_rows,
    )
    session_id = body['session_id']
    validate_request(body, session_id)

    def write(conn):
        _check_transcript_write_guards(holder, conn, session_id, None,
            reject_active_turn_lease=True, reject_active_compression_lock=True)
        session = conn.execute('SELECT source FROM sessions WHERE id = ?', (session_id,)).fetchone()
        if session is None or session['source'] != 'api_server':
            raise ValueError('protected event target must be an existing native API session')
        previous = conn.execute(
            'SELECT event FROM operator_protected_events WHERE session_id = ? AND issuer = ? AND submission_id = ?',
            (session_id, issuer, body['submission_id'])).fetchone()
        if previous:
            event = json.loads(previous['event'])
            if {k: event[k] for k in body} != body:
                raise ValueError('protected submission identity conflict')
            return event, False
        rows = conn.execute('SELECT event FROM operator_protected_events WHERE session_id = ? ORDER BY event_order',
                            (session_id,)).fetchall()
        existing = validate_stream(EventStream(session_id, tuple(row['event'] for row in rows)))
        if not set(body['corrects']) <= {e['source_uid'] for e in existing}:
            raise ValueError('correction must reference preceding same-session events')
        event = dict(body, issuer=issuer, order=len(existing) + 1, source_uid=mint_uid())
        message = {'role': 'user', 'content': body['text'], 'message_uid': event['source_uid']}
        inserted, calls = _insert_message_rows(holder, conn, session_id, [message])
        _bump_session_counters(conn, session_id, inserted, calls, unit=False)
        conn.execute('INSERT INTO operator_protected_events (session_id, event_order, issuer, submission_id, event) '
                     'VALUES (?, ?, ?, ?, ?)',
                     (session_id, event['order'], issuer, body['submission_id'], canonical(event)))
        return event, True
    return holder._execute_write(write)


def refuse_history_mutation(conn, session_id: str) -> None:
    """Unsupported rewrites cannot detach durable custody from a warm transcript.

    Check inside the writer transaction, before row or caller-dictionary mutation.
    In-place compression has its separate validated custody/commit path.
    """
    if conn.execute('SELECT 1 FROM operator_protected_events WHERE session_id = ? LIMIT 1',
                    (session_id,)).fetchone():
        raise ValueError('protected event history cannot be rewound, replaced or cleared')


def restore_stream(holder: StoreHolder, session_id: str, messages):
    """Attach the session's journaled stream to the caller's replay projection."""
    from agent.protected_events import FIELD, EventStream
    with holder._read_ctx() as conn:
        rows = conn.execute('SELECT event FROM operator_protected_events WHERE session_id = ? ORDER BY event_order',
                            (session_id,)).fetchall()
    if rows:
        if not messages:
            # Refuse an erased transcript rather than silently making its journal disappear.
            raise ValueError('protected event journal has no transcript carrier')
        messages[0][FIELD] = EventStream(session_id, tuple(row['event'] for row in rows))
    return messages
