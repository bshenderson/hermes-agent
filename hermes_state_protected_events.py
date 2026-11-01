"""Transactional journal for signature-verified operator ingress.

Not a public setter. Caller is the signature-verifying API transport; ordinary
message inserts, metadata, tools and imports never write this journal.
"""
import json

from agent.message_metadata import mint_uid
from agent.protected_events import EventStream, FIELD, canonical, validate_request, validate_stream


def append_verified_event(db, body, issuer):
    session_id = body['session_id']
    validate_request(body, session_id)

    def write(conn):
        db._check_transcript_write_guards(conn, session_id, None,
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
        inserted, calls = db._insert_message_rows(conn, session_id, [message])
        db._bump_session_counters(conn, session_id, inserted, calls, unit=False)
        conn.execute('INSERT INTO operator_protected_events (session_id, event_order, issuer, submission_id, event) '
                     'VALUES (?, ?, ?, ?, ?)',
                     (session_id, event['order'], issuer, body['submission_id'], canonical(event)))
        return event, True
    return db._execute_write(write)


def refuse_history_mutation(conn, session_id):
    """Unsupported rewrites cannot detach durable custody from a warm transcript.

    Check inside the writer transaction, before row or caller-dictionary mutation.
    In-place compression has its separate validated custody/commit path.
    """
    if conn.execute('SELECT 1 FROM operator_protected_events WHERE session_id = ? LIMIT 1',
                    (session_id,)).fetchone():
        raise ValueError('protected event history cannot be rewound, replaced or cleared')


def restore_stream(db, session_id, messages):
    rows = db._read_all('SELECT event FROM operator_protected_events WHERE session_id = ? ORDER BY event_order',
                        (session_id,))
    if rows:
        if not messages:
            # Refuse an erased transcript rather than silently making its journal disappear.
            raise ValueError('protected event journal has no transcript carrier')
        messages[0][FIELD] = EventStream(session_id, tuple(row['event'] for row in rows))
    return messages
