"""Exact, session-scoped tool archive references for opt-in compaction.

No source writes, authorization inference, or cross-session traversal. The owner
supplies SessionDB/session_id. References attest content at each validation, not
permanent availability. Missing/changed originals refuse replacement.
"""
import hashlib
import json

from agent.compression_budget import CompressionBudgetError

PREFIX = 'HERMES_ARCHIVED_TOOL_RANGE_V1 '
MAX_ROWS = 4096
MAX_BYTES = 8 * 1024 * 1024


def is_reference(value):
    return isinstance(value, str) and value.startswith(PREFIX)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def _snapshot(db, session_id, through=None):
    if db is None or not isinstance(session_id, str) or not session_id or len(session_id) > 128:
        raise CompressionBudgetError('durable tool archive owner unavailable')
    try:
        with db._read_ctx() as conn:
            conn.execute('SAVEPOINT compression_tool_archive')
            try:
                owner = conn.execute('SELECT profile_name FROM sessions WHERE id=?', (session_id,)).fetchone()
                if owner is None:
                    raise CompressionBudgetError('durable tool archive session unavailable')
                cursor = conn.execute(
                    'SELECT id, message_uid, tool_call_id, substr(content,1,?) AS content, '
                    'length(CAST(content AS BLOB)) AS source_bytes FROM messages '
                    'WHERE session_id=? AND role=\'tool\' AND COALESCE(_compressed_summary,0)=0 '
                    'AND id<=? ORDER BY id LIMIT ?',
                    (MAX_BYTES + 1, session_id, through if through is not None else 9223372036854775807, MAX_ROWS + 1))
                rows, size = [], 0
                try:
                    for row in cursor:
                        size += row['source_bytes'] or 0
                        if len(rows) >= MAX_ROWS or size > MAX_BYTES:
                            raise CompressionBudgetError('durable tool archive budget exceeded')
                        if not isinstance(row['content'], str) or not row['message_uid']:
                            raise CompressionBudgetError('durable tool archive identity unavailable')
                        rows.append({key: row[key] for key in ('id', 'message_uid', 'tool_call_id', 'content')})
                finally:
                    cursor.close()
            finally:
                conn.execute('RELEASE compression_tool_archive')
        digest = hashlib.sha256(_canonical({'session_id': session_id, 'profile': owner['profile_name'], 'rows': rows}).encode('ascii')).hexdigest()
        return rows, digest
    except CompressionBudgetError:
        raise
    except Exception as error:
        raise CompressionBudgetError('durable tool archive unavailable') from error


def resolve_reference(db, session_id, reference):
    try:
        value = json.loads(reference.removeprefix(PREFIX))
        if (not is_reference(reference) or not isinstance(value, dict)
                or set(value) != {'session_id', 'first', 'through', 'sha256'}
                or value['session_id'] != session_id or type(value['through']) is not int or value['through'] <= 0
                or type(value['first']) is not int or value['first'] <= 0
                or not isinstance(value['sha256'], str) or len(value['sha256']) != 64
                or reference != PREFIX + _canonical(value)):
            raise ValueError('invalid archive identity')
        rows, digest = _snapshot(db, session_id, value['through'])
        if not rows or rows[0]['id'] != value['first'] or rows[-1]['id'] != value['through'] or digest != value['sha256']:
            raise ValueError('tool archive changed or missing')
        return rows
    except CompressionBudgetError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise CompressionBudgetError('durable tool archive reference invalid or changed') from error


def validate_references(compressor, records):
    for record in records:
        if is_reference(record):
            resolve_reference(getattr(compressor, '_session_db', None), getattr(compressor, '_session_id', ''), record)


def archive_current_tools(compressor, messages, previous):
    """Return one cumulative reference only after every current tool body is durable."""
    validate_references(compressor, previous)
    tools = [row for row in messages if row.get('role') == 'tool' and not compressor._is_context_summary_message(row)]
    if not tools:
        return [record for record in previous if is_reference(record)]
    from agent.context_compressor import _redact_compaction_text
    db, sid = getattr(compressor, '_session_db', None), getattr(compressor, '_session_id', '')
    rows, digest = _snapshot(db, sid)
    for message in tools:
        text = message.get('content')
        if not isinstance(text, str) or not any(
                _redact_compaction_text(row['content']) == _redact_compaction_text(text)
                and row['tool_call_id'] == message.get('tool_call_id') for row in rows):
            raise CompressionBudgetError('tool content lacks exact durable source custody')
    return [PREFIX + _canonical({'session_id': sid, 'first': rows[0]['id'], 'through': rows[-1]['id'], 'sha256': digest})]
