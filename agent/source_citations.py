"""Owner-side decoded-source citation validation over a durable SessionDB snapshot.

Only the authenticated session owner may supply db/session_id/profile. This module
is not an authorization endpoint or a sandbox against code or database writers.
No caller-supplied history, summary flag, source text or decoded ledger is accepted.
"""
import hashlib
import json

from agent.compression_preservation import CONTRACT, START, END, decode_records, encode_records
from agent.context_compressor import _handoff_only_content, _redact_compaction_text

MAX_ROWS = 4096
MAX_BYTES = 8 * 1024 * 1024
MAX_CITATIONS = 32
MAX_QUOTE_BYTES = 8192
MAX_MATCHES = 128


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def _snapshot(db, session_id, profile):
    if any(not isinstance(value, str) or not value or len(value) > 128
           for value in (session_id, profile)):
        raise ValueError('citation owner identity required')
    rows, size = [], 0
    # One SELECT owns the SQLite read snapshot, including profile and archive rows.
    # No writes, load repairs, metadata adoption or cross-session ancestry expansion.
    with db._read_ctx() as conn:
        cursor = conn.execute(
            'SELECT s.profile_name, m.id, m.message_uid, m.role, substr(m.content, 1, ?) AS content, '
            'length(CAST(m.content AS BLOB)) AS source_bytes, '
            'm.active, m._compressed_summary FROM sessions s JOIN messages m '
            'ON m.session_id = s.id WHERE s.id = ? ORDER BY m.id LIMIT ?',
            (MAX_BYTES + 1, session_id, MAX_ROWS + 1))
        try:
            for row in cursor:
                if row['profile_name'] != profile:
                    raise ValueError('citation owner mismatch')
                content = row['content']
                if not isinstance(content, str) or not row['message_uid']:
                    raise ValueError('unsupported citation source identity')
                size += row['source_bytes']
                if len(rows) == MAX_ROWS or size > MAX_BYTES:
                    raise ValueError('citation source budget exceeded')
                rows.append(dict(row))
        finally:
            cursor.close()
    if not rows:
        raise ValueError('citation owner source unavailable')
    return rows, _digest({'session_id': session_id, 'profile': profile, 'rows': rows})


def checkpoint_identity(db, *, session_id, profile):
    """Bind a request to the current durable snapshot; owner supplies scope."""
    return _snapshot(db, session_id, profile)[1]


def validate_citations(db, *, session_id, profile, checkpoint, citations):
    """Return grounded spans with all original origins, or refuse the whole batch.

    Validity is at this read snapshot, not perpetual validity or execution proof.
    Consumers must retain the returned checkpoint and revalidate after mutations.
    """
    if not isinstance(citations, list) or not 1 <= len(citations) <= MAX_CITATIONS:
        raise ValueError('bounded nonempty citation list required')
    for quote in citations:
        if not isinstance(quote, str) or not quote or len(quote.encode('utf-8')) > MAX_QUOTE_BYTES:
            raise ValueError('invalid citation quotation')
        if _redact_compaction_text(quote) != quote:
            raise ValueError('citation quotation requires redaction')
    rows, current = _snapshot(db, session_id, profile)
    if not isinstance(checkpoint, str) or current != checkpoint:
        raise ValueError('citation checkpoint stale or mismatched')
    originals = [row for row in rows if not row['_compressed_summary']
                 and row['role'] in ('user', 'assistant', 'tool')
                 and not any(marker in row['content'] for marker in (CONTRACT, START, END))]
    canonical = []
    for row in rows:
        if not row['active'] or not row['_compressed_summary']:
            continue
        frame = _handoff_only_content(row['content'])
        if not isinstance(frame, str):
            raise ValueError('unsupported citation checkpoint content')
        records = decode_records(frame)  # documented single JSON decode + actual post-decode redaction
        if encode_records(records) not in frame:
            raise ValueError('citation checkpoint is not canonical sanitized source')
        for index, record in enumerate(records):
            origins = [original for original in originals
                       if record in _redact_compaction_text(original['content'])]
            if not record or not origins:
                raise ValueError('citation checkpoint lacks original source custody')
            if len(origins) > MAX_MATCHES or len(canonical) >= MAX_MATCHES:
                raise ValueError('citation match budget exceeded')
            canonical.append((row, index, record, origins))
    if not canonical:
        raise ValueError('trusted citation checkpoint unavailable')
    grounded = []
    matches = 0
    for quote in citations:
        sources = []
        for carrier, index, record, origins in canonical:
            start = 0
            while (offset := record.find(quote, start)) >= 0:
                matches += len(origins)
                if matches > MAX_MATCHES:
                    raise ValueError('citation match budget exceeded')
                sources.append({'carrier_row_id': carrier['id'], 'carrier_uid': carrier['message_uid'],
                    'record_index': index, 'record_sha256': _digest(record),
                    'start': offset, 'end': offset + len(quote),
                    'original_rows': [{'row_id': row['id'], 'message_uid': row['message_uid']}
                                      for row in origins]})
                start = offset + 1
        if not sources:
            raise ValueError('citation absent from trusted decoded source')
        grounded.append({'quote': quote, 'sources': sources})
    return {'version': 1, 'session_id': session_id, 'profile': profile,
            'checkpoint': current, 'citations': grounded}
