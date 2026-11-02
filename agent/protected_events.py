"""Operator event custody. Only the trusted SessionDB journal produces streams.

This is not tool authorization or a sandbox against code/DB writers. Wire metadata,
roles, model prose and historical lexical records never establish issuance.
"""
from typing import NamedTuple
import json

from agent.compression_budget import CompressionBudgetError

FIELD = '_protected_event_stream'
START = '<!-- HERMES_OPERATOR_EVENTS_V1 -->'
END = '<!-- END_HERMES_OPERATOR_EVENTS_V1 -->'
KINDS = frozenset({'hold', 'unknown_outcome', 'constraint', 'commitment', 'correction'})
REQUEST_FIELDS = frozenset({'version', 'session_id', 'submission_id', 'kind', 'text', 'corrects'})
EVENT_FIELDS = REQUEST_FIELDS | {'issuer', 'order', 'source_uid'}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def signing_bytes(body):
    return b'HERMES_OPERATOR_PROTECTED_EVENT_V1\n' + canonical(body).encode('ascii')


def validate_request(body, session_id):
    if not isinstance(body, dict) or set(body) != REQUEST_FIELDS:
        raise ValueError('invalid protected event fields')
    if type(body['version']) is not int or body['version'] != 1:
        raise ValueError('unsupported protected event version')
    if body['session_id'] != session_id:
        raise ValueError('protected event session mismatch')
    for field in ('session_id', 'submission_id'):
        value = body[field]
        if not isinstance(value, str) or not value or len(value) > 128 or any(ord(c) < 32 for c in value):
            raise ValueError('invalid protected event identity')
    if body['kind'] not in KINDS:
        raise ValueError('invalid protected event kind')
    if not isinstance(body['text'], str) or not body['text'] or len(body['text'].encode('utf-8')) > 8192:
        raise ValueError('invalid protected event text size')
    refs = body['corrects']
    if not isinstance(refs, list) or len(refs) > 64 or any(not isinstance(r, str) or not r for r in refs):
        raise ValueError('invalid protected event corrections')
    if len(set(refs)) != len(refs) or bool(refs) != (body['kind'] == 'correction'):
        raise ValueError('correction must reference preceding events; other kinds cannot correct')
    # Exact custody must never bypass the normal secret-redaction boundary.
    from agent.context_compressor import _redact_compaction_text
    if _redact_compaction_text(body['text']) != body['text']:
        raise ValueError('protected text requires redaction; do not submit secrets')
    canonical(body)  # Refuse surrogates/unsupported JSON rather than normalizing.


class EventStream(NamedTuple):
    session_id: str
    records: tuple[str, ...]


def validate_stream(stream, session_id=None):
    try:
        if session_id and stream.session_id != session_id:
            raise ValueError('protected event session mismatch')
        events, seen = [], set()
        for position, raw in enumerate(stream.records, 1):
            event = json.loads(raw)
            if not isinstance(event, dict) or set(event) != EVENT_FIELDS:
                raise ValueError('invalid protected event record')
            validate_request({k: event[k] for k in REQUEST_FIELDS}, stream.session_id)
            if type(event['order']) is not int or event['order'] != position:
                raise ValueError('invalid protected event order')
            if not isinstance(event['issuer'], str) or not event['issuer'].startswith('ed25519:'):
                raise ValueError('invalid protected event issuer')
            uid = event['source_uid']
            if not isinstance(uid, str) or not uid or uid in seen or not set(event['corrects']) <= seen:
                raise ValueError('invalid protected event source or correction')
            if canonical(event) != raw:
                raise ValueError('noncanonical protected event record')
            events.append(event)
            seen.add(uid)
        return events
    except (ValueError, TypeError, KeyError, UnicodeError) as error:
        raise CompressionBudgetError(str(error)) from error


def collect_events(compressor, messages, *, session_id=None):
    events = None
    for row in messages:
        stream = row.get(FIELD) if isinstance(row, dict) else None
        # JSON metadata is not an internal journal capability.
        if not isinstance(stream, EventStream):
            continue
        current = validate_stream(stream, session_id or getattr(compressor, '_session_id', None))
        if events is not None and current != events:
            raise CompressionBudgetError('conflicting protected event streams')
        events = current
    return events or []


def encode_events(events):
    if not events:
        return ''
    # Escape framing characters even inside protected text. Decode restores exact text.
    payload = canonical(events).replace('<', '\\u003c').replace('>', '\\u003e')
    return '\n' + START + '\n' + payload + '\n' + END


def decode_events(text):
    if not isinstance(text, str) or START not in text:
        return []
    if text.count(START) != 1 or text.count(END) != 1:
        raise CompressionBudgetError('ambiguous protected event framing')
    try:
        events = json.loads(text.split(START, 1)[1].split(END, 1)[0])
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            raise ValueError('invalid protected event frame shape')
        return events
    except ValueError as error:
        raise CompressionBudgetError('invalid protected event framing') from error


def strip_model_events(text):
    while START in text:
        before, after = text.split(START, 1)
        if END not in after:
            raise CompressionBudgetError('incomplete model protected event block')
        text = before + after.split(END, 1)[1]
    if END in text:
        raise CompressionBudgetError('unmatched model protected event block')
    return text


def carry_stream(original, candidate):
    streams = [row[FIELD] for row in original if isinstance(row, dict) and isinstance(row.get(FIELD), EventStream)]
    if streams and candidate:
        # Internal custody only; provider request assembly strips FIELD.
        candidate[0] = dict(candidate[0], **{FIELD: streams[0]})


class SourceRecords(list):
    """Attempt-local mandatory allocation, not mutable compressor/global state."""
    def __init__(self, records, events):
        super().__init__(records)
        self.events = events
