"""Opt-in, source-bound recurring single-line user/assistant quotations.

References attest source data, never model execution or operator authority.
Original rows are read through the already-bound owner SessionDB; no writes.
"""
import hashlib
import json

from agent.compression_budget import CompressionBudgetError

PREFIX = 'HERMES_SELECTED_SOURCE_V1 '
MAX_ROWS = 4096
MAX_BYTES = 8 * 1024 * 1024
MAX_ORIGINS = 32
MAX_QUOTES = 128


def _fail(message):
    raise CompressionBudgetError('selected source custody: '+message)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def is_reference(value):
    return isinstance(value, str) and value.startswith('HERMES_SELECTED_SOURCE_')


def enabled():
    from agent.auxiliary_client import _get_auxiliary_task_config
    value = _get_auxiliary_task_config('compression').get('selected_source_custody', False)
    if type(value) is not bool:
        _fail('opt-in must be boolean')
    return value


def _snapshot(compressor):
    db = getattr(compressor, '_session_db', None)
    sid = getattr(compressor, '_session_id', None)
    if db is None or not isinstance(sid, str) or not sid or len(sid)>128:
        _fail('owning durable store unavailable')
    try:
        with db._read_ctx() as conn:
            conn.execute('SAVEPOINT selected_source_custody')
            try:
                owner = conn.execute('SELECT profile_name FROM sessions WHERE id=?', (sid,)).fetchone()
                if owner is None:
                    _fail('owning session unavailable')
                cursor = conn.execute(
                    'SELECT id,message_uid,role,CASE WHEN length(CAST(content AS BLOB))>? THEN NULL ELSE content END AS content, '
                    'length(CAST(content AS BLOB)) AS source_bytes FROM messages '
                    "WHERE session_id=? AND role IN ('user','assistant') "
                    'AND COALESCE(_compressed_summary,0)=0 ORDER BY id LIMIT ?',
                    (MAX_BYTES, sid, MAX_ROWS+1))
                rows, size = [], 0
                try:
                    for row in cursor:
                        size += row['source_bytes'] or 0
                        if len(rows)>=MAX_ROWS or size>MAX_BYTES:
                            _fail('original snapshot budget exceeded')
                        if not isinstance(row['content'], str) or not isinstance(row['message_uid'], str) or not row['message_uid']:
                            _fail('original text identity unavailable')
                        rows.append(dict(row))
                finally:
                    cursor.close()
                profile = owner['profile_name']
            finally:
                conn.execute('RELEASE selected_source_custody')
        return sid, profile, rows
    except CompressionBudgetError:
        raise
    except Exception as error:
        raise CompressionBudgetError('selected source custody: owning durable store unavailable') from error


def preflight(compressor):
    _snapshot(compressor)


def _serialized(compressor, row):
    return '\n\n'.join(compressor._serialize_records_for_summary(
        [{'role':row['role'], 'content':row['content']}], lossless=True)).encode('utf-8')


def _quote(quote):
    if (not isinstance(quote, str) or not quote.strip() or '\n' in quote or '\r' in quote
            or len(quote.encode('utf-8'))>8192 or is_reference(quote)):
        _fail('unsupported single-line text quotation')
    from agent.context_compressor import _redact_compaction_text
    if _redact_compaction_text(quote) != quote:
        _fail('quotation is not sanitized')


def bind_quotes(compressor, quotes, *, _project_serializer_role=True):
    quotes = list(dict.fromkeys(quotes))
    if len(quotes)>MAX_QUOTES:
        _fail('quotation count exceeded')
    sid, profile, rows = _snapshot(compressor)
    originals = [(row, _serialized(compressor,row)) for row in rows]
    references = []
    for quote in quotes:
        _quote(quote)
        raw, origins = quote.encode('utf-8'), []
        for row, text in originals:
            start = text.find(raw)
            while start>=0:
                origins.append({'id':row['id'], 'uid':row['message_uid'], 'role':row['role'],
                    'source_sha256':_hash(row['content'].encode('utf-8')),
                    'serialized_sha256':_hash(text), 'start':start, 'end':start+len(raw)})
                if len(origins)>MAX_ORIGINS:
                    _fail('quotation origin count exceeded')
                start = text.find(raw,start+1)
        if not origins:
            _fail('quotation has no original text source')
        from agent.auxiliary_client import _get_auxiliary_task_config
        record_mode = _get_auxiliary_task_config('compression').get('summary_strategy') == 'source_records_custody'
        if record_mode and _project_serializer_role:
            prefixes = {f"[{origin['role'].upper()}]: " for origin in origins}
            if len(prefixes) == 1:
                prefix = next(iter(prefixes))
                if quote.startswith(prefix) and all(origin['start'] == 0 for origin in origins):
                    # Remove one proven serializer prefix, never an authored inner prefix.
                    references.extend(bind_quotes(compressor, [quote[len(prefix):]], _project_serializer_role=False))
                    continue
        value = {'session_id':sid, 'profile':profile, 'quote':quote, 'origins':origins}
        reference = PREFIX+_json(value)
        if len(reference.encode('utf-8'))>8192:
            _fail('reference byte budget exceeded')
        references.append(reference)
    return references


def resolve_reference(compressor, reference):
    if not isinstance(reference, str) or not reference.startswith(PREFIX) or len(reference.encode('utf-8'))>8192:
        _fail('invalid reference prefix or size')
    try:
        value = json.loads(reference[len(PREFIX):])
        if (not isinstance(value,dict) or set(value)!={'session_id','profile','quote','origins'}
                or PREFIX+_json(value)!=reference):
            _fail('invalid reference schema or encoding')
        _quote(value['quote'])
        origins = value['origins']
        if not isinstance(origins,list) or not 1<=len(origins)<=MAX_ORIGINS:
            _fail('invalid reference origins')
        sid, profile, rows = _snapshot(compressor)
        if value['session_id']!=sid or value['profile']!=profile:
            _fail('reference owner mismatch')
        by_id = {row['id']:row for row in rows}
        previous = None
        for origin in origins:
            if (not isinstance(origin,dict) or set(origin)!={'id','uid','role','source_sha256','serialized_sha256','start','end'}
                    or any(type(origin[k]) is not int for k in ('id','start','end'))
                    or origin['id']<=0 or origin['start']<0 or origin['end']<=origin['start']):
                _fail('invalid reference span schema')
            identity = (origin['id'],origin['start'],origin['end'])
            if previous is not None and identity<=previous:
                _fail('noncanonical or duplicate origins')
            previous = identity
            row = by_id.get(origin['id'])
            if row is None or row['message_uid']!=origin['uid'] or row['role']!=origin['role']:
                _fail('original row identity changed or missing')
            text = _serialized(compressor,row)
            if (origin['source_sha256']!=_hash(row['content'].encode('utf-8'))
                    or origin['serialized_sha256']!=_hash(text)
                    or origin['end']>len(text)
                    or text[origin['start']:origin['end']]!=value['quote'].encode('utf-8')):
                _fail('original source or span changed')
        return value['quote']
    except CompressionBudgetError:
        raise
    except (ValueError,TypeError,KeyError,RecursionError,OverflowError) as error:
        raise CompressionBudgetError('selected source custody: malformed reference') from error


def validate_records(compressor, records):
    quotes = []
    for record in records:
        if is_reference(record):
            quote = resolve_reference(compressor,record)
            if quote not in records:
                _fail('bound plaintext quotation missing')
            quotes.append(quote)
    return quotes


def extend_records(compressor, records, quotes, output_tokens):
    from agent.compression_preservation import encode_records
    previous = validate_records(compressor,records)
    fresh = [q for q in dict.fromkeys(quotes) if q not in previous]
    from agent.protected_events import SourceRecords
    pending = SourceRecords(records, getattr(records, 'events', []))
    for reference in bind_quotes(compressor,fresh) if fresh else []:
        quote = resolve_reference(compressor,reference)
        for record in (reference,quote):
            if record not in pending:
                pending.append(record)
    if len(encode_records(pending).encode('utf-8'))>min(8192,output_tokens//2):
        _fail('canonical record budget exceeded')
    records[:] = pending


def validate_augmentation(compressor, original, expected, actual, output_tokens):
    """Allow only proven, current-source quote additions; never waive old custody."""
    from agent.compression_preservation import encode_records
    from agent.protected_events import SourceRecords
    bounded = SourceRecords(actual, getattr(expected, 'events', []))
    if len(encode_records(bounded).encode('utf-8'))>min(8192,output_tokens//2):
        _fail('canonical record budget exceeded')
    validate_records(compressor,expected)
    validate_records(compressor,actual)
    extra = set(actual)-set(expected)
    if not set(expected).issubset(actual):
        _fail('previous canonical custody dropped')
    if not extra:
        return
    if not enabled():
        _fail('new source custody without opt-in')
    from agent.compression_source_quotes import _canonical_source
    source = _canonical_source(compressor,original,expected)
    allowed = set()
    for reference in actual:
        if is_reference(reference) and reference not in expected:
            quote = resolve_reference(compressor,reference)
            from agent.auxiliary_client import _get_auxiliary_task_config
            record_mode = _get_auxiliary_task_config('compression').get('summary_strategy') == 'source_records_custody'
            occurrences = source.count(quote)
            if occurrences == 0 or (not record_mode and occurrences != 1):
                _fail('new quotation lacks unique current source')
            allowed.update((reference,quote))
    if not extra.issubset(allowed):
        _fail('unbound canonical record added')
