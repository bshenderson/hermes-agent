"""Source custody for opt-in lossless compression, not model-summary inference.

Only fresh canonical messages and versioned historical records are sources.
Legacy handoffs cannot establish source custody and refuse replacement.
"""
import json
import re

from agent.compression_budget import CompressionBudgetError

CONTRACT = '<!-- HERMES_CHECKPOINT_PRESERVATION_V2 -->'
START = '<!-- HERMES_SOURCE_COMMITMENTS_V1 -->'
END = '<!-- END_HERMES_SOURCE_COMMITMENTS_V1 -->'
_FIELD = re.compile(r'^\s*(?:owner|assignee|deadline|due)\s*:', re.IGNORECASE)
_MODAL = re.compile(r'\b(?:must|shall|deadline|due)\b', re.IGNORECASE)
_NAMED = re.compile(r"^(?:[^:\n]{1,80}:\s*)?[A-Z][\w'-]{0,60}\s+(?i:must|shall)\b")
_DATE = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b'
    r'|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b', re.IGNORECASE,
)


def _text(content):
    from agent.context_compressor import _part_text
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(text for part in content if isinstance(text := _part_text(part), str))
    raise CompressionBudgetError('unsupported source handoff content')


def decode_records(content):
    """Decode only canonical historical data; redact AFTER decoding."""
    from agent.context_compressor import _redact_compaction_text
    text = _text(content)
    if text.count(CONTRACT) != 1 or text.count(START) != 1 or text.count(END) != 1:
        raise CompressionBudgetError('missing or ambiguous checkpoint source custody')
    prefix, body = text.split(START, 1)
    payload, _ = body.split(END, 1)
    if not prefix.rstrip().endswith(CONTRACT):
        raise CompressionBudgetError('unsupported checkpoint source custody')
    records = []
    for line in payload.splitlines():
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError as error:
            raise CompressionBudgetError('invalid historical source data') from error
        if not isinstance(value, str) or any(char in value for char in ('\n', '\r')):
            raise CompressionBudgetError('invalid historical source record')
        records.append(_redact_compaction_text(value))
    return records


def collect_records(compressor, messages, output_tokens):
    """Harvest the unpruned canonical transcript, never prior model prose."""
    from agent.context_compressor import _handoff_only_content, COMPRESSED_SUMMARY_METADATA_KEY
    records = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') == 'system':
            continue
        fresh = message
        if compressor._is_context_summary_message(message):
            if not message.get(COMPRESSED_SUMMARY_METADATA_KEY):
                raise CompressionBudgetError('untyped handoff cannot establish source custody')
            records.extend(decode_records(_handoff_only_content(message.get('content'))))
            fresh = compressor._strip_context_summary_handoff_message(message)
        if fresh is None:
            continue
        serialized = compressor._serialize_records_for_summary([fresh], lossless=True)
        if not serialized:
            continue
        role = str(fresh.get('role', '')).upper()
        prefix = (f"[TOOL RESULT {fresh.get('tool_call_id', '')}]: " if role == 'TOOL' else f'[{role}]: ')
        # Remove only the serializer-owned prefix, not matching literal tool data.
        text = '\n\n'.join(serialized).removeprefix(prefix)
        records.extend(line for line in text.splitlines()
                       if _FIELD.search(line) or _NAMED.search(line)
                       or (_MODAL.search(line) and _DATE.search(line)))
    records = list(dict.fromkeys(records))
    if len(encode_records(records).encode('utf-8')) > min(8192, output_tokens // 2):
        raise CompressionBudgetError('source commitments exceed preservation budget')
    return records


def encode_records(records):
    # Square/angle brackets cannot collide with end/think/handoff markers.
    quoted = []
    for record in records:
        value = json.dumps(record, ensure_ascii=False)
        for char, replacement in (('<', '\\u003c'), ('>', '\\u003e'),
                                  ('[', '\\u005b'), (']', '\\u005d')):
            value = value.replace(char, replacement)
        quoted.append(value)
    return CONTRACT + '\n' + START + '\n' + '\n'.join(quoted) + '\n' + END


def _bounded_user_excerpts(turns, byte_budget):
    from agent import context_compressor as cc
    header = '\n\n' + cc._LEAN_USER_MESSAGES_HEADING + '\n'
    footer = '\n(Quoted excerpts only; omitted text remains in pre-compaction session history.)'
    remaining = byte_budget - len((header + footer).encode('utf-8'))
    quoted = []
    for message in reversed(turns):
        if message.get('role') != 'user':
            continue
        text = cc._content_text_for_contains(message.get('content'))
        if cc._synthetic_user_row(text):
            continue
        text = cc._redact_compaction_text(text.strip())
        separator = '\n\n' if quoted else ''
        low, high = 0, min(len(text), cc._LEAN_USER_MESSAGE_MAX_CHARS)
        best = ''
        while low <= high:
            limit = (low + high) // 2
            if limit < len(text) and limit <= cc.ELISION_MARKER_MAX_LEN:
                low = limit + 1
                continue
            excerpt = cc.elide(text, limit)
            rendered = separator + '> ' + excerpt.replace('\n', '\n> ')
            if len(rendered.encode('utf-8')) <= remaining:
                best = rendered
                low = limit + 1
            else:
                high = limit - 1
        if not best:
            break
        quoted.append(best)
        remaining -= len(best.encode('utf-8'))
    return header + ''.join(quoted) + footer if quoted else ''


def augment_summary(compressor, summary, turns, records=None, output_tokens=None):
    """Allocate optional presentation after reserving the complete checkpoint."""
    from agent import context_compressor as cc
    if getattr(compressor, 'tail_mode', 'lean') != 'lean':
        return summary
    sections = (
        (cc._LEAN_ANCHOR_HEADING, lambda: cc._redact_compaction_text(cc._build_anchor_index(turns))),
        (cc._LEAN_USER_MESSAGES_HEADING, lambda: cc._redact_compaction_text(cc._build_verbatim_user_section(turns))),
        (cc._LEAN_RECOVERY_HEADING, lambda: cc._build_recovery_footer(
            getattr(compressor, '_session_id', '') or '', len(turns))),
    )
    if records is None:
        for heading, build in sections:
            if heading not in summary:
                summary += build()
        return summary
    if output_tokens is None:
        raise CompressionBudgetError('checkpoint output budget is required for source records')
    # All immutable custody and assembly bytes precede optional quote allocation.
    framing = cc.SUMMARY_PREFIX + '\n\n\n' + encode_records(records) + '\n\n' + cc._SUMMARY_END_MARKER
    remaining = output_tokens - len((framing + summary).encode('utf-8'))
    if remaining < 0:
        raise CompressionBudgetError('mandatory checkpoint exceeds output byte budget')
    additions = {}
    # Keep the archive pointer before spending room on anchors or quotations.
    for index in (2, 0):
        heading, build = sections[index]
        if heading not in summary:
            text = build()
            size = len(text.encode('utf-8'))
            if size <= remaining:
                additions[index] = text
                remaining -= size
    if cc._LEAN_USER_MESSAGES_HEADING not in summary:
        additions[1] = _bounded_user_excerpts(turns, remaining)
    return summary + ''.join(additions.get(index, '') for index in range(3))


def frame_summary(compressor, summary, records, output_tokens):
    """Restore trusted records only AFTER all model-summary normalization."""
    from agent.context_compressor import _redact_compaction_text
    summary = compressor._strip_summary_prefix(summary)
    # Model-emitted blocks never acquire source authority.
    while START in summary:
        before, rest = summary.split(START, 1)
        if END not in rest:
            raise CompressionBudgetError('incomplete model source block')
        summary = before + rest.split(END, 1)[1]
    if END in summary:
        raise CompressionBudgetError('unmatched model source block')
    summary = _redact_compaction_text(summary.replace(CONTRACT, '')).rstrip()
    result = compressor._with_summary_prefix(summary + '\n\n' + encode_records(records))
    if len(result.encode('utf-8')) > output_tokens:
        raise CompressionBudgetError('final checkpoint exceeds output byte budget')
    if decode_records(result) != records:
        raise CompressionBudgetError('checkpoint source records changed during framing')
    return result


def validate_checkpoint(compressor, original, candidate, output_tokens):
    """Gate the exact candidate presented for installation, including fallback."""
    from agent.context_compressor import _handoff_only_content, COMPRESSED_SUMMARY_METADATA_KEY
    if candidate == original:
        return
    expected = collect_records(compressor, original, output_tokens)
    actual, handoff_bytes, carriers = [], 0, 0
    for message in candidate:
        if not compressor._is_context_summary_message(message):
            continue
        if not message.get(COMPRESSED_SUMMARY_METADATA_KEY):
            raise CompressionBudgetError('untyped handoff cannot establish source custody')
        content = _handoff_only_content(message.get('content'))
        text = _text(content)
        decoded = decode_records(text)
        if encode_records(decoded) not in text:
            raise CompressionBudgetError('checkpoint source encoding is not canonical sanitized data')
        # A second parser pass must see the same records after normalization.
        if decode_records(compressor._strip_summary_prefix(text)) != decoded:
            raise CompressionBudgetError('checkpoint source records changed during normalization')
        actual.extend(decoded)
        handoff_bytes += len(text.encode('utf-8'))
        carriers += 1
    if not carriers or set(actual) != set(expected):
        raise CompressionBudgetError('final checkpoint source records differ from canonical history')
    if handoff_bytes > output_tokens:
        raise CompressionBudgetError('final checkpoint exceeds output byte budget')


def refuse_replacement(compressor, previous_summary, error):
    """Attempt ownership is enforced by the compressor's guarded attributes."""
    from agent.conversation_compression import _raise_if_stale_attempt
    _raise_if_stale_attempt(compressor)
    compressor._previous_summary = previous_summary
    compressor._last_compress_aborted = True
    compressor._last_compression_made_progress = False
    compressor._last_summary_error = str(error)


def reject_installation(agent, original, candidate, attempt):
    """Final gate after caller transforms; rollback remains attempt-owned."""
    from agent.context_compressor import lossless_settings
    compressor = agent.context_compressor
    try:
        budget = lossless_settings()
        if budget:
            validate_checkpoint(compressor, original, candidate, budget['output_tokens'])
    except CompressionBudgetError as error:
        attempt.restore_compressor(compressor)
        refuse_replacement(compressor, attempt.snapshot.get('_previous_summary'), error)
        return True
    return False
