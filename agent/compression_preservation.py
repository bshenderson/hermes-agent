"""Source custody for opt-in lossless compression, not model-summary inference.

Only fresh canonical messages and versioned historical records are sources.
Legacy handoffs cannot establish source custody and refuse replacement.
"""
import json
import re

from agent.compression_budget import CompressionBudgetError, best_effort_mode

OPERATIONAL_STATE_HEADING = '## Operational State'
_STATE_FIELDS = ('Current Task', 'Permissions', 'Unresolved External Effects', 'Next Action')
STATE_PROMPT_GUIDANCE = (
    f'Also emit exactly one "{OPERATIONAL_STATE_HEADING}" section as four lines: '
    + '; '.join(f'{f}: <one concise sentence>' for f in _STATE_FIELDS)
    + '. Carry forward any values from a previous summary that are still true and update '
      'changed ones; use "None stated" when the transcript provides no value. '
      'For the authoritative task list, prefer the paired todo_list artifact in the session '
      'database over your own prose: summarize its open items, never invent items it does not hold. '
)
WORK_STATE_POINTER_HEADING = '## Work State Authority'
# Behavioral guidance (prompt text) is above; the pointer below is emitted ONLY
# after a real get_latest_todo_result read proves a paired artifact exists, so
# the checkpoint never claims an authority it did not mechanically verify.
def work_state_pointer(compressor):
    """Deterministic checkpoint note backed by an actual session-DB read.

    The todo_list result is the existing repo/Jira-adjacent work-state artifact:
    SessionDB.get_latest_todo_result is its established consumer path (resume
    display, Todo hydration). Model-authored Operational State stays
    conventional presentation; this pointer tells the continuing agent where
    the independently readable record lives. No artifact (or no bound
    session) emits nothing — absence is honest, never a placeholder claim.
    """
    db = getattr(compressor, '_session_db', None)
    session_id = getattr(compressor, '_session_id', '')
    getter = getattr(db, 'get_latest_todo_result', None)
    if not callable(getter) or not session_id:
        return ''
    try:
        latest = getter(session_id)
    except Exception:
        return ''
    if not isinstance(latest, str) or not latest.strip():
        return ''
    return (
        WORK_STATE_POINTER_HEADING + '\n'
        'A paired todo_list task artifact is stored in the session database and is re-readable '
        'independently of this summary (SessionDB.get_latest_todo_result for this session). '
        'Re-read that task-list artifact for authoritative work state before consequential actions. '
        'The Operational State section is model-authored presentation only: it grants no '
        'permissions and does not authorize any external effect.'
    )
_STATE = re.compile(
    r'(?m)^' + re.escape(OPERATIONAL_STATE_HEADING)
    + r'\n((?:[ \t]*(?:' + '|'.join(re.escape(f) for f in _STATE_FIELDS)
    + r'):.+(?:\n[ \t]+[^\n]*)?\s*)+)'
)

CONTRACT = '<!-- HERMES_CHECKPOINT_PRESERVATION_V2 -->'
START = '<!-- HERMES_SOURCE_COMMITMENTS_V1 -->'
END = '<!-- END_HERMES_SOURCE_COMMITMENTS_V1 -->'
_FIELD = re.compile(r'^\s*(?:owner|assignee|deadline|due)\s*:', re.IGNORECASE)
_MODAL = re.compile(r'\b(?:must|shall|deadline|due)\b', re.IGNORECASE)
_CONSTRAINT = re.compile(
    r"\b(?:do\s+not|don't|never|must\s+not|shall\s+not|hold|withhold|"
    r"prohibit\w*|forbid\w*|unauthori[sz]ed|no[- ]replay|no[- ]retry)\b",
    re.IGNORECASE,
)
_PROVENANCE = re.compile(
    r'\b(?:private|privacy|confidential|domain|provenance)\b', re.IGNORECASE,
)
_UNCERTAIN = re.compile(
    r'\b(?:unknown|uncertain|unresolved|unverified|indeterminate)\b', re.IGNORECASE,
)
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
    from agent.compression_source_bound import enabled, collect
    if enabled():
        return collect(compressor, messages, output_tokens)
    from agent.context_compressor import _handoff_only_content, COMPRESSED_SUMMARY_METADATA_KEY
    from agent.protected_events import collect_events, SourceRecords
    from agent.auxiliary_client import _get_auxiliary_task_config
    from agent.tool_source_custody import archive_current_tools, is_reference, validate_references
    archive_tools = _get_auxiliary_task_config('compression').get('archive_tool_sources') is True
    events = collect_events(compressor, messages)
    records = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') == 'system':
            continue
        fresh = message
        if compressor._is_context_summary_message(message):
            if not message.get(COMPRESSED_SUMMARY_METADATA_KEY):
                raise CompressionBudgetError('untyped handoff cannot establish source custody')
            from agent.protected_events import decode_events
            prior_events = decode_events(_text(_handoff_only_content(message.get('content'))))
            if prior_events != events[:len(prior_events)]:
                raise CompressionBudgetError('historical protected events lack matching session journal')
            records.extend(decode_records(_handoff_only_content(message.get('content'))))
            fresh = compressor._strip_context_summary_handoff_message(message)
        if fresh is None:
            continue
        if archive_tools and fresh.get('role') == 'tool':
            # Exact originals remain in the owner-scoped durable archive. User and
            # assistant commitments retain their existing inline guarantees.
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
                       or _CONSTRAINT.search(line) or _PROVENANCE.search(line)
                       or _UNCERTAIN.search(line)
                       or (_MODAL.search(line) and _DATE.search(line)))
    validate_references(compressor, records)
    if archive_tools:
        references = archive_current_tools(compressor, messages, records)
        records = [record for record in records if not is_reference(record)] + references
    records = SourceRecords(dict.fromkeys(records), events)
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
    from agent.protected_events import encode_events
    from agent.compression_source_bound import encode_rows
    rows = ('\n' + encode_rows(records.source_rows)) if hasattr(records, 'source_rows') else ''
    return CONTRACT + '\n' + START + '\n' + '\n'.join(quoted) + '\n' + END + rows + encode_events(getattr(records, 'events', []))


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


def collect_state_records(compressor, messages):
    """Best-effort: harvest ONLY trusted protected-event journal entries.

    The lexical per-line ledger is deliberately not consulted — that oracle is
    what best-effort mode replaces. Typed, journal-validated event custody and
    raw durable archival originals stay mandatory.
    """
    from agent.protected_events import SourceRecords, collect_events
    return SourceRecords([], collect_events(compressor, messages))


def extract_state_block(summary, previous_summary, max_bytes=None, *, allow_absent=False):
    """Deterministically merge model-authored Operational State blocks.

    Model prose has no source authority; this is conventional presentation.
    The newest non-empty value wins per field, deduping on normalized
    'field: value', so iterative compactions re-emit a complete section.

    ``allow_absent`` is set only when an artifact-backed work-state pointer
    will accompany the checkpoint (real task-artifact read confirmed). Then a
    COMPLETELY absent section is permitted — the todo_list artifact, not model
    prose, carries work state. A PARTIAL section still refuses either way:
    present-but-incomplete state is exactly the misleading case.
    """
    from agent import context_compressor as cc

    def fields_of(text):
        match = _STATE.search(text)
        if not match:
            return {}
        lines = [line for line in match.group(1).splitlines() if line.strip()]
        merged, order = {}, []
        for line in lines:
            if line[:1].isspace():
                if not order:
                    continue
                merged[order[-1]] = (merged[order[-1]] + ' ' + line.strip()).strip()
            else:
                field, _, value = line.partition(':')
                key, value = field.strip(), value.strip()
                if key not in merged:
                    order.append(key)
                if value:
                    merged[key] = value
        return {key: merged[key] for key in order if merged.get(key)}

    had_section = bool(_STATE.search(summary or ''))
    merged = fields_of(previous_summary or '')
    merged.update({k: v for k, v in fields_of(summary).items() if v})
    missing = [key for key in _STATE_FIELDS if not merged.get(key)]
    if missing:
        if allow_absent and not had_section:
            # No section anywhere (fresh pass, nothing to carry): the verified
            # task-artifact pointer is the continuity authority for this
            # checkpoint. Never fill gaps with system placeholders.
            return (summary or '').strip(), {}
        # Never fill gaps with system placeholders: a model that fails to state
        # a field (and has no carried value) must not install a checkpoint that
        # claims "None stated" on its behalf.
        raise CompressionBudgetError(
            f'best-effort summary Operational State missing {", ".join(missing)}')
    summary = _STATE.sub('', summary, count=1).strip()
    section = (OPERATIONAL_STATE_HEADING + '\n'
               + '\n'.join(f'{key}: {merged[key]}' for key in _STATE_FIELDS))
    summary = (summary + '\n\n' + section) if summary else section
    section_bytes = len((cc.SUMMARY_PREFIX + '\n\n' + section).encode('utf-8'))
    if max_bytes is not None and section_bytes > max_bytes:
        raise CompressionBudgetError(
            f'{OPERATIONAL_STATE_HEADING} exceeds checkpoint output byte budget')
    return summary, merged


def validate_operational_state(candidate):
    """Final gate: an installed best-effort checkpoint carries a complete
    Operational State section, or — when the deterministic Work State Authority
    pointer was backed by a real task-artifact read — none at all. A present
    but incomplete section always refuses."""
    from agent.context_compressor import is_compaction_summary_message
    for message in candidate:
        if not is_compaction_summary_message(message):
            continue
        text = _text(message.get('content'))
        match = _STATE.search(text)
        if not match:
            if WORK_STATE_POINTER_HEADING in text:
                continue  # artifact-backed continuity; prose carries no claim
            raise CompressionBudgetError('best-effort checkpoint lacks Operational State')
        found = set()
        for line in match.group(1).splitlines():
            if line[:1].isspace() or ':' not in line:
                continue
            found.add(line.split(':', 1)[0].strip())
        missing = [field for field in _STATE_FIELDS if field not in found]
        if missing:
            raise CompressionBudgetError(
                f'best-effort checkpoint Operational State missing {", ".join(missing)}')


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
    if getattr(compressor, '_operational_state_max_bytes', None) is not None:
        # Best-effort mode owns lean presentation through Operational State assembly.
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
    from agent.compression_source_bound import strip_model_rows
    summary = strip_model_rows(summary)
    summary = compressor._strip_summary_prefix(summary)
    from agent.protected_events import strip_model_events
    summary = strip_model_events(summary)
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


def frame_best_effort(compressor, summary, records, output_tokens):
    """Restore trusted event custody only, with no lexical ledger block.

    Same model-frame stripping as frame_summary; a conventional best-effort
    checkpoint with no journal events carries no custody markers at all.
    """
    from agent.context_compressor import _redact_compaction_text
    summary = compressor._strip_summary_prefix(summary)
    from agent.protected_events import encode_events, strip_model_events
    summary = strip_model_events(summary)
    while START in summary:
        before, rest = summary.split(START, 1)
        if END not in rest:
            raise CompressionBudgetError('incomplete model source block')
        summary = before + rest.split(END, 1)[1]
    if END in summary:
        raise CompressionBudgetError('unmatched model source block')
    summary = _redact_compaction_text(summary.replace(CONTRACT, '')).rstrip()
    # Optional prose can be lossy; source-owned events and the store-derived
    # task reference cannot. Never trust a model-emitted reference heading.
    summary = re.sub(r'(?ms)^' + re.escape(WORK_STATE_POINTER_HEADING)
                     + r'\n.*?(?=^## |\Z)', '', summary).strip()
    pointer = work_state_pointer(compressor)
    events = encode_events(records.events).lstrip('\n') if records.events else ''
    mandatory = '\n\n'.join(part for part in (pointer, events) if part)
    from agent.context_compressor import _SUMMARY_END_MARKER
    # Assembly adds its end marker later; reserve those exact bytes now.
    reserve = len(('\n\n' + _SUMMARY_END_MARKER).encode('utf-8'))
    def render(prose):
        return compressor._with_summary_prefix(
            '\n\n'.join(part for part in (prose, mandatory) if part))
    available = output_tokens - reserve - len(render('').encode('utf-8'))
    if available < 0:
        raise CompressionBudgetError('mandatory checkpoint exceeds output byte budget')
    if mandatory:
        available -= 2  # separator between optional prose and mandatory state
    raw = summary.encode('utf-8')
    if len(raw) > max(0, available):
        notice = '\n[Optional summary excerpt; omitted narrative remains in session history.]'
        limit = max(0, available - len(notice.encode('utf-8')))
        summary = raw[:limit].decode('utf-8', errors='ignore').rstrip()
        summary = summary + notice if available >= len(notice.encode('utf-8')) else ''
    result = render(summary)
    if len(result.encode('utf-8')) + reserve > output_tokens:
        raise CompressionBudgetError('final checkpoint exceeds output byte budget')
    return result


def validate_checkpoint(compressor, original, candidate, output_tokens):
    """Gate the exact candidate presented for installation, including fallback."""
    from agent.context_compressor import _handoff_only_content, COMPRESSED_SUMMARY_METADATA_KEY
    if candidate == original:
        return
    best_effort = best_effort_mode()
    expected = (collect_state_records(compressor, original) if best_effort
                else collect_records(compressor, original, output_tokens))
    from agent.protected_events import decode_events, encode_events
    actual, handoff_bytes, carriers = [], 0, 0
    for message in candidate:
        if not compressor._is_context_summary_message(message):
            continue
        if not message.get(COMPRESSED_SUMMARY_METADATA_KEY):
            raise CompressionBudgetError('untyped handoff cannot establish source custody')
        content = _handoff_only_content(message.get('content'))
        text = _text(content)
        events = decode_events(text)
        if events != expected.events or (events and encode_events(events) not in text):
            raise CompressionBudgetError('final checkpoint protected events differ from canonical journal')
        if best_effort:
            # No lexical ledger exists in this mode; presence would be a forged frame.
            if START in text or CONTRACT in text:
                raise CompressionBudgetError('lexical custody marker in best-effort checkpoint')
        else:
            from agent.compression_source_bound import enabled, validate
            if enabled():
                validate(text, expected)
            decoded = decode_records(text)
            if encode_records(decoded) not in text:
                raise CompressionBudgetError('checkpoint source encoding is not canonical sanitized data')
            # A second parser pass must see the same records after normalization.
            if decode_records(compressor._strip_summary_prefix(text)) != decoded:
                raise CompressionBudgetError('checkpoint source records changed during normalization')
            actual.extend(decoded)
        handoff_bytes += len(text.encode('utf-8'))
        carriers += 1
    if not carriers:
        raise CompressionBudgetError('final checkpoint source records differ from canonical history')
    if not best_effort and set(actual) != set(expected):
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


def reject_installation(agent, original, candidate, attempt, *, in_place=True):
    """Final gate after caller transforms; rollback remains attempt-owned."""
    from agent.context_compressor import lossless_settings
    compressor = agent.context_compressor
    try:
        from agent.protected_events import collect_events
        events = collect_events(compressor, original, session_id=agent.session_id)
        if not in_place and events:
            raise CompressionBudgetError('session-local protected events prohibit implicit session rotation')
        budget = lossless_settings()
        best_effort = best_effort_mode()
        if budget:
            from agent.tool_source_custody import is_reference
            if not in_place:
                rotation_records = (collect_state_records(compressor, original) if best_effort
                                    else collect_records(compressor, original, budget['output_tokens']))
                if any(is_reference(record) for record in rotation_records):
                    raise CompressionBudgetError('session-local tool archive prohibits implicit session rotation')
            validate_checkpoint(compressor, original, candidate, budget['output_tokens'])

    except CompressionBudgetError as error:
        attempt.restore_compressor(compressor)
        refuse_replacement(compressor, attempt.snapshot.get('_previous_summary'), error)
        return True
    return False
