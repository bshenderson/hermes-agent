"""Bounded complete canonical-source projections, never semantic keyword selection.

The dictionary is visible historical text; ordered row runs bind every projected
line occurrence to its role and reconstruction digest. Digests are integrity,
not authentication: custody still belongs to the engine's canonical history.
"""
import hashlib
import json

from agent.compression_budget import CompressionBudgetError
from agent.protected_events import SourceRecords

START = '<!-- HERMES_SOURCE_ROWS_V1 -->'
END = '<!-- END_HERMES_SOURCE_ROWS_V1 -->'
MAX_PROJECTED_BYTES = 1048576
MAX_ROWS = 4096


class BoundRecords(SourceRecords):
    def __init__(self, records, events, rows):
        super().__init__(records, events)
        self.source_rows = rows


def enabled():
    from agent.auxiliary_client import _get_auxiliary_task_config
    config = _get_auxiliary_task_config('compression')
    if config.get('summary_strategy') != 'source_bound':
        return False
    if config.get('lossless_input') is not True or config.get('archive_tool_sources') is True:
        raise CompressionBudgetError('source_bound requires inline lossless_input source custody')
    return True


def encode_rows(rows):
    payload = json.dumps(rows, ensure_ascii=False, separators=(',', ':'))
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e')
    return START + '\n' + payload + '\n' + END


def strip_model_rows(text):
    while START in text:
        before, rest = text.split(START, 1)
        if END not in rest:
            raise CompressionBudgetError('incomplete model source-row block')
        text = before + rest.split(END, 1)[1]
    if END in text:
        raise CompressionBudgetError('unmatched model source-row block')
    return text


def _restore_row(row, records, remaining):
    if not isinstance(row, dict) or set(row) != {'role', 'tool_call_id_sha256', 'runs', 'sha256'}:
        raise CompressionBudgetError('invalid source-row schema')
    if (row['role'] not in ('user', 'assistant', 'tool') or not isinstance(row['runs'], list)
            or not isinstance(row['tool_call_id_sha256'], str)
            or (row['role'] != 'tool' and row['tool_call_id_sha256'])):
        raise CompressionBudgetError('invalid source-row role or runs')
    identity = row['tool_call_id_sha256']
    if identity and (len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity)):
        raise CompressionBudgetError('invalid source-row tool identity digest')
    parts, size = [], 0
    for run in row['runs']:
        if (not isinstance(run, list) or len(run) != 2 or any(type(n) is not int for n in run)
                or not 0 <= run[0] < len(records) or run[1] <= 0):
            raise CompressionBudgetError('invalid source-row reference')
        line = records[run[0]]
        size += (len(line.encode('utf-8')) + 1) * run[1]
        if size > remaining:
            raise CompressionBudgetError('source-row reconstruction exceeds bounded allocation')
        parts.extend([line] * run[1])
    text = '\n'.join(parts)
    if hashlib.sha256(text.encode('utf-8')).hexdigest() != row['sha256']:
        raise CompressionBudgetError('source-row reconstruction digest mismatch')
    return (row['role'], text, identity), size


def decode_rows(text, records):
    if text.count(START) != 1 or text.count(END) != 1:
        raise CompressionBudgetError('missing or ambiguous source-row custody')
    payload = text.split(START, 1)[1].split(END, 1)[0]
    if len(payload.encode('utf-8')) > 8192:
        raise CompressionBudgetError('source-row envelope exceeds preservation budget')
    try:
        rows = json.loads(payload)
    except (ValueError, RecursionError) as error:
        raise CompressionBudgetError('invalid source-row JSON') from error
    if not isinstance(rows, list) or len(rows) > MAX_ROWS or encode_rows(rows) not in text:
        raise CompressionBudgetError('noncanonical source-row custody')
    restored, remaining = [], MAX_PROJECTED_BYTES
    for row in rows:
        item, size = _restore_row(row, records, remaining)
        restored.append(item)
        remaining -= size
    return restored


def _make_records(projections, events):
    records, indexes, rows, size = [], {}, [], 0
    for role, text, tool_call_id in projections:
        if any(char in text for char in ('\r', '\u0085', '\u2028', '\u2029')):
            raise CompressionBudgetError('unsupported source-row line separator')
        lines = text.split('\n')
        normalized = '\n'.join(lines)
        size += len(normalized.encode('utf-8')) + len(lines)
        if size > MAX_PROJECTED_BYTES or len(rows) >= MAX_ROWS:
            raise CompressionBudgetError('canonical source exceeds bounded allocation')
        runs = []
        for line in lines:
            if line not in indexes:
                indexes[line] = len(records)
                records.append(line)
            index = indexes[line]
            if runs and runs[-1][0] == index:
                runs[-1][1] += 1
            else:
                runs.append([index, 1])
        rows.append({'role': role, 'tool_call_id_sha256': tool_call_id, 'runs': runs,
                     'sha256': hashlib.sha256(normalized.encode('utf-8')).hexdigest()})
    return BoundRecords(records, events, rows)


def collect(compressor, messages, output_tokens):
    from agent.compression_preservation import _text, decode_records, encode_records
    from agent.context_compressor import _handoff_only_content, COMPRESSED_SUMMARY_METADATA_KEY
    from agent.protected_events import collect_events, decode_events
    events = collect_events(compressor, messages)
    projections = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') == 'system':
            continue
        fresh = message
        if compressor._is_context_summary_message(message):
            if not message.get(COMPRESSED_SUMMARY_METADATA_KEY):
                raise CompressionBudgetError('untyped handoff cannot establish source-row custody')
            text = _text(_handoff_only_content(message.get('content')))
            prior_events = decode_events(text)
            if prior_events != events[:len(prior_events)]:
                raise CompressionBudgetError('source-row events differ from canonical journal')
            historical = decode_records(text)
            projections.extend(decode_rows(text, historical))
            fresh = compressor._strip_context_summary_handoff_message(message)
        if fresh is None:
            continue
        role = fresh.get('role')
        if role not in ('user', 'assistant', 'tool'):
            raise CompressionBudgetError('unsupported canonical source-row role')
        serialized = compressor._serialize_records_for_summary([fresh], lossless=True)
        if not serialized:
            continue
        prefix = f"[TOOL RESULT {fresh.get('tool_call_id', '')}]: " if role == 'tool' else f'[{role.upper()}]: '
        tool_call_id = fresh.get('tool_call_id', '') if role == 'tool' else ''
        if not isinstance(tool_call_id, str):
            raise CompressionBudgetError('unsupported source-row tool identity')
        identity = hashlib.sha256(tool_call_id.encode('utf-8')).hexdigest() if tool_call_id else ''
        projections.append((role, '\n\n'.join(serialized).removeprefix(prefix), identity))
    result = _make_records(projections, events)
    if len(encode_records(result).encode('utf-8')) > min(8192, output_tokens // 2):
        raise CompressionBudgetError('complete source projection exceeds preservation budget')
    return result


def validate(text, expected):
    from agent.compression_preservation import decode_records, encode_records
    actual = decode_records(text)
    restored = decode_rows(text, actual)
    # Rebuilding proves canonical dictionary order, complete use and row ordering.
    rebuilt = _make_records(restored, expected.events)
    if encode_records(rebuilt) != encode_records(expected) or encode_records(expected) not in text:
        raise CompressionBudgetError('source-row custody differs from canonical history')
