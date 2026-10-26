"""Versioned, bounded operational tables from explicitly enrolled native tools.

Data is rendered verbatim, never inferred from prompts or model prose. This is
presentation, not authority: the producer still enforces account/profile access.
"""
import copy
import json

SCHEMA = 'homelab.operational-result.v1'
MAX_BYTES = 131072
MAX_ROWS = 1024
ENROLLED_TOOLS = frozenset({'home_inventory'})


def validate(value):
    if not isinstance(value, dict) or set(value) != {
            'schema', 'kind', 'title', 'complete', 'facts', 'columns', 'rows'}:
        raise ValueError('invalid operational result fields')
    if value['schema'] != SCHEMA or value['kind'] != 'table' or value['complete'] is not True:
        raise ValueError('unsupported or incomplete operational result')
    def label(text):
        return isinstance(text, str) and 0 < len(text) <= 1024
    if not label(value['title']):
        raise ValueError('invalid title')
    facts, columns, rows = value['facts'], value['columns'], value['rows']
    if not isinstance(facts, dict) or len(facts) > 16:
        raise ValueError('invalid facts')
    for key, item in facts.items():
        if not label(key) or not (label(item) or type(item) is int or type(item) is bool):
            raise ValueError('invalid fact')
    if not isinstance(columns, list) or not 1 <= len(columns) <= 8 or not all(map(label, columns)):
        raise ValueError('invalid columns')
    if len(set(columns)) != len(columns):
        raise ValueError('duplicate columns')
    if not isinstance(rows, list) or len(rows) > MAX_ROWS:
        raise ValueError('row limit')
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns) or not all(isinstance(cell, str) and len(cell) <= 1024 for cell in row):
            raise ValueError('invalid row')
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False)
    if len(encoded.encode()) > MAX_BYTES:
        raise ValueError('operational result byte limit')
    return copy.deepcopy(value)


def project(tool_name, result):
    if tool_name not in ENROLLED_TOOLS:
        return None
    try:
        if isinstance(result, str):
            if len(result.encode()) > MAX_BYTES * 2:
                raise ValueError('tool result byte limit')
            result = json.loads(result)
        if not isinstance(result, dict) or 'operational_result' not in result:
            raise ValueError('enrolled producer omitted complete typed result')
        return {'operational_result': validate(result['operational_result'])}
    except (ValueError, TypeError, RecursionError, OverflowError):
        return {'operational_result_error': 'Typed result unavailable: invalid, incomplete or oversized producer data. No exact-result delivery is claimed.'}


def home_result(inventory):
    """Adapter for an actual metadata inventory; never accepts generated text."""
    if inventory.get('schema') != 'homelab.home-inventory.v1' or inventory.get('complete') is not True:
        raise ValueError('complete native inventory required')
    entries, counts = inventory['entries'], inventory['counts']
    if not isinstance(entries, list) or len(entries) > MAX_ROWS:
        raise ValueError('inventory row limit')
    computed = dict.fromkeys(('file', 'directory', 'symlink', 'other'), 0)
    names = []
    for row in entries:
        if not isinstance(row, dict) or set(row) != {'name', 'type'} or row['type'] not in computed or not isinstance(row['name'], str):
            raise ValueError('invalid inventory entry')
        computed[row['type']] += 1
        names.append(row['name'])
    if len(set(names)) != len(names) or names != sorted(names):
        raise ValueError('inventory names must be complete, unique and sorted')
    if type(inventory['total_entries']) is not int or inventory['total_entries'] != len(entries):
        raise ValueError('inventory entry total mismatch')
    if counts != computed or any(type(n) is not int for n in counts.values()):
        raise ValueError('inventory type counts mismatch')
    return validate({'schema': SCHEMA, 'kind': 'table', 'title': 'Home directory inventory',
        'complete': True, 'facts': {'host': inventory['host'], 'account': inventory['username'],
        'path': inventory['path'], 'total_entries': len(entries), **computed},
        'columns': ['Name', 'Type'], 'rows': [[row['name'], row['type']] for row in entries]})


def render(value):
    value = validate(value)
    # JSON-quoted cells preserve control characters, literal names and Unicode.
    # A dynamically sized code fence prevents producer data becoming markup.
    lines = [json.dumps(value['title'], ensure_ascii=False)]
    lines += [json.dumps(key, ensure_ascii=False) + ': ' + json.dumps(item, ensure_ascii=False)
              for key, item in value['facts'].items()]
    lines.append(' | '.join(json.dumps(item, ensure_ascii=False) for item in value['columns']))
    lines += [' | '.join(json.dumps(cell, ensure_ascii=False) for cell in row) for row in value['rows']]
    text = '\n'.join(lines)
    runs = [len(run) for run in __import__('re').findall(r'`+', text)]
    fence = '`' * max(3, max(runs, default=0) + 1)
    return '\n\nAuthoritative tool result (complete; metadata only):\n\n' + fence + 'text\n' + text + '\n' + fence + '\n'
