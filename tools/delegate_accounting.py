"""Observational delegation accounting; never a budget enforcement source."""
import json


def nested_count(content):
    """Only structured execute_code receipts carry a nested-call observation."""
    try:
        value = json.loads(content)
    except (ValueError, TypeError, RecursionError):
        return None
    count = value.get('tool_calls_made') if isinstance(value, dict) else None
    return count if type(count) is int and count >= 0 else None


def record_receipt(target, content):
    if target.get('tool') != 'execute_code':
        return
    count = nested_count(content)
    if 'nested_tool_calls_reported' in target and target['nested_tool_calls_reported'] != count:
        target['receipt_conflict'] = True
    target['nested_tool_calls_reported'] = count


def summarize(trace, api_calls, duration):
    """Missing receipts/time attribution stay unknown, not zero execution."""
    available = isinstance(trace, list)
    nested = [row for row in (trace or []) if row.get('tool') == 'execute_code']
    known = [row['nested_tool_calls_reported'] for row in nested
             if row.get('nested_tool_calls_reported') is not None
             and not row.get('identity_conflict') and not row.get('receipt_conflict')]
    return {
        'claim_type': 'transcript_observation',
        'api_calls': api_calls if type(api_calls) is int and api_calls >= 0 else None,
        'transcript_tool_calls': len(trace) if available else None,
        'nested_tool_calls_reported': sum(known) if known else None,
        'execute_code_receipts_observed': len(known) if available else None,
        'execute_code_receipts_unknown': len(nested) - len(known) if available else None,
        'elapsed_seconds': duration,
        'pending_seconds': None,
        'execution_seconds': None,
        'timing_scope': 'total elapsed only; pending/execution split not instrumented',
    }
