"""Versioned dispatch refusal, not a heuristic over provider error prose."""


def is_no_replay_refusal(error):
    """Only the nonretrying wire contract can terminate all recovery paths."""
    if getattr(error, 'status_code', None) != 412:
        return False
    body = getattr(error, 'body', None)
    if not isinstance(body, dict):
        return False
    value = body.get('error', body)  # OpenAI SDK unwraps the error object.
    return (isinstance(value, dict)
            and value.get('contract') == 'inference.no_replay.v1'
            and value.get('type') == 'no_replay_refusal'
            and value.get('code') == 'dispatch_refused'
            and value.get('retryable') is False)
