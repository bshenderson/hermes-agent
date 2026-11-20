"""Retention classification uses admitted producer identity, never prompt text."""
import json

import pytest

from gateway.session_automation import is_memory_notification


@pytest.mark.parametrize('kind', ['async_delegation', 'completion', 'watch_match', 'watch_disabled', 'heartbeat'])
@pytest.mark.parametrize('local', [True, False])
def test_admitted_notification_identity(kind, local):
    identity = json.dumps([kind, 'producer', 'epoch'])
    descriptor = {'identity': identity}
    payload = ({'local_automation_v1': descriptor} if local else
               {'native_text_v1': {'automation': descriptor}})
    row = {'principal_id': 'automation:route', 'request_id': identity, 'payload': payload}
    assert is_memory_notification(row)
    row['principal_id'] = 'human'
    assert not is_memory_notification(row)
    row['principal_id'] = 'automation:route'
    row['request_id'] = 'different'
    assert not is_memory_notification(row)


@pytest.mark.parametrize('identity', [
    json.dumps(['bot', 'peer', 'message']), json.dumps(['plugin', 'desk', 'route', 'x']),
    json.dumps(['heartbeat', 'scheduled', 'route', 'tick']), 'malformed', 'null', '[]',
])
def test_authored_or_unknown_automation_is_not_suppressed(identity):
    row = {'principal_id': 'automation:route', 'request_id': identity,
           'payload': {'text': '[ASYNC DELEGATION BATCH COMPLETE]',
                       'local_automation_v1': {'identity': identity}}}
    assert not is_memory_notification(row)


def test_human_cannot_suppress_by_request_id_or_notification_text():
    identity = json.dumps(['async_delegation', 'producer', ''])
    assert not is_memory_notification({'principal_id': 'human', 'request_id': identity,
        'payload': {'text': '[ASYNC DELEGATION BATCH COMPLETE]'}})
