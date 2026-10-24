"""Media controller contracts; synthetic tool results, no GPU/network effects."""
import json
from types import SimpleNamespace

import pytest

from agent.media_request_action import build_prelude, eligible
from agent.turn_scripted_prelude import PreludeFinal, play_prelude


def context():
    return {'profile': 'media', 'subject': 'test-owner', 'chat_id': 'test-chat', 'user_message_id': 'test-user-message'}


def agent():
    return SimpleNamespace(tools=[{'function': {'name': name}} for name in ('media_match_workflow', 'media_generate')])


def match():
    return json.dumps({'success': True, 'match': {'requires_image': False, 'needs_input': False, 'modality': 'image-generation'}, 'job_lifecycle': {'resource_class': 'gpu0-media-image', 'artifact_finalization': 'async_status_then_artifact_poll'}, 'media_generate_args': {'workflow_id': 'synthetic-workflow', 'prompt': 'synthetic prompt', 'wait_for_completion': False}})


def test_native_action_submits_once_and_retains_hold(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_MEDIA_MATCH_WORKFLOW_AUTO_SUBMIT', raising=False)
    prompt = 'Generate an anime image of a fox'
    steps = build_prelude(agent(), profile='media', turn=context(), prompt=prompt)
    assert next(steps)[1] == 'media_match_workflow'
    assert steps.send(match())[1] == 'media_generate'
    with pytest.raises(StopIteration) as ended:
        steps.send(json.dumps({'success': True, 'submission': {'prompt_id': 'synthetic-job', 'status_url': '/synthetic-status'}}))
    assert isinstance(ended.value.value, PreludeFinal)
    assert 'submitted' in ended.value.value.text
    duplicate = build_prelude(agent(), profile='media', turn=context(), prompt=prompt)
    with pytest.raises(StopIteration) as ended:
        next(duplicate)
    assert 'already handled or interrupted' in ended.value.value.text


@pytest.mark.parametrize('value', [None, {'success': False}, {'success': True}])
def test_unknown_or_failed_result_has_no_retry(tmp_path, monkeypatch, value):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_MEDIA_MATCH_WORKFLOW_AUTO_SUBMIT', raising=False)
    steps = build_prelude(agent(), profile='media', turn=context(), prompt='Generate an image of a fox')
    next(steps)
    steps.send(match())
    with pytest.raises(StopIteration) as ended:
        steps.send(json.dumps(value))
    assert 'No automatic retry' in ended.value.value.text


@pytest.mark.parametrize('prompt', ['Explain how to generate an image', 'Edit this image', 'Generate a video of a fox', 'Generate an image from the attached photo'])
def test_out_of_scope_has_no_prelude(prompt):
    assert not eligible(prompt)
    assert build_prelude(agent(), profile='media', turn=context(), prompt=prompt) is None


def test_other_profile_has_no_prelude():
    assert build_prelude(agent(), profile='govcon', turn=context(), prompt='Generate an image of a fox') is None


def test_no_prelude_preserves_native_model_path():
    assert play_prelude(None, None, None) == ('run', None)
