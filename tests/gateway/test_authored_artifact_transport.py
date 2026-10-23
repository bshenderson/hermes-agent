"""GXTD-749: opt-in authored artifacts share no Media receipt/polling state."""
import base64
import hashlib
import importlib.util


def transport():
    name = 'gateway.platforms.api_server_artifacts'
    assert importlib.util.find_spec(name), 'Missing automatic authored-artifact transport'
    return __import__(name, fromlist=['artifact_event'])


def test_signed_non_media_turn_transports_two_real_pngs(tmp_path):
    from agent.chat_location import native_context, sign_context
    module = transport()
    user = 'user-one'
    ids = {'chat_id': '11111111-1111-4111-8111-111111111111',
           'message_id': '22222222-2222-4222-8222-222222222222',
           'user_message_id': '33333333-3333-4333-8333-333333333333'}
    body = {'messages': [{'role': 'user', 'content': 'Draw two diagrams'}],
            'hermes_artifact_transport': 'inline-v1'}
    context = native_context(user_id=user, metadata=ids, now=1000)
    body['hermes_chat_context'] = sign_context(context, body=body, user_id=user,
                                               profile='homelab-coordinator', key='fixture-key', now=1000)
    binding = module.artifact_binding(body, profile='homelab-coordinator', key='fixture-key', now=1000)
    assert binding == {'subject_id': user, **ids}
    # Real valid PNG bytes with distinct data; publication reads the actual files.
    from PIL import Image
    first, second = tmp_path/'physical.png', tmp_path/'logical.png'
    Image.new('RGB', (8, 8), 'red').save(first)
    Image.new('RGB', (8, 8), 'blue').save(second)
    event = module.artifact_event(f'MEDIA:{first}\n\nMEDIA:{second}\nMEDIA:{first}', binding=binding, artifact_root=tmp_path)
    assert event['schema'] == 'hermes.artifacts.v1'
    assert event['origin'] == binding
    assert len(event['artifacts']) == 2
    for record, path in zip(event['artifacts'], (first, second)):
        assert base64.b64decode(record['data_base64'], validate=True) == path.read_bytes()
        assert record['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record['filename'] == path.name
        assert record['content_type'] == 'image/png'
    assert module.artifact_binding(body, profile='media', key='fixture-key', now=1000) is None


def test_foreign_profile_output_is_not_read(tmp_path):
    module = transport()
    own, foreign = tmp_path/'own', tmp_path/'foreign'
    own.mkdir()
    foreign.mkdir()
    output = foreign/'private.png'
    output.write_bytes(b'private-image')
    event = module.artifact_event(f'MEDIA:{output}', binding={'subject_id': 'own'}, artifact_root=own)
    assert event['artifacts'] == []
    assert event['errors'] == ['artifact_outside_profile_output_root']


def test_symlink_escape_and_oversized_output_are_refused(tmp_path):
    module = transport()
    root = tmp_path/'output'
    root.mkdir()
    foreign = tmp_path/'foreign.png'
    foreign.write_bytes(b'foreign')
    (root/'escape.png').symlink_to(foreign)
    large = root/'large.png'
    with large.open('wb') as stream:
        stream.truncate(module.MAX_FILE_BYTES + 1)
    event = module.artifact_event(f'MEDIA:{root / "escape.png"}\nMEDIA:{large}',
                                  binding={'subject_id': 'owner'}, artifact_root=root)
    assert not event['artifacts']
    assert event['errors'] == ['artifact_outside_profile_output_root', 'artifact_read_unavailable']


def test_unsigned_and_body_tampered_turns_refuse_transport():
    module = transport()
    import pytest
    from agent.chat_location import native_context, sign_context
    ids = {'chat_id': '11111111-1111-4111-8111-111111111111',
           'message_id': '22222222-2222-4222-8222-222222222222',
           'user_message_id': '33333333-3333-4333-8333-333333333333'}
    body = {'hermes_artifact_transport': 'inline-v1', 'messages': ['original']}
    with pytest.raises(ValueError):
        module.artifact_binding(body, profile='research', key='fixture', now=1000)
    context = native_context(user_id='owner', metadata=ids, now=1000)
    body['hermes_chat_context'] = sign_context(context, body=body, user_id='owner', profile='research', key='fixture', now=1000)
    body['messages'] = ['changed']
    with pytest.raises(ValueError):
        module.artifact_binding(body, profile='research', key='fixture', now=1000)
