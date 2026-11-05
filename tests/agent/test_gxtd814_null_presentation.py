"""Present-null metadata must never restore broad recall or select older custody."""
import json

import pytest

from agent import compression_archive as archive
from hermes_state import SessionDB


@pytest.mark.parametrize('key,with_valid_carrier', [
    (archive.CARRIER_KEY, False), (archive.AVAILABILITY_KEY, False),
    (archive.AVAILABILITY_KEY, True)])
@pytest.mark.parametrize('value', [None, False, 7, 'malformed',
    {'state': 'unavailable', 'reason_code': []},
    {'state': 'unavailable', 'reason_code': {}},
    {'state': 'unavailable', 'reason_code': None},
    {'state': 'unavailable', 'reason_code': False},
    {'state': 'unavailable', 'reason_code': 7}])
def test_newest_invalid_key_blocks_older_valid_receipt(
        tmp_path, monkeypatch, key, with_valid_carrier, value):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({
        'auxiliary': {'compression': {'archive_receipts': True}}}))
    path = tmp_path / 'null.db'
    with SessionDB(db_path=path) as db:
        db.create_session('s', 'cli', profile_name='research')
        db.append_message('s', 'user', 'ARCHIVED-NULL-SENTINEL')
        summary = {'role': 'assistant', 'content': 'Controlled first summary.',
                   '_compressed_summary': True}
        archive.request_archive_mint([summary])
        db.archive_and_compact('s', [summary])
        old = archive.resolve_presentation(db, 's', 'research')
        assert old.startswith(archive.PREFIX)
        # Simulate malformed durable metadata above an older valid carrier.
        # No model output is being trusted to create receipt authority.
        metadata = {archive.CARRIER_KEY: old} if with_valid_carrier else {}
        metadata[key] = value
        db.append_messages_batch('s', [{
            'role': 'assistant', 'content': 'Controlled newer summary.',
            '_compressed_summary': True, 'display_metadata': metadata}])
    with SessionDB(db_path=path) as reopened:
        with pytest.raises(ValueError):
            archive.resolve_presentation(reopened, 's', 'research')
        from run_agent import AIAgent
        agent = AIAgent(api_key='fixture', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-null-fixture', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='s', session_db=reopened)
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None and binding[0] is None
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': 'ARCHIVED-NULL-SENTINEL'}, None))
        assert response['success'] is False
        assert 'ARCHIVED-NULL-SENTINEL' not in json.dumps(response)


@pytest.mark.parametrize('key', [archive.CARRIER_KEY, archive.AVAILABILITY_KEY])
@pytest.mark.parametrize('value', [None, False, 7, {}, [],
    {'state': 'unavailable', 'reason_code': []},
    {'state': 'unavailable', 'reason_code': {}},
    {'state': 'unavailable', 'reason_code': None},
    {'state': 'unavailable', 'reason_code': False},
    {'state': 'unavailable', 'reason_code': 7}])
def test_invalid_presented_carrier_rolls_back_publication(tmp_path, key, value):
    with SessionDB(db_path=tmp_path / 'publication.db') as db:
        db.create_session('s', 'cli', profile_name='research')
        db.append_message('s', 'user', 'Original source remains live.')
        before = db.get_messages('s')
        with pytest.raises(ValueError):
            db.archive_and_compact('s', [{
                'role': 'assistant', 'content': 'Controlled summary.',
                '_compressed_summary': True,
                'display_metadata': {key: value}}])
        assert db.get_messages('s') == before


@pytest.mark.parametrize('metadata', [None, {}, {'ordinary_key': None}])
def test_true_key_absence_keeps_ordinary_behavior(tmp_path, metadata):
    with SessionDB(db_path=tmp_path / 'absence.db') as db:
        db.create_session('s', 'cli', profile_name='research')
        db.append_message('s', 'user', 'Original source.')
        db.archive_and_compact('s', [{
            'role': 'assistant', 'content': 'Controlled ordinary summary.',
            '_compressed_summary': True, 'display_metadata': metadata}])
        assert archive.resolve_presentation(db, 's', 'research') is None


@pytest.mark.parametrize('reason', [[], {}, None, False, 7, 'unknown'])
def test_unavailable_classifier_rejects_malformed_nested_reason(reason):
    status = {'state': 'unavailable', 'reason_code': reason}
    with pytest.raises(ValueError):
        archive.validate_availability_status(status)
    assert archive.carrier_is_unavailable({
        'display_metadata': {archive.AVAILABILITY_KEY: status}}) is False


def test_unavailable_classifier_accepts_known_reason():
    status = {'state': 'unavailable', 'reason_code': 'archive_source_budget_exceeded'}
    assert archive.validate_availability_status(status) == status
    assert archive.carrier_is_unavailable({
        'display_metadata': {archive.AVAILABILITY_KEY: status}}) is True
