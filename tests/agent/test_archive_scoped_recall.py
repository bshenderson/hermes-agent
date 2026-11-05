"""GXTD-814: scoped deferred authority and bounded full-row archive recall."""
import json

import pytest

from hermes_state import SessionDB


def _publish(db, content='Exact omitted KESTREL detail.', neighbors=False):
    from agent.compression_archive import request_archive_mint
    db.create_session('owned', 'cli', profile_name='default')
    if neighbors:
        db.append_message('owned', 'assistant', 'BEFORE ' + 'x' * 5000)
    db.append_message('owned', 'user', content)
    if neighbors:
        db.append_message('owned', 'assistant', 'AFTER ' + 'y' * 5000)
    carrier = {'role': 'assistant', 'content': 'Summary.', '_compressed_summary': True}
    request_archive_mint([carrier])
    db.archive_and_compact('owned', [carrier], watermark=db.get_active_message_watermark('owned'))


def _agent(tmp_path, monkeypatch, db, *, disabled=False):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({
        'tools': {'tool_search': {'enabled': 'on', 'defer': ['session_search']}},
        'auxiliary': {'compression': {'archive_receipts': True}},
    }))
    from run_agent import AIAgent
    return AIAgent(api_key='fixture', provider='custom', model='fixture',
                   base_url='http://127.0.0.1:9/v1', quiet_mode=True,
                   skip_context_files=True, skip_memory=True, max_iterations=1,
                   enabled_toolsets=['session_search'],
                   disabled_toolsets=['session_search'] if disabled else None,
                   session_id='owned', session_db=db)


def test_deferred_authorized_tool_recalls_through_real_agent(tmp_path, monkeypatch):
    from agent.inline_tool_executors import _session_search
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        _publish(db)
        agent = _agent(tmp_path, monkeypatch, db)
        assert 'session_search' not in agent.valid_tool_names
        assert 'tool_call' in agent.valid_tool_names
        result = json.loads(_session_search(agent, {'query': 'KESTREL'}, None))
        assert result.get('success') is True, result
        assert 'Exact omitted KESTREL detail.' in json.dumps(result)


def test_large_neighbors_do_not_hide_complete_matching_row(tmp_path):
    from agent.compression_archive import ArchiveRecall, MAX_RESULT_BYTES, resolve_presentation
    content = 'KESTREL ' + 'z' * 4574
    with SessionDB(db_path=tmp_path / 'large.db') as db:
        _publish(db, content, neighbors=True)
        reference = resolve_presentation(db, 'owned', 'default')
        result = ArchiveRecall(db, 'owned', 'default', allowed_tools={'session_search'}).invoke(
            reference, {'query': 'KESTREL'})
        assert len(result.encode()) <= MAX_RESULT_BYTES
        parsed = json.loads(result)
        assert parsed['success'] is True
        assert [row['content'] for row in parsed['messages']] == [content]
        assert parsed['context_omitted'] is True
        assert parsed['mode'] == 'archive-recall'


@pytest.mark.parametrize('control', ['disabled', 'foreign-session', 'foreign-profile', 'invalid-carrier', 'scope-fault'])
def test_deferred_recall_refuses_without_owned_authority(tmp_path, monkeypatch, control):
    from agent.compression_archive import CARRIER_KEY
    from agent.inline_tool_executors import _session_search
    with SessionDB(db_path=tmp_path / 'control.db') as db:
        _publish(db, 'KESTREL PRIVATE ORIGINAL')
        if control == 'invalid-carrier':
            db._write_sql('UPDATE messages SET display_metadata=? WHERE _compressed_summary=1',
                          (json.dumps({CARRIER_KEY: 'malformed'}),))
        if control == 'scope-fault':
            import model_tools
            original = model_tools.get_tool_definitions
            def definitions(*args, **kwargs):
                if kwargs.get('skip_tool_search_assembly'):
                    raise RuntimeError('controlled scoped catalog failure')
                return original(*args, **kwargs)
            monkeypatch.setattr(model_tools, 'get_tool_definitions', definitions)
        agent = _agent(tmp_path, monkeypatch, db, disabled=control == 'disabled')
        arguments = {'query': 'KESTREL'}
        if control == 'foreign-session':
            arguments['session_id'] = 'foreign'
        if control == 'foreign-profile':
            arguments['profile'] = 'personal'
        result = json.loads(_session_search(agent, arguments, None))
        assert result['success'] is False, result
        assert 'PRIVATE ORIGINAL' not in json.dumps(result)


def test_oversized_matching_row_still_refuses_without_truncation(tmp_path):
    from agent.compression_archive import ArchiveRecall, resolve_presentation
    with SessionDB(db_path=tmp_path / 'oversize.db') as db:
        _publish(db, 'KESTREL ' + 'z' * 9000, neighbors=True)
        reference = resolve_presentation(db, 'owned', 'default')
        with pytest.raises(ValueError, match='result budget exceeded'):
            ArchiveRecall(db, 'owned', 'default', allowed_tools={'session_search'}).invoke(
                reference, {'query': 'KESTREL'})
