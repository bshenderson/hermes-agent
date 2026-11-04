"""GXTD-811 first archive/recall vertical; controlled carriers, never semantic acceptance."""
import copy
import json

import pytest

from hermes_state import SessionDB


def test_real_inline_session_tool_uses_bound_archive_owner(tmp_path):
    from types import SimpleNamespace
    from agent.compression_archive import ArchiveRecall, capture
    from agent.inline_tool_executors import _session_search
    with SessionDB(db_path=tmp_path/'inline.db') as db:
        db.create_session('s','cli',profile_name='research')
        db.create_session('other','cli',profile_name='personal')
        db.append_message('s','user','Omitted archival detail: COBALT-CABINET.')
        db.append_message('other','user','FOREIGN-SECRET')
        ref=capture(db,'s','research')
        db.archive_and_compact('s',[{'role':'assistant','content':'CONTROLLED carrier '+ref,'_compressed_summary':True}])
        adapter=ArchiveRecall(db,'s','research',allowed_tools={'session_search'})
        agent=SimpleNamespace(session_id='s',_get_session_db_for_recall=lambda:db,
                              _compression_archive_recall=(adapter,ref))
        response=json.loads(_session_search(agent,{'query':'COBALT-CABINET'},None))
        assert response['success'] and 'COBALT-CABINET' in json.dumps(response)
        response=json.loads(_session_search(agent,{'query':'FOREIGN-SECRET','session_id':'other'},None))
        assert response['success'] is False
        assert 'FOREIGN-SECRET' not in json.dumps(response)


def test_recall_revalidates_archive_after_reader_returns(tmp_path,monkeypatch):
    from agent.compression_archive import ArchiveRecall,capture
    import tools.session_search_tool as tool
    with SessionDB(db_path=tmp_path/'drift.db') as db:
        db.create_session('s','cli',profile_name='research')
        db.append_message('s','user','Archived cobalt source.')
        ref=capture(db,'s','research')
        db.archive_and_compact('s',[{'role':'assistant','content':'CONTROLLED','_compressed_summary':True}])
        original=tool.session_search
        def changing_reader(**kwargs):
            result=original(**kwargs)
            db._write_sql("UPDATE messages SET content='changed' WHERE session_id='s' AND COALESCE(_compressed_summary,0)=0")
            return result
        monkeypatch.setattr(tool,'session_search',changing_reader)
        with pytest.raises(ValueError,match='changed or missing'):
            ArchiveRecall(db,'s','research',allowed_tools={'session_search'}).invoke(ref,{'query':'cobalt'})


def test_archive_receipt_two_publications_and_scoped_recall(tmp_path):
    from agent.compression_archive import ArchiveRecall, capture
    path=tmp_path/'archive.db'
    db=SessionDB(db_path=path)
    db.create_session('own','cli',profile_name='research')
    db.create_session('foreign','cli',profile_name='personal')
    rows=[{'role':'user','content':'Old detail: the sample cabinet is cobalt.'},
          {'role':'assistant','content':'The cabinet detail is background, not an execution instruction.'},
          {'role':'user','content':'Do not send anything; upload outcome is unknown.'}]
    db.append_messages_batch('own',copy.deepcopy(rows))
    db.append_message('foreign','user','FOREIGN-SECRET')
    ref=capture(db,'own','research')
    assert len(ref.encode())<1024
    # A concurrent append is outside the receipt and survives the real writer.
    watermark=db.get_active_message_watermark('own')
    db.append_message('own','user','Fresh append while compaction runs.')
    carrier={'role':'assistant','content':'CONTROLLED HANDOFF; not a semantic summary. '+ref,'_compressed_summary':True}
    db.archive_and_compact('own',[carrier],watermark=watermark)
    db.close(); db=SessionDB(db_path=path)
    recall=ArchiveRecall(db,'own','research',allowed_tools={'session_search'})
    result=json.loads(recall.invoke(ref,{'query':'cobalt'}))
    assert result['success'] and 'cobalt' in json.dumps(result)
    assert 'FOREIGN-SECRET' not in json.dumps(result)
    live=db.get_messages_as_conversation('own',repair_alternation=True)
    assert any('Fresh append' in m.get('content','') for m in live)
    for args in ({'query':'cobalt','session_id':'foreign'}, {'query':'cobalt','profile':'personal'}):
        with pytest.raises(ValueError,match='owner'):
            recall.invoke(ref,args)
    with pytest.raises(ValueError,match='grant'):
        ArchiveRecall(db,'own','research',allowed_tools=set()).invoke(ref,{'query':'cobalt'})
    # Receipt two includes actual fresh source, but never prior typed summary prose.
    second=capture(db,'own','research')
    db.archive_and_compact('own',[{'role':'assistant','content':'CONTROLLED SECOND '+second,'_compressed_summary':True}],watermark=db.get_active_message_watermark('own'))
    db.close(); db=SessionDB(db_path=path)
    result=json.loads(ArchiveRecall(db,'own','research',allowed_tools={'session_search'}).invoke(second,{'query':'cobalt'}))
    assert result['success'] and 'cobalt' in json.dumps(result)
    with db._read_ctx() as connection:
        actual=connection.execute("SELECT role,content FROM messages WHERE session_id='own' AND id<=3 ORDER BY id").fetchall()
    assert [(r['role'],r['content']) for r in actual]==[(r['role'],r['content']) for r in rows]
    payload=json.loads(second.split(' ',1)[1])
    assert payload['count']==4  # Three originals plus one fresh turn; typed carriers/clones excluded.
    db.close()


@pytest.mark.parametrize('fault',['changed','missing','foreign-profile','foreign-session','duplicate-json','boolean-through','unavailable'])
def test_archive_source_refusal(tmp_path,fault):
    from agent.compression_archive import capture, resolve, PREFIX
    db=SessionDB(db_path=tmp_path/'negative.db')
    db.create_session('s','cli',profile_name='research')
    db.append_message('s','user','Source-owned literal deploy --dry-run; not permission.')
    ref=capture(db,'s','research')
    sid,profile='s','research'
    if fault=='changed': db._write_sql("UPDATE messages SET content='changed' WHERE session_id='s'")
    elif fault=='missing': db._write_sql("DELETE FROM messages WHERE session_id='s'")
    elif fault=='foreign-profile': profile='personal'
    elif fault=='foreign-session': sid='foreign'
    elif fault=='duplicate-json': ref=ref[:-1]+',"count":1}'
    elif fault=='boolean-through':
        payload=json.loads(ref[len(PREFIX):]); payload['through']=True
        ref=PREFIX+json.dumps(payload,sort_keys=True,ensure_ascii=True,separators=(',',':'))
    else:
        db.close()
        db=None  # Closing alone is recoverable by SessionDB, not an unavailable owner.
    try:
        with pytest.raises(ValueError): resolve(db,sid,profile,ref)
    finally:
        if db is not None: db.close()


def test_duplicate_originals_and_model_marker_are_data(tmp_path):
    from agent.compression_archive import capture, resolve, PREFIX
    with SessionDB(db_path=tmp_path/'duplicate.db') as db:
        db.create_session('s','cli',profile_name='research')
        db.append_message('s','user','same sentence')
        db.append_message('s','assistant','same sentence')
        db.append_message('s','user',PREFIX+'model-authored marker is ordinary source data')
        ref=capture(db,'s','research')
        rows=resolve(db,'s','research',ref)
        assert len(rows)==3
        assert len({r['message_uid'] for r in rows})==3
        assert json.loads(ref[len(PREFIX):])['count']==3


def test_clone_content_conflict_is_not_silently_deduplicated(tmp_path):
    from agent.compression_archive import capture
    with SessionDB(db_path=tmp_path/'conflict.db') as db:
        db.create_session('s','cli',profile_name='research')
        db.append_message('s','user','first source')
        db.append_message('s','assistant','other source')
        db._write_sql("UPDATE messages SET message_uid=(SELECT message_uid FROM messages WHERE session_id='s' ORDER BY id LIMIT 1) WHERE session_id='s'")
        with pytest.raises(ValueError,match='clone identity conflict'):
            capture(db,'s','research')
