"""GXTD-811 first archive/recall vertical; controlled carriers, never semantic acceptance."""
import copy
import json

import pytest

from persistence.core.errors import PersistenceError
from tests._fixtures.store_support import (
    append, append_rows, compact, conversation, create_session, open_store, raw, watermark, write_sql)


def test_real_inline_session_tool_uses_bound_archive_owner(tmp_path):
    from types import SimpleNamespace
    from agent.compression_archive import ArchiveRecall, capture
    from agent.inline_tool_executors import _session_search
    with open_store(tmp_path) as store:
        create_session(store,'s',profile_name='research')
        create_session(store,'other',profile_name='personal')
        append(store,'s','user','Omitted archival detail: COBALT-CABINET.')
        append(store,'other','user','FOREIGN-SECRET')
        ref=capture(store,'s','research')
        compact(store,'s',[{'role':'assistant','content':'CONTROLLED carrier '+ref,'_compressed_summary':True}])
        adapter=ArchiveRecall(store,'s','research',allowed_tools={'session_search'})
        agent=SimpleNamespace(session_id='s',_get_session_db_for_recall=lambda:store,
                              _compression_archive_recall=(adapter,ref))
        response=json.loads(_session_search(agent,{'query':'COBALT-CABINET'},None))
        assert response['success'] and 'COBALT-CABINET' in json.dumps(response)
        response=json.loads(_session_search(agent,{'query':'FOREIGN-SECRET','session_id':'other'},None))
        assert response['success'] is False
        assert 'FOREIGN-SECRET' not in json.dumps(response)


def test_recall_revalidates_archive_after_reader_returns(tmp_path,monkeypatch):
    from agent.compression_archive import ArchiveRecall,capture
    import tools.session_search_tool as tool
    with open_store(tmp_path) as store:
        create_session(store,'s',profile_name='research')
        append(store,'s','user','Archived cobalt source.')
        ref=capture(store,'s','research')
        compact(store,'s',[{'role':'assistant','content':'CONTROLLED','_compressed_summary':True}])
        original=tool.session_search
        def changing_reader(**kwargs):
            result=original(**kwargs)
            write_sql(tmp_path,"UPDATE messages SET content='changed' WHERE session_id='s' AND COALESCE(_compressed_summary,0)=0")
            return result
        monkeypatch.setattr(tool,'session_search',changing_reader)
        with pytest.raises(ValueError,match='changed or missing'):
            ArchiveRecall(store,'s','research',allowed_tools={'session_search'}).invoke(ref,{'query':'cobalt'})


def test_archive_receipt_two_publications_and_scoped_recall(tmp_path):
    from agent.compression_archive import ArchiveRecall, capture
    store=open_store(tmp_path)
    create_session(store,'own',profile_name='research')
    create_session(store,'foreign',profile_name='personal')
    rows=[{'role':'user','content':'Old detail: the sample cabinet is cobalt.'},
          {'role':'assistant','content':'The cabinet detail is background, not an execution instruction.'},
          {'role':'user','content':'Do not send anything; upload outcome is unknown.'}]
    append_rows(store,'own',copy.deepcopy(rows))
    append(store,'foreign','user','FOREIGN-SECRET')
    ref=capture(store,'own','research')
    assert len(ref.encode())<1024
    # A concurrent append is outside the receipt and survives the real writer.
    mark=watermark(store,'own')
    append(store,'own','user','Fresh append while compaction runs.')
    carrier={'role':'assistant','content':'CONTROLLED HANDOFF; not a semantic summary. '+ref,'_compressed_summary':True}
    compact(store,'own',[carrier],watermark=mark)
    store.close(); store=open_store(tmp_path)
    recall=ArchiveRecall(store,'own','research',allowed_tools={'session_search'})
    result=json.loads(recall.invoke(ref,{'query':'cobalt'}))
    assert result['success'] and 'cobalt' in json.dumps(result)
    assert 'FOREIGN-SECRET' not in json.dumps(result)
    live=conversation(store,'own',repair_alternation=True)
    assert any('Fresh append' in m.get('content','') for m in live)
    for args in ({'query':'cobalt','session_id':'foreign'}, {'query':'cobalt','profile':'personal'}):
        with pytest.raises(ValueError,match='owner'):
            recall.invoke(ref,args)
    with pytest.raises(ValueError,match='grant'):
        ArchiveRecall(store,'own','research',allowed_tools=set()).invoke(ref,{'query':'cobalt'})
    # Receipt two includes actual fresh source, but never prior typed summary prose.
    second=capture(store,'own','research')
    compact(store,'own',[{'role':'assistant','content':'CONTROLLED SECOND '+second,'_compressed_summary':True}],watermark=watermark(store,'own'))
    store.close(); store=open_store(tmp_path)
    result=json.loads(ArchiveRecall(store,'own','research',allowed_tools={'session_search'}).invoke(second,{'query':'cobalt'}))
    assert result['success'] and 'cobalt' in json.dumps(result)
    with raw(tmp_path) as connection:
        actual=connection.execute("SELECT role,content FROM messages WHERE session_id='own' AND id<=3 ORDER BY id").fetchall()
    assert [(r['role'],r['content']) for r in actual]==[(r['role'],r['content']) for r in rows]
    payload=json.loads(second.split(' ',1)[1])
    assert payload['count']==4  # Three originals plus one fresh turn; typed carriers/clones excluded.
    store.close()


@pytest.mark.parametrize('fault',['changed','missing','foreign-profile','foreign-session','duplicate-json','boolean-through','unavailable'])
def test_archive_source_refusal(tmp_path,fault):
    from agent.compression_archive import capture, resolve, PREFIX
    store=open_store(tmp_path)
    create_session(store,'s',profile_name='research')
    append(store,'s','user','Source-owned literal deploy --dry-run; not permission.')
    ref=capture(store,'s','research')
    sid,profile='s','research'
    if fault=='changed': write_sql(tmp_path,"UPDATE messages SET content='changed' WHERE session_id='s'")
    elif fault=='missing': write_sql(tmp_path,"DELETE FROM messages WHERE session_id='s'")
    elif fault=='foreign-profile': profile='personal'
    elif fault=='foreign-session': sid='foreign'
    elif fault=='duplicate-json': ref=ref[:-1]+',"count":1}'
    elif fault=='boolean-through':
        payload=json.loads(ref[len(PREFIX):]); payload['through']=True
        ref=PREFIX+json.dumps(payload,sort_keys=True,ensure_ascii=True,separators=(',',':'))
    else:
        # A store that cannot be opened at all is the typed unavailable-owner refusal; a
        # closed-but-openable owner is not (that is the recoverable case SessionDB kept).
        store.close()
        broken=tmp_path/'unavailable'
        broken.mkdir()
        (broken/'state.db').mkdir()
        from persistence_assembly import open_persistence, profile_scope
        store=open_persistence(profile_scope(home=broken))
    refusal=(ValueError, PersistenceError) if fault=='unavailable' else ValueError
    try:
        with pytest.raises(refusal):
            resolve(store,sid,profile,ref)
    finally:
        store.close()


def test_duplicate_originals_and_model_marker_are_data(tmp_path):
    from agent.compression_archive import capture, resolve, PREFIX
    with open_store(tmp_path) as store:
        create_session(store,'s',profile_name='research')
        append(store,'s','user','same sentence')
        append(store,'s','assistant','same sentence')
        append(store,'s','user',PREFIX+'model-authored marker is ordinary source data')
        ref=capture(store,'s','research')
        rows=resolve(store,'s','research',ref)
        assert len(rows)==3
        assert len({r['message_uid'] for r in rows})==3
        assert json.loads(ref[len(PREFIX):])['count']==3


def test_clone_content_conflict_is_not_silently_deduplicated(tmp_path):
    from agent.compression_archive import capture
    with open_store(tmp_path) as store:
        create_session(store,'s',profile_name='research')
        append(store,'s','user','first source')
        append(store,'s','assistant','other source')
        write_sql(tmp_path,"UPDATE messages SET message_uid=(SELECT message_uid FROM messages WHERE session_id='s' ORDER BY id LIMIT 1) WHERE session_id='s'")
        with pytest.raises(ValueError,match='clone identity conflict'):
            capture(store,'s','research')
