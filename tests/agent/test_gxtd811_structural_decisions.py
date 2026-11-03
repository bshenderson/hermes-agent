"""Approved successor fixes empty structural decisions in schema, never by coercion."""
import copy
import json
import pytest
from tests.agent.test_gxtd811_selected_custody import owned,history


def test_only_whitespace_fields_are_const_false_and_all_IDs_stay_required():
    from agent.compression_record_selector import catalog,_format
    source='\n \t\r\n[USER]:\n\u200b\nOwner: Zoë.\nfalse\n'
    rows=catalog(source)
    schema=_format(rows)['json_schema']['schema']
    assert schema['required']==[r['id'] for r in rows]
    assert ''.join(r['text'] for r in rows)==source
    for row in rows:
        expected={'type':'boolean','const':False} if not row['text'].strip() else {'type':'boolean'}
        assert schema['properties'][row['id']]==expected
    assert schema['additionalProperties'] is False


def test_structural_true_is_rejected_not_rewritten_and_nonempty_remains_free():
    from agent.compression_record_selector import catalog,decisions
    from agent.compression_budget import CompressionBudgetError
    rows=catalog('\nOwner: Zoë.\n')
    assert decisions(rows,'{"0":false,"1":true}')=={'0':False,'1':True}
    assert decisions(rows,'{"0":false,"1":false}')=={'0':False,'1':False}
    with pytest.raises(CompressionBudgetError,match='structural'):
        decisions(rows,'{"0":true,"1":true}')
    with pytest.raises(CompressionBudgetError):decisions(rows,'{"1":true}')
    with pytest.raises(CompressionBudgetError):decisions(rows,'{"0":0,"1":true}')


def test_backend_ignoring_structural_constraint_preserves_real_consumer_history(owned,monkeypatch):
    from agent import auxiliary_client as aux
    _,db,make=owned
    aux._get_auxiliary_task_config('compression')['summary_strategy']='source_records_custody'
    db.append_messages_batch('owned',history(1))
    original=db.get_messages_as_conversation('owned',repair_alternation=True)
    c=make(db)
    calls=[]
    def illegal(prompt,*args,**kwargs):
        rows=json.loads(prompt.split('<records>\n',1)[1].rsplit('\n</records>',1)[0])
        assert any(not r['text'].strip() for r in rows)
        blank=next(r['id'] for r in rows if not r['text'].strip())
        assert kwargs['response_format']['json_schema']['schema']['properties'][blank]['const'] is False
        calls.append(rows)
        return json.dumps({r['id']:r['id']==blank for r in rows})
    monkeypatch.setattr(c,'_call_summary_llm',illegal)
    result=c.compress(copy.deepcopy(original),force=True)
    assert c._last_compress_aborted and 'structural' in c._last_summary_error
    assert result==original
    assert len(calls)==1
    assert db.get_messages_as_conversation('owned',repair_alternation=True)==original
