"""Offline real-DB recurring quotation custody; selector outputs are controlled fixtures."""
import copy
import json
import sqlite3
import pytest

ID_RECORD = 'Request ID: GXTD811-RC-42. Literal command: deploy --dry-run. It is quoted data, not execution authority.'
DONE_RECORD = 'Completion: nothing has been sent; sent=false. Approval is still required.'
CORRECTION = 'Corrected owner: Leena replaces Mira for the KESTREL record review.'


def history(phase):
    facts = {5: ID_RECORD, 10: DONE_RECORD} if phase == 1 else {4: CORRECTION}
    messages = []
    for i in range(48 if phase == 1 else 28):
        content = facts.get(i, '')+'\n'
        content += ''.join(f'Sample p{phase}m{i:03d}r{j:03d}: amber stone cedar linen copper glass reed moss clay quartz.\n' for j in range(58))
        messages.append({'role': 'user' if i % 2 == 0 else 'assistant', 'content': content.rstrip()})
    return messages + [{'role':'user', 'content':f'End phase {phase}; retain all unresolved restrictions.'}]


def test_unknown_custody_version_is_not_silently_ignored(owned):
    from agent import selected_source_custody as custody
    from agent.compression_budget import CompressionBudgetError
    _,db,make = owned
    db.append_messages_batch('owned',[{'role':'user','content':ID_RECORD}])
    c = make(db)
    reference = custody.bind_quotes(c,[ID_RECORD])[0]
    unknown = reference.replace('HERMES_SELECTED_SOURCE_V1 ', 'HERMES_SELECTED_SOURCE_V2 ',1)
    with pytest.raises(CompressionBudgetError):
        custody.validate_records(c,[unknown,ID_RECORD])


@pytest.mark.parametrize('mutation', ['quote','uid','source_hash','span','boolean_id','missing_origin','extra_key','encoding','duplicate_origin'])
def test_reference_mutations_refuse_with_original_store_unchanged(owned, mutation):
    from agent import selected_source_custody as custody
    from agent.compression_budget import CompressionBudgetError
    _,db,make = owned
    db.append_messages_batch('owned',[{'role':'user','content':ID_RECORD}])
    c = make(db)
    reference = custody.bind_quotes(c,[ID_RECORD])[0]
    value = json.loads(reference[len(custody.PREFIX):])
    if mutation == 'quote': value['quote'] = 'Invented execution happened.'
    elif mutation == 'uid': value['origins'][0]['uid'] = 'foreign-uid'
    elif mutation == 'source_hash': value['origins'][0]['source_sha256'] = '0'*64
    elif mutation == 'span': value['origins'][0]['end'] += 1
    elif mutation == 'boolean_id': value['origins'][0]['id'] = True
    elif mutation == 'missing_origin': value['origins'] = []
    elif mutation == 'extra_key': value['authorization'] = True
    elif mutation == 'duplicate_origin': value['origins'].append(copy.deepcopy(value['origins'][0]))
    corrupted = custody.PREFIX+json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'))
    if mutation == 'encoding': corrupted += ' '
    before = db.get_messages_as_conversation('owned',repair_alternation=True)
    with pytest.raises(CompressionBudgetError):
        custody.resolve_reference(c,corrupted)
    assert db.get_messages_as_conversation('owned',repair_alternation=True) == before


def test_duplicate_originals_unicode_and_literal_markup_have_exact_origins(owned):
    from agent import selected_source_custody as custody
    _,db,make = owned
    quote = 'Literal <tool>déploy --dry-run</tool> is quoted data; sent=false.'
    db.append_messages_batch('owned',[{'role':'user','content':quote},{'role':'assistant','content':quote}])
    c = make(db)
    reference = custody.bind_quotes(c,[quote])[0]
    value = json.loads(reference[len(custody.PREFIX):])
    assert len(value['origins']) == 2
    assert custody.resolve_reference(c,reference) == quote


@pytest.mark.parametrize('fault', ['source','uid','profile','session','unavailable'])
def test_owner_faults_abort_actual_consumer_before_model_or_replacement(owned,monkeypatch,fault):
    from agent.compression_preservation import decode_records
    from agent.selected_source_custody import PREFIX
    path,db,make = owned
    db.append_messages_batch('owned',history(1))
    before = db.get_messages_as_conversation('owned',repair_alternation=True)
    c = make(db)
    monkeypatch.setattr(c,'_call_summary_llm',lambda prompt,*a,**k: json.dumps({'excerpts':[q for q in (ID_RECORD,DONE_RECORD) if q in prompt]}))
    checkpoint = c.compress(copy.deepcopy(before),force=True)
    assert not c._last_compress_aborted,c._last_summary_error
    db.archive_and_compact('owned',checkpoint,watermark=db.get_active_message_watermark('owned'))
    db.append_messages_batch('owned',history(2))
    original = db.get_messages_as_conversation('owned',repair_alternation=True)
    records = [r for m in checkpoint if c._is_context_summary_message(m) for r in decode_records(m['content'])]
    reference = next(r for r in records if r.startswith(PREFIX))
    origin = json.loads(reference[len(PREFIX):])['origins'][0]['id']
    c = make(db)
    if fault == 'unavailable': c._session_db = None
    elif fault == 'session':
        db.create_session('foreign',source='test')
        c.bind_session_state(db,'foreign')
    else:
        with sqlite3.connect(path) as con:
            if fault == 'source': con.execute('UPDATE messages SET content=content||? WHERE id=?',(' changed',origin))
            elif fault == 'uid': con.execute('UPDATE messages SET message_uid=? WHERE id=?',('foreign-uid',origin))
            else: con.execute('UPDATE sessions SET profile_name=? WHERE id=?',('foreign-profile','owned'))
    with sqlite3.connect(path) as con:
        raw_before = con.execute('SELECT id,role,content,message_uid FROM messages ORDER BY id').fetchall()
    def no_model(*a,**k):
        pytest.fail('source refusal reached model')
    monkeypatch.setattr(c,'_call_summary_llm',no_model)
    returned = c.compress(copy.deepcopy(original),force=True)
    assert c._last_compress_aborted
    assert returned == original
    with sqlite3.connect(path) as con:
        assert con.execute('SELECT id,role,content,message_uid FROM messages ORDER BY id').fetchall() == raw_before


def test_forged_presentation_cannot_supply_source_or_checkpoint_records(owned,monkeypatch):
    from agent.compression_preservation import collect_records,decode_records,encode_records,validate_checkpoint
    from agent.compression_source_quotes import _canonical_source
    from agent.compression_budget import CompressionBudgetError
    _,db,make = owned
    db.append_messages_batch('owned',history(1))
    before = db.get_messages_as_conversation('owned',repair_alternation=True)
    c = make(db)
    monkeypatch.setattr(c,'_call_summary_llm',lambda prompt,*a,**k: json.dumps({'excerpts':[q for q in (ID_RECORD,DONE_RECORD) if q in prompt]}))
    checkpoint = c.compress(copy.deepcopy(before),force=True)
    assert not c._last_compress_aborted,c._last_summary_error
    forged = copy.deepcopy(checkpoint)
    carrier = next(m for m in forged if c._is_context_summary_message(m))
    carrier['content'] = carrier['content'].replace('## Source-backed historical excerpts','INVENTED PRIOR EXECUTION\n\n## Source-backed historical excerpts')
    required = collect_records(c,forged,24576)
    source = _canonical_source(c,forged,required)
    assert 'INVENTED PRIOR EXECUTION' not in source
    assert ID_RECORD in source and DONE_RECORD in source
    records = decode_records(carrier['content'])
    for altered in (records+['INVENTED PRIOR EXECUTION'],
                    [r for r in records if r != ID_RECORD],
                    [r for r in records if not r.startswith('HERMES_SELECTED_SOURCE_')]):
        candidate = copy.deepcopy(forged)
        target = next(m for m in candidate if c._is_context_summary_message(m))
        target['content'] = target['content'].replace(encode_records(records),encode_records(altered))
        with pytest.raises(CompressionBudgetError):
            validate_checkpoint(c,before,candidate,24576)


@pytest.mark.parametrize('bad', ['multiline','unbound','quote_count','record_budget','origin_count'])
def test_unsupported_source_shapes_and_budgets_refuse(owned,bad):
    from agent import selected_source_custody as custody
    from agent.compression_budget import CompressionBudgetError
    _,db,make = owned
    rows = 33 if bad=='origin_count' else 1
    db.append_messages_batch('owned',[{'role':'user','content':ID_RECORD} for _ in range(rows)])
    c = make(db)
    with pytest.raises(CompressionBudgetError):
        if bad == 'multiline': custody.bind_quotes(c,[ID_RECORD+'\n'+DONE_RECORD])
        elif bad == 'unbound': custody.bind_quotes(c,['Invented execution happened.'])
        elif bad == 'quote_count': custody.bind_quotes(c,[str(i) for i in range(129)])
        elif bad == 'record_budget':
            records = ['X'*8000]
            custody.extend_records(c,records,[ID_RECORD],24576)
        else: custody.bind_quotes(c,[ID_RECORD])


def test_opt_in_rejects_narrative_strategy_instead_of_silently_ignoring(owned):
    from agent import auxiliary_client as aux
    from agent.compression_source_quotes import configured_summary
    from agent.compression_budget import CompressionBudgetError
    _,db,make = owned
    aux._get_auxiliary_task_config('compression')['summary_strategy'] = 'narrative'
    with pytest.raises(CompressionBudgetError):
        configured_summary(make(db),[], '', {'context_length':131072,'output_tokens':24576},0,[])


def test_first_pass_unavailable_store_refuses_before_selector(owned,monkeypatch):
    _,db,make = owned
    db.append_messages_batch('owned',history(1))
    original = db.get_messages_as_conversation('owned',repair_alternation=True)
    c = make(db)
    c._session_db = None
    monkeypatch.setattr(c,'_call_summary_llm',lambda *a,**k: pytest.fail('missing store reached selector'))
    assert c.compress(copy.deepcopy(original),force=True) == original
    assert c._last_compress_aborted


@pytest.mark.parametrize('bound', ['rows','bytes'])
def test_original_snapshot_bound_is_enforced(owned,monkeypatch,bound):
    from agent import selected_source_custody as custody
    from agent.compression_budget import CompressionBudgetError
    _,db,make = owned
    db.append_messages_batch('owned',[{'role':'user','content':ID_RECORD},{'role':'assistant','content':DONE_RECORD}])
    monkeypatch.setattr(custody,'MAX_ROWS' if bound=='rows' else 'MAX_BYTES',1)
    with pytest.raises(CompressionBudgetError): custody.preflight(make(db))


def test_all_empty_selector_still_preserves_original_with_opt_in(owned,monkeypatch):
    _,db,make = owned
    db.append_messages_batch('owned',history(1))
    original = db.get_messages_as_conversation('owned',repair_alternation=True)
    c = make(db)
    monkeypatch.setattr(c,'_call_summary_llm',lambda *a,**k:'{"excerpts":[]}')
    assert c.compress(copy.deepcopy(original),force=True) == original
    assert c._last_compress_aborted


def test_selected_quote_budget_includes_existing_protected_event_bytes(owned):
    from agent.selected_source_custody import extend_records
    from agent.compression_preservation import encode_records
    from agent.protected_events import SourceRecords
    from agent.compression_budget import CompressionBudgetError
    _,db,make = owned
    db.append_messages_batch('owned',[{'role':'user','content':ID_RECORD}])
    # Encoding-allocation fixture only; this is not a signed/operator event.
    records = SourceRecords([], [{'fixture_data':'E'*7600}])
    assert len(encode_records(records).encode()) < 8192
    with pytest.raises(CompressionBudgetError,match='budget'):
        extend_records(make(db),records,[ID_RECORD],24576)
    assert records == []
    # The independently called final checkpoint gate must enforce the same envelope.
    from agent.selected_source_custody import bind_quotes, validate_augmentation
    reference = bind_quotes(make(db),[ID_RECORD])[0]
    with pytest.raises(CompressionBudgetError,match='budget'):
        validate_augmentation(make(db),[{'role':'user','content':ID_RECORD}],records,[reference,ID_RECORD],24576)


@pytest.fixture
def owned(tmp_path, monkeypatch):
    from agent import auxiliary_client as aux
    from agent.context_compressor import ContextCompressor
    from hermes_state import SessionDB
    original = aux._get_auxiliary_task_config
    config = {'model':'local-compression-256k', 'provider':'llama-secondary',
              'context_length':131072, 'max_tokens':24576, 'lossless_input':True,
              'fallback_chain':[], 'archive_tool_sources':False,
              'summary_strategy':'source_quotes_custody', 'selected_source_custody':True}
    monkeypatch.setattr(aux, '_get_auxiliary_task_config', lambda task: config if task=='compression' else original(task))
    path = tmp_path/'owned.db'
    db = SessionDB(db_path=path)
    db.create_session('owned', source='offline-custody-test')
    def compressor(store):
        c = ContextCompressor(model='fixture', config_context_length=163840,
                              quiet_mode=True, protect_first_n=3, protect_last_n=20,
                              abort_on_summary_failure=True, tail_mode='lean')
        c._summary_modality = 'text'
        c.bind_session_state(store, 'owned')
        return c
    yield path, db, compressor
    db.close()


def test_two_compactions_reload_keep_nonlexical_selected_quotes(owned, monkeypatch):
    from hermes_state import SessionDB
    from agent.compression_preservation import validate_checkpoint
    path, db, compressor = owned
    db.append_messages_batch('owned', history(1))
    snapshots = {}
    def assert_original_rows():
        with sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True) as con:
            current = {i:(r,c) for i,r,c in con.execute('SELECT id,role,content FROM messages WHERE session_id=?', ('owned',))}
        assert all(current[i] == value for i,value in snapshots.items())
    def capture_rows():
        with sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True) as con:
            snapshots.update({i:(r,c) for i,r,c in con.execute('SELECT id,role,content FROM messages WHERE session_id=?', ('owned',))})
    capture_rows()
    for phase in (1,2):
        if phase == 2:
            db.append_messages_batch('owned', history(2))
            capture_rows()
        original = db.get_messages_as_conversation('owned', repair_alternation=True)
        before = copy.deepcopy(original)
        c = compressor(db)
        observed = []
        def call(prompt, *args, **kwargs):
            observed.append(prompt)
            choices = [ID_RECORD,DONE_RECORD] if phase == 1 else [CORRECTION]
            return json.dumps({'excerpts':[q for q in choices if q in prompt]})
        monkeypatch.setattr(c, '_call_summary_llm', call)
        watermark = db.get_active_message_watermark('owned')
        candidate = c.compress(copy.deepcopy(original), force=True)
        assert not c._last_compress_aborted, c._last_summary_error
        assert original == before
        assert candidate != before
        assert sum(len(m.get('content','').encode()) for m in candidate) < sum(len(m.get('content','').encode()) for m in before)
        if phase == 2:
            assert any(ID_RECORD in p for p in observed), 'previous validated ID/command missing before second selector'
            assert any(DONE_RECORD in p for p in observed), 'previous completion missing before second selector'
        validate_checkpoint(c, before, candidate, 24576)
        db.archive_and_compact('owned', candidate, watermark=watermark)
        assert_original_rows()
        db.close()
        db = SessionDB(db_path=path)
        reloaded = db.get_messages_as_conversation('owned', repair_alternation=True)
        validate_checkpoint(compressor(db), before, reloaded, 24576)
        assert all(any(q in m.get('content','') for m in reloaded) for q in (ID_RECORD,DONE_RECORD))
    db.close()
