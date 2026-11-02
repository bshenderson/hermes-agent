"""GXTD-810 citation owner entrypoint; disposable real SQLite, no model calls."""
import pytest

from hermes_state import SessionDB
from agent.compression_preservation import encode_records
from tests.agent.test_gxtd641_checkpoint_preservation import compressor, history, make_agent


def test_decoded_citation_resolves_to_checkpoint_and_original_rows(tmp_path):
    from agent import source_citations
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', 'api_server', profile_name='research')
        quote = 'Source [B]: Cedar belongs to Felix, due October 19.'
        db.append_message('s', 'user', quote)
        db.append_message('s', 'assistant', encode_records([quote]), _compressed_summary=True)
        checkpoint = source_citations.checkpoint_identity(db, session_id='s', profile='research')
        result = source_citations.validate_citations(db, session_id='s', profile='research',
            checkpoint=checkpoint, citations=[quote])
        assert result['checkpoint'] == checkpoint
        assert result['citations'][0]['quote'] == quote
        assert result['citations'][0]['sources'][0]['record_index'] == 0
        assert result['citations'][0]['sources'][0]['original_rows'][0]['message_uid']


@pytest.fixture
def source(tmp_path):
    from agent import source_citations
    with SessionDB(db_path=tmp_path / 'citations.db') as db:
        db.create_session('s', 'api_server', profile_name='research')
        quote = 'Source [B]: Felix must deliver Cedar October 19.'
        db.append_message('s', 'user', quote)
        db.append_message('s', 'assistant', encode_records([quote]), _compressed_summary=True)
        checkpoint = source_citations.checkpoint_identity(db, session_id='s', profile='research')
        yield db, quote, checkpoint


@pytest.mark.parametrize('attack', ['foreign-profile', 'foreign-session', 'missing-session',
                                   'stale-content', 'stale-row', 'forged-carrier',
                                   'forged-record', 'missing-original', 'missing-frame',
                                   'duplicate-frame', 'no-summary-marker', 'model-prose'])
def test_refuses_custody_or_scope_failure_without_mutation(source, attack):
    from agent import source_citations
    db, quote, checkpoint = source
    sid, profile = 's', 'research'
    if attack == 'foreign-profile':
        profile = 'default'
    elif attack == 'foreign-session':
        db.create_session('other', 'api_server', profile_name='research')
        db.append_message('other', 'user', quote)
        db.append_message('other', 'assistant', encode_records([quote]), _compressed_summary=True)
        sid = 'other'  # text equality cannot reuse another session's checkpoint
    elif attack == 'missing-session':
        sid = 'absent'
    else:
        if attack == 'stale-content':
            db._write_sql("UPDATE messages SET content = content || ' changed' WHERE session_id = ?", ('s',))
        elif attack == 'stale-row':
            db.append_message('s', 'user', 'new pending work')
        elif attack == 'forged-carrier':
            db._write_sql('UPDATE messages SET _compressed_summary = 0 WHERE session_id = ?', ('s',))
            db.append_message('s', 'user', encode_records([quote]))
        elif attack == 'forged-record':
            quote = 'Source [B]: Invented must deliver October 19.'
            db._write_sql('UPDATE messages SET content = ? WHERE _compressed_summary = 1', (encode_records([quote]),))
        elif attack == 'missing-original':
            db._write_sql('DELETE FROM messages WHERE _compressed_summary = 0')
        elif attack == 'missing-frame':
            db._write_sql('UPDATE messages SET content = ? WHERE _compressed_summary = 1', ('plain model prose',))
        elif attack == 'duplicate-frame':
            db._write_sql('UPDATE messages SET content = content || content WHERE _compressed_summary = 1')
        elif attack == 'no-summary-marker':
            db._write_sql('UPDATE messages SET _compressed_summary = 0')
        else:
            db._write_sql('UPDATE messages SET content = ? || content WHERE _compressed_summary = 1', ('Model says imaginary source.\n',))
            quote = 'Model says imaginary source.'
        if not attack.startswith('stale-'):
            # Bind freshly mutated bytes: refusal must be custody, not only stale hashing.
            checkpoint = source_citations.checkpoint_identity(db, session_id=sid, profile=profile)
    before = db.get_messages('s', include_inactive=True)
    with pytest.raises(ValueError):
        source_citations.validate_citations(db, session_id=sid, profile=profile,
                                          checkpoint=checkpoint, citations=[quote])
    assert db.get_messages('s', include_inactive=True) == before


def test_duplicate_origins_and_span_occurrences_are_retained(source):
    from agent import source_citations
    db, quote, _ = source
    db.append_message('s', 'user', quote)
    checkpoint = source_citations.checkpoint_identity(db, session_id='s', profile='research')
    result = source_citations.validate_citations(db, session_id='s', profile='research',
        checkpoint=checkpoint, citations=[quote])
    origins = result['citations'][0]['sources'][0]['original_rows']
    assert len(origins) == 2
    assert len({row['message_uid'] for row in origins}) == 2


@pytest.mark.parametrize('quote', ['', '2024-10-19', 'Felix October 19', 'x' * 8193])
def test_literal_changes_and_invalid_quotes_refuse(source, quote):
    from agent import source_citations
    db, _, checkpoint = source
    with pytest.raises(ValueError):
        source_citations.validate_citations(db, session_id='s', profile='research',
            checkpoint=checkpoint, citations=[quote])


def test_encoded_secret_cannot_be_cited_or_restored(tmp_path):
    from agent import source_citations
    from agent.context_compressor import _redact_compaction_text
    with SessionDB(db_path=tmp_path / 'secret.db') as db:
        db.create_session('s', 'api_server', profile_name='research')
        secret = 'https://' + 'fixture-user:fixture-password' + '@example.invalid/path'
        db.append_message('s', 'user', secret)
        safe = _redact_compaction_text(secret)
        assert safe != secret
        db.append_message('s', 'assistant', encode_records([safe]), _compressed_summary=True)
        checkpoint = source_citations.checkpoint_identity(db, session_id='s', profile='research')
        result = source_citations.validate_citations(db, session_id='s', profile='research',
            checkpoint=checkpoint, citations=[safe])
        assert 'fixture-password' not in str(result)
        with pytest.raises(ValueError, match='redaction'):
            source_citations.validate_citations(db, session_id='s', profile='research',
                checkpoint=checkpoint, citations=[secret])
        encoded = encode_records([secret]).replace('fixture-password', ''.join(f'\\u{ord(c):04x}' for c in 'fixture-password'))
        db._write_sql('UPDATE messages SET content = ? WHERE _compressed_summary = 1', (encoded,))
        checkpoint = source_citations.checkpoint_identity(db, session_id='s', profile='research')
        with pytest.raises(ValueError, match='sanitized'):
            source_citations.validate_citations(db, session_id='s', profile='research',
                checkpoint=checkpoint, citations=[safe])


def test_literal_escape_is_not_recursively_decoded(tmp_path):
    from agent import source_citations
    with SessionDB(db_path=tmp_path / 'escape.db') as db:
        db.create_session('s', 'api_server', profile_name='research')
        quote = r'Owner [B]: preserve literal \u005b and Ω.'
        db.append_message('s', 'user', quote)
        db.append_message('s', 'assistant', encode_records([quote]), _compressed_summary=True)
        checkpoint = source_citations.checkpoint_identity(db, session_id='s', profile='research')
        assert source_citations.validate_citations(db, session_id='s', profile='research',
            checkpoint=checkpoint, citations=[quote])['citations'][0]['quote'] == quote
        with pytest.raises(ValueError):
            source_citations.validate_citations(db, session_id='s', profile='research',
                checkpoint=checkpoint, citations=[quote.replace(r'\u005b', '[')])


def test_real_two_compactions_reload_then_owner_citation(compressor, tmp_path):
    from agent import source_citations
    quote = 'Source [B]: Felix must deliver Cedar October 19.'
    path = tmp_path / 'compaction.db'
    db = SessionDB(db_path=path)
    try:
        db.create_session('source-session', 'api_server', profile_name='research')
        agent = make_agent(compressor, db)
        agent.compression_in_place = True
        for _ in range(2):
            db.append_messages_batch('source-session', history(quote))
            original = db.get_messages_as_conversation('source-session', repair_alternation=True)
            result, _ = agent._compress_context(original, 'Existing system prompt.', force=True)
            assert result != original
            db.close()
            db = SessionDB(db_path=path)
            agent._session_db = db
            checkpoint = source_citations.checkpoint_identity(db, session_id='source-session', profile='research')
            resolved = source_citations.validate_citations(db, session_id='source-session', profile='research',
                checkpoint=checkpoint, citations=[quote])
            assert resolved['citations'][0]['quote'] == quote
    finally:
        db.close()


def test_match_budget_refuses_instead_of_returning_truncated_origins(source, monkeypatch):
    from agent import source_citations
    db, quote, _ = source
    db.append_message('s', 'user', quote)
    monkeypatch.setattr(source_citations, 'MAX_MATCHES', 1, raising=False)
    checkpoint = source_citations.checkpoint_identity(db, session_id='s', profile='research')
    with pytest.raises(ValueError, match='match budget'):
        source_citations.validate_citations(db, session_id='s', profile='research',
            checkpoint=checkpoint, citations=[quote])


@pytest.mark.parametrize('limit,value', [('MAX_ROWS', 1), ('MAX_BYTES', 16)])
def test_source_budget_refuses_before_result(source, monkeypatch, limit, value):
    from agent import source_citations
    db, _, _ = source
    monkeypatch.setattr(source_citations, limit, value)
    with pytest.raises(ValueError, match='source budget'):
        source_citations.checkpoint_identity(db, session_id='s', profile='research')


@pytest.mark.parametrize('citations', [[], 'text', [None], [True], ['x'] * 33])
def test_unsupported_citation_batch_refuses(source, citations):
    from agent import source_citations
    db, _, checkpoint = source
    with pytest.raises(ValueError):
        source_citations.validate_citations(db, session_id='s', profile='research',
            checkpoint=checkpoint, citations=citations)


def test_database_unavailable_does_not_return_cached_citation(source, monkeypatch):
    from agent import source_citations
    db, quote, checkpoint = source
    def unavailable():
        raise OSError('fixture store unavailable')
    monkeypatch.setattr(db, '_read_ctx', unavailable)
    with pytest.raises(OSError):
        source_citations.validate_citations(db, session_id='s', profile='research',
            checkpoint=checkpoint, citations=[quote])
