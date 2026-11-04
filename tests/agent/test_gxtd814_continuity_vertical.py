"""GXTD-814 offline continuity vertical: trusted receipt minting at real
compaction publication, atomic validation, and ordinary-resume binding.

Everything runs on disposable DB/HOME; the summary adapter is a controlled
fixture (never a native result); the executor is the real inline
``session_search`` route. The accepted best_effort family and profile
isolation are exercised unchanged alongside this new opt-in carrier.
"""
import copy
import json
import time

import pytest

from hermes_state import SessionDB


OMITTED = 'COBALT-CABINET-OMITTED'


def _write_config(tmp_path, **compression):
    base = {'model': 'local-compression-256k', 'context_length': 262144,
            'max_tokens': 24576, 'lossless_input': True}
    base.update(compression)
    (tmp_path / 'config.yaml').write_text(json.dumps({'auxiliary': {'compression': base}}))


def _session(db, session_id='vertical', profile='research'):
    db.create_session(session_id, 'cli', profile_name=profile)


def _source_rows(marker=OMITTED):
    return [{'role': 'user', 'content': f'Original turn 1 with the exact {marker} detail.'},
            {'role': 'assistant', 'content': 'Acknowledge turn 1.'},
            {'role': 'user', 'content': 'Original turn 3 unrelated prose.'},
            {'role': 'assistant', 'content': 'Final acknowledgement.'}]


def _summary_row(content='Controlled compaction summary.'):
    return {'role': 'assistant', 'content': content, '_compressed_summary': True}


def test_typed_carrier_with_drifted_source_refuses_publication_atomically(tmp_path):
    """Producer gap 1 (missing publication gate): a receipt presented through
    the trusted typed carrier must be verified INSIDE the real
    ``_execute_transcript_write`` transaction; source drift introduced after
    minting refuses the whole compaction and preserves the original rows."""
    from agent.compression_archive import capture, install_carrier

    path = tmp_path / 'drift.db'
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        reference = capture(db, 'vertical', 'research')
        carrier = _summary_row()
        install_carrier(carrier, reference)
        # Trusted drift: rewrite the archived source before publication.
        db._write_sql("UPDATE messages SET content='rewritten' "
                      "WHERE session_id='vertical' AND id=1")
        with pytest.raises(ValueError, match='changed or missing'):
            db.archive_and_compact('vertical', [copy.deepcopy(carrier)])
    with SessionDB(db_path=path) as reopened:
        with reopened._read_ctx() as conn:
            original = conn.execute(
                "SELECT content FROM messages WHERE session_id='vertical' AND id=1").fetchone()
            active = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id='vertical' AND active=1").fetchone()[0]
            summaries = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id='vertical' "
                "AND COALESCE(_compressed_summary,0)=1").fetchone()[0]
    assert 'rewritten' in original['content'], 'the prior committed drift stands'
    assert active == 4, 'refusal must leave every original row active (archive rolled back)'
    assert summaries == 0, 'refusal must publish no summary row'


def test_publication_mints_receipt_from_real_archived_source(tmp_path):
    """Trusted mint request → the published summary row carries a receipt
    captured from the actual archived source, resolvable after reopen."""
    from agent.compression_archive import (
        CARRIER_KEY, PREFIX, request_archive_mint, resolve)

    path = tmp_path / 'mint.db'
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        carrier = _summary_row()
        request_archive_mint([carrier])
        watermark = db.get_active_message_watermark('vertical')
        db.archive_and_compact('vertical', [carrier], watermark=watermark)
    with SessionDB(db_path=path) as reopened:
        with reopened._read_ctx() as conn:
            row = conn.execute(
                "SELECT display_metadata FROM messages WHERE session_id='vertical' "
                "AND COALESCE(_compressed_summary,0)=1 ORDER BY id DESC LIMIT 1").fetchone()
            archived = conn.execute(
                "SELECT content FROM messages WHERE session_id='vertical' "
                "AND active=0 AND compacted=1").fetchall()
        meta = json.loads(row['display_metadata'])
        reference = meta[CARRIER_KEY]
        assert isinstance(reference, str) and reference.startswith(PREFIX)
        rows = resolve(reopened, 'vertical', 'research', reference)
        assert any(OMITTED in r['content'] for r in rows), \
            'receipt coverage must include the real archived source'
        assert any(OMITTED in r['content'] for r in archived), \
            'the omitted detail lives in archived originals'


# ── Vertical slice: publication -> DB reopen -> ordinary construction -> bound recall ──


def _publish_with_receipt(db, *, session_id='vertical', profile='research'):
    """Real store publication with the trusted mint request (the exact shape
    the config-gated product seam produces)."""
    from agent.compression_archive import request_archive_mint
    _session(db, session_id, profile)
    db.append_messages_batch(session_id, copy.deepcopy(_source_rows()))
    carrier = _summary_row()
    request_archive_mint([carrier])
    watermark = db.get_active_message_watermark(session_id)
    db.archive_and_compact(session_id, [carrier], watermark=watermark)


def test_full_vertical_real_publication_reopen_ordinary_agent_bound_recall(tmp_path, monkeypatch):
    """The assigned vertical: publication mints from real archived source,
    the DB is closed and REOPENED, an ordinary AIAgent is CONSTRUCTED with
    that session (trusted owner DB/session/grants), and the real inline
    executor recalls the exact omitted source ONLY through the bound path."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'vertical.db'
    with SessionDB(db_path=path) as db:
        _publish_with_receipt(db)
    # Ordinary construction: real AIAgent on a custom route (no dispatch is
    # possible or attempted), bound to the reopened DB and the owned session.
    from run_agent import AIAgent
    with SessionDB(db_path=path) as reopened:
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='vertical', session_db=reopened)
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None, 'ordinary construction with carrier must bind from the typed carrier'
        adapter, _reference = binding
        assert adapter._db is reopened and adapter._session_id == 'vertical'
        assert 'session_search' in adapter._allowed_tools, 'grant must come from loaded tool surface'
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': OMITTED}, None))
        assert response['success'] is True, response
        assert OMITTED in json.dumps(response), 'bound recall must return the exact omitted source'
        del agent


def test_config_disabled_construction_stays_unbound(tmp_path, monkeypatch):
    """Profile/opt-in isolation: same durable carrier, config absent ->
    construction performs no binding and the inline route keeps accepted
    broad behavior."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path)  # no archive_receipts key
    path = tmp_path / 'disabled.db'
    with SessionDB(db_path=path) as db:
        _publish_with_receipt(db)
    from run_agent import AIAgent
    with SessionDB(db_path=path) as reopened:
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='vertical', session_db=reopened)
        assert getattr(agent, '_compression_archive_recall', None) is None, \
            'disabled config must leave the accepted surface unchanged'
        del agent


def test_missing_carrier_session_preserves_existing_behavior(tmp_path, monkeypatch):
    """Missing receipt on an old accepted session: config ENABLED, but the
    session never carried a receipt -> construction stays unbound and the
    ordinary reader still serves accepted broad search."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'legacy.db'
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_message('vertical', 'user', 'Ordinary session with PLAIN-LEGACY detail.')
    from run_agent import AIAgent
    with SessionDB(db_path=path) as reopened:
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='vertical', session_db=reopened)
        assert getattr(agent, '_compression_archive_recall', None) is None
        response = json.loads(_session_search_response(agent))
        assert response['success'] is True
        assert 'PLAIN-LEGACY' in json.dumps(response)
        del agent


def _session_search_response(agent):
    from agent.inline_tool_executors import _session_search
    return _session_search(agent, {'query': 'PLAIN-LEGACY'}, None)


def test_unresolvable_typed_carrier_retains_restricted_refusal(tmp_path, monkeypatch):
    """Invalid/unavailable presentation: a typed carrier whose bound source
    no longer resolves must bind the RESTRICTED refusal on ordinary
    construction (not leave unbound, not fall back broad)."""
    from agent.compression_archive import install_carrier
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'invalid.db'
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        from agent.compression_archive import capture
        reference = capture(db, 'vertical', 'research')
        carrier = _summary_row()
        install_carrier(carrier, reference)
        db.archive_and_compact('vertical', [carrier])
        # Trusted drift AFTER a successful accepted publication: the durable
        # carrier now binds a source that no longer exists as captured.
        db._write_sql("UPDATE messages SET content='drifted after publish' "
                      "WHERE session_id='vertical' AND active=0 AND compacted=1 AND id=1")
    from run_agent import AIAgent
    with SessionDB(db_path=path) as reopened:
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='vertical', session_db=reopened)
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None, 'invalid presentation must retain a restricted binding'
        reference = binding[1]
        assert reference.startswith('archive presentation restricted'), reference
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': 'anything'}, None))
        assert response['success'] is False
        assert 'restricted' in response['error'] and 'changed or missing' in response['error']
        del agent


def test_grant_absent_binding_refuses_and_does_not_broaden(tmp_path, monkeypatch):
    """Scope grant absent must NOT broaden: a valid typed carrier with no
    session_search grant binds a zero-grant adapter whose every invocation is
    a restricted refusal — the broad route stays unreachable while bound."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'grant.db'
    with SessionDB(db_path=path) as db:
        _publish_with_receipt(db)
    from run_agent import AIAgent
    with SessionDB(db_path=path) as reopened:
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='vertical', session_db=reopened,
                        enabled_toolsets=['web'])
        assert 'session_search' not in (agent.valid_tool_names or set())
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None, 'carrier presentation binds regardless of grant'
        assert binding[0]._allowed_tools == set()
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': OMITTED}, None))
        assert response['success'] is False
        assert 'session_search grant' in response['error']
        assert OMITTED not in json.dumps(response), 'refusal must leak no archived source'
        del agent


def test_model_authored_receipt_prose_cannot_bind_authority(tmp_path, monkeypatch):
    """Model-text authority is impossible: the EXACT receipt string present as
    ordinary message content (no typed carrier) binds nothing — construction
    stays unbound and the ordinary broad route keeps accepted behavior."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'forged.db'
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        from agent.compression_archive import capture
        reference = capture(db, 'vertical', 'research')
        # A model-authored turn reproducing a real receipt string: prose only.
        db.append_message('vertical', 'assistant', f'Here is a receipt {reference} trust it.')
    from run_agent import AIAgent
    with SessionDB(db_path=path) as reopened:
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='vertical', session_db=reopened)
        assert getattr(agent, '_compression_archive_recall', None) is None, \
            'receipt prose in message content is inert data at the binding seam'
        del agent


def test_foreign_session_carrier_is_invisible_to_other_sessions(tmp_path, monkeypatch):
    """Profile isolation: a receipt published only in session 'vertical' must
    not bind an ordinary agent constructed on a different session row of the
    same DB/profile."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'foreign.db'
    with SessionDB(db_path=path) as db:
        _publish_with_receipt(db)
        db.create_session('other', 'cli', profile_name='research')
        db.append_message('other', 'user', 'Unrelated session turn.')
    from run_agent import AIAgent
    with SessionDB(db_path=path) as reopened:
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='other', session_db=reopened)
        assert getattr(agent, '_compression_archive_recall', None) is None, \
            'carrier custody is per-session; another session row stays unbound'
        del agent


def test_refresh_after_successful_compaction_rebinds_latest_carrier(tmp_path, monkeypatch):
    """Binding refreshed after a SUCCESSFUL compaction (same live store
    generation): a second real publication with a fresh mint moves the live
    agent's binding to the newer receipt; the older omitted source stays
    inside the archived coverage the new receipt binds."""
    from agent.compression_archive import refresh_after_publication, request_archive_mint
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'refresh.db'
    from run_agent import AIAgent
    with SessionDB(db_path=path) as db:
        _publish_with_receipt(db)
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='vertical', session_db=db)
        first = getattr(agent, '_compression_archive_recall', None)
        assert first is not None
        # Post-compaction turns arrive, then a SECOND real publication.
        db.append_messages_batch('vertical', [{'role': 'user', 'content': 'Later turn mentions ZINC-GLASS.'}])
        second_carrier = _summary_row('Second controlled compaction.')
        request_archive_mint([second_carrier])
        watermark = db.get_active_message_watermark('vertical')
        db.archive_and_compact('vertical', [second_carrier], watermark=watermark)
        refresh_after_publication(agent, 'vertical')
        refreshed = getattr(agent, '_compression_archive_recall', None)
        assert refreshed is not None and refreshed[1] != first[1], \
            'binding must move to the newest typed carrier'
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': 'ZINC-GLASS'}, None))
        assert response['success'] is True and 'ZINC-GLASS' in json.dumps(response)
        del agent


def test_foreign_owner_carrier_refuses_via_owner_mismatch(tmp_path, monkeypatch):
    """A receipt captured under one owner presented under a DIFFERENT owner
    session row refuses at the resolver (owner mismatch) — the publication
    gate blocks it, so no carrier referencing another owner persists."""
    from agent.compression_archive import capture, install_carrier
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'xowner.db'
    with SessionDB(db_path=path) as db:
        _session(db, 'vertical', 'research')
        db.create_session('other', 'cli', profile_name='personal')
        db.append_messages_batch('other', copy.deepcopy(_source_rows()))
        foreign = capture(db, 'other', 'personal')
        carrier = _summary_row()
        install_carrier(carrier, foreign)
        # Refusal class depends on shape: owner mismatch when both snapshots
        # resolve, coverage-invalid when the presenting session has no rows
        # under the foreign bound. Either way, publication refuses.
        with pytest.raises(ValueError, match='owner mismatch|changed or missing|coverage invalid'):
            db.archive_and_compact('vertical', [copy.deepcopy(carrier)])
    with SessionDB(db_path=path) as reopened:
        with reopened._read_ctx() as conn:
            summaries = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id='vertical'").fetchone()[0]
        assert summaries == 0, 'foreign-owner publication must leave no rows'


# ── BD-1: mint-side budget cap must degrade, never abort publication ────────


def _degrade_to_budget_cap(monkeypatch):
    """Shrink the frozen source cap so the session's real row count exceeds
    it at mint time (the exact production condition: cumulative non-summary
    source crosses MAX_ROWS)."""
    import agent.compression_archive as archive
    monkeypatch.setattr(archive, 'MAX_ROWS', 3)


def test_mint_budget_cap_publishes_without_receipt_with_durable_unavailable_status(tmp_path, monkeypatch):
    """BD-1 approved policy: at the KNOWN BUDGET cap the compaction must
    still publish (archive + summary), carry NO valid receipt, and stamp a
    durable typed UNAVAILABLE status on the summary. The original archived
    source stays byte-exact; the only degraded fact is recall availability."""
    import agent.compression_archive as archive
    from agent.compression_archive import request_archive_mint

    _degrade_to_budget_cap(monkeypatch)
    path = tmp_path / 'bd1.db'
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        carrier = _summary_row()
        request_archive_mint([carrier])
        watermark = db.get_active_message_watermark('vertical')
        published = db.archive_and_compact('vertical', [carrier], watermark=watermark)
        assert published == 1, 'compaction itself must still publish at the cap'
        with db._read_ctx() as conn:
            original = conn.execute(
                "SELECT content,active,compacted FROM messages WHERE session_id='vertical' AND id=1"
            ).fetchone()
            row = conn.execute(
                "SELECT display_metadata FROM messages WHERE session_id='vertical' "
                "AND COALESCE(_compressed_summary,0)=1 ORDER BY id DESC LIMIT 1").fetchone()
    assert original['content'] == _source_rows()[0]['content'], \
        'the original archived source must be preserved exactly'
    assert (original['active'], original['compacted']) == (0, 1), 'source must be archived'
    meta = json.loads(row['display_metadata'])
    assert archive.CARRIER_KEY not in meta, 'no valid receipt may persist when the mint degraded'
    status = meta[archive.AVAILABILITY_KEY]
    assert status['state'] == 'unavailable', status
    assert status['reason_code'] == 'archive_source_budget_exceeded', status


def test_identity_conflict_still_refuses_publication_atomically_at_cap_degradation(tmp_path, monkeypatch):
    """BD-1 boundary: ONLY the known budget class may degrade. A clone
    identity conflict (real source corruption) must still refuse the whole
    publication atomically even though the mint path now catches budget
    errors."""
    from agent.compression_archive import request_archive_mint

    path = tmp_path / 'bd1-identity.db'
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        # Trusted corruption: re-key one archived row to another's uid with
        # different content -> clone identity conflict at snapshot time.
        db._write_sql("UPDATE messages SET message_uid = ("
                      "SELECT message_uid FROM messages WHERE session_id='vertical' AND id=1"
                      ") WHERE session_id='vertical' AND id=2")
        carrier = _summary_row()
        request_archive_mint([carrier])
        watermark = db.get_active_message_watermark('vertical')
        with pytest.raises(ValueError, match='clone identity conflict'):
            db.archive_and_compact('vertical', [carrier], watermark=watermark)
    with SessionDB(db_path=path) as reopened:
        with reopened._read_ctx() as conn:
            active = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id='vertical' AND active=1").fetchone()[0]
            summaries = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id='vertical' "
                "AND COALESCE(_compressed_summary,0)=1").fetchone()[0]
    assert active == 4 and summaries == 0, \
        'identity corruption must keep the atomic refusal (no degraded publish)'


def test_unavailable_status_binds_narrow_refusal_never_broad(tmp_path, monkeypatch):
    """Approved policy (BD-1 recall side): a durable machine-owned UNAVAILABLE
    status binds a NARROW refusal with a clear reason — never the broad
    fallback — even when the archived rows still contain the query. Invalid
    or foreign status shapes are NOT trusted availability facts: they leave
    the ordinary surface unchanged."""
    import agent.compression_archive as archive
    from agent.compression_archive import request_archive_mint

    _degrade_to_budget_cap(monkeypatch)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'bd1-bind.db'
    from run_agent import AIAgent
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        carrier = _summary_row()
        request_archive_mint([carrier])
        watermark = db.get_active_message_watermark('vertical')
        db.archive_and_compact('vertical', [carrier], watermark=watermark)
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='vertical', session_db=db)
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None, 'durable unavailable status must bind a restricted refusal'
        assert binding[0] is None
        assert binding[1].startswith('archive presentation restricted') and \
            'unavailable' in binding[1], binding[1]
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': OMITTED}, None))
        assert response['success'] is False, 'unavailable status must refuse narrowly'
        assert OMITTED not in json.dumps(response), 'narrow refusal leaks no archived source'
        del agent


def test_foreign_unavailable_status_shape_is_not_trusted(tmp_path, monkeypatch):
    """Machine-owned trust: a lookalike availability status written outside
    this module's writer (unknown reason code, via direct DB write) is never
    read as an availability FACT. A PRESENT but malformed typed key is a
    presentation fault: the recall surface stays a final restricted refusal
    — it never falls back broad, and never cites the foreign reason."""
    import agent.compression_archive as archive
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'bd1-foreign.db'
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        from agent.compression_archive import request_archive_mint
        carrier = _summary_row()
        request_archive_mint([carrier])
        watermark = db.get_active_message_watermark('vertical')
        db.archive_and_compact('vertical', [carrier], watermark=watermark)
        # A publication, then a foreign lookalike status stamped directly
        # onto the summary row's typed metadata (unknown reason code).
        db._write_sql(
            "UPDATE messages SET display_metadata = ? WHERE session_id='vertical' "
            "AND COALESCE(_compressed_summary,0)=1",
            (json.dumps({archive.AVAILABILITY_KEY: {'state': 'unavailable',
                                                    'reason_code': 'whatever-i-said-so'}}),))
    from run_agent import AIAgent
    with SessionDB(db_path=path) as reopened:
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='vertical', session_db=reopened)
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None, \
            'a present malformed typed key is a presentation fault: restricted, never broad'
        assert binding[0] is None
        assert binding[1].startswith('archive presentation restricted'), binding[1]
        assert 'whatever-i-said-so' not in binding[1], \
            'a foreign reason string is never quoted as an availability fact'
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': OMITTED}, None))
        assert response['success'] is False
        assert OMITTED not in json.dumps(response), 'refusal leaks no archived source'
        del agent


def test_transient_store_fault_fail_closed_then_recovered_broad_on_recall(tmp_path, monkeypatch):
    """BD-2 approved policy: a transient owner-store fault at construction
    binds the restricted marker (intended fail-closed retained under
    uncertainty), but a future recall runs a bounded recovery probe. Once the
    SAME owner/store positively establishes no typed key, the accepted broad
    route is restored — carrier-less sessions are not permanently refused."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'probe-fail.db'
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_message('vertical', 'user', 'Surviving ordinary turn PROBE-SAFE.')
    from run_agent import AIAgent
    with SessionDB(db_path=path) as reopened:
        original = reopened._read_ctx

        def broken_read_ctx():
            raise RuntimeError('simulated store outage')
        reopened._read_ctx = broken_read_ctx
        try:
            agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                            provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                            skip_context_files=True, skip_memory=True, max_iterations=1,
                            session_id='vertical', session_db=reopened)
        finally:
            del reopened._read_ctx
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None, 'an unavailable owner store must retain a restricted binding'
        assert len(binding) == 3 and binding[2] == 'store-unavailable', \
            'the transient fault must be tagged for bounded recovery'
        # Store recovered. One recall triggers the bounded probe; positive
        # no-carrier absence restores the accepted broad surface.
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': 'PROBE-SAFE'}, None))
        assert response['success'] is True, response
        assert 'PROBE-SAFE' in json.dumps(response)
        assert getattr(agent, '_compression_archive_recall', None) is None, \
            'positive absence clears the transient restriction'
        del agent


def test_transient_recovery_probe_never_clears_on_failed_probe(tmp_path, monkeypatch):
    """BD-2: a recovery probe that FAILS keeps the restriction exactly. While
    the store stays unavailable, repeated recalls all refuse narrowly and the
    probe remains bounded (attempts increment up to the cap, then stop)."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'probe-stay.db'
    from run_agent import AIAgent
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_message('vertical', 'user', 'Ordinary turn PROBE-STAY.')
        original = db._read_ctx

        def broken_read_ctx():
            raise RuntimeError('simulated store outage')
        db._read_ctx = broken_read_ctx
        try:
            agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                            provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                            skip_context_files=True, skip_memory=True, max_iterations=1,
                            session_id='vertical', session_db=db)
            binding = getattr(agent, '_compression_archive_recall', None)
            assert binding is not None and binding[2] == 'store-unavailable'
            from agent.inline_tool_executors import _session_search
            for _ in range(3):
                response = json.loads(_session_search(agent, {'query': 'PROBE-STAY'}, None))
                assert response['success'] is False
                assert 'restricted' in response['error']
            state = getattr(agent, '_compression_archive_recovery', None)
            assert isinstance(state, dict) and state['attempts'] >= 1, \
                'each bounded probe records its attempt'
            # Failed probes never clear, regardless of count.
            assert getattr(agent, '_compression_archive_recall', None) is binding
        finally:
            del db._read_ctx
        # Store recovered: force the bounded window open deterministically;
        # the NEXT probe may resolve broad — but only via a POSITIVE
        # observation.
        agent._compression_archive_recovery = {'attempts': 0, 'last': time.monotonic() - 60}
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': 'PROBE-STAY'}, None))
        assert response['success'] is True, response
        del agent


def test_transient_recovery_rebinds_recovered_carrier(tmp_path, monkeypatch):
    """BD-2: recovery with a RECOVERED carrier rebinds the narrow recall path
    — clearing is reserved for positively established ABSENCE."""
    import time
    from agent.compression_archive import request_archive_mint
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'probe-carrier.db'
    from run_agent import AIAgent
    with SessionDB(db_path=path) as db:
        _publish_with_receipt(db)  # real publication, valid carrier
        original = db._read_ctx

        def broken_read_ctx():
            raise RuntimeError('simulated store outage')
        db._read_ctx = broken_read_ctx
        try:
            agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                            provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                            skip_context_files=True, skip_memory=True, max_iterations=1,
                            session_id='vertical', session_db=db)
            binding = getattr(agent, '_compression_archive_recall', None)
            assert binding is not None and binding[2] == 'store-unavailable'
        finally:
            del db._read_ctx
        # Store recovered mid-lifetime; force the probe window open
        # deterministically (bounded recovery is time-backed off).
        agent._compression_archive_recovery = {'attempts': 0, 'last': time.monotonic() - 60}
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': OMITTED}, None))
        assert response['success'] is True, response
        rebound = getattr(agent, '_compression_archive_recall', None)
        assert rebound is not None and len(rebound) == 2, \
            'a recovered valid carrier rebinds the narrow path, not broad'
        assert OMITTED in json.dumps(response)
        del agent


def test_transient_recovery_runs_on_successful_refresh(tmp_path, monkeypatch):
    """BD-2 recovery seam 2: a SUCCESSFUL compaction refresh on a tagged
    restricted, carrier-less session runs the bounded probe, so the
    publication itself (which proves the store answered) restores behavior
    without waiting for a recall."""
    import time
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'probe-refresh.db'
    from agent.compression_archive import refresh_after_publication
    from run_agent import AIAgent
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_message('vertical', 'user', 'Legacy turn, no carrier ever.')
        original = db._read_ctx

        def broken_read_ctx():
            raise RuntimeError('simulated store outage')
        db._read_ctx = broken_read_ctx
        try:
            agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                            provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                            skip_context_files=True, skip_memory=True, max_iterations=1,
                            session_id='vertical', session_db=db)
        finally:
            del db._read_ctx
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None and binding[2] == 'store-unavailable'
        agent._compression_archive_recovery = {'attempts': 0, 'last': time.monotonic() - 60}
        refresh_after_publication(agent, 'vertical')
        assert getattr(agent, '_compression_archive_recall', None) is None, \
            'successful refresh + positive absence = transient restriction cleared'
        del agent


def test_unavailable_status_is_final_for_probes(tmp_path, monkeypatch):
    """BD-1/BD-2 interaction: an established-unavailable binding is NOT
    transient — recovery probes never clear it (they cannot manufacture a
    receipt). Only a NEWER valid carrier supersedes it."""
    import time
    import agent.compression_archive as archive
    from agent.compression_archive import request_archive_mint
    _degrade_to_budget_cap(monkeypatch)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _write_config(tmp_path, archive_receipts=True)
    path = tmp_path / 'final-unavailable.db'
    from run_agent import AIAgent
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        carrier = _summary_row()
        request_archive_mint([carrier])
        watermark = db.get_active_message_watermark('vertical')
        db.archive_and_compact('vertical', [carrier], watermark=watermark)
        agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                        provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=1,
                        session_id='vertical', session_db=db)
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None and len(binding) == 2, 'final binding, untagged'
        agent._compression_archive_recovery = {'attempts': 0, 'last': time.monotonic() - 60}
        from agent.compression_archive import attempt_transient_recovery_probe
        assert attempt_transient_recovery_probe(agent) is False, \
            'a final unavailable binding is never probed'
        assert getattr(agent, '_compression_archive_recall', None) is binding
        del agent


# NOTE (GXTD-814 BD-2, explicit approved policy correction): the previous
# test_producer_failure_never_blocks_ordinary_agent asserted PERMANENT
# narrow refusal after a transient store fault on a carrier-less session —
# i.e. the BD-2 defect itself. The approved policy keeps the fail-closed
# binding under uncertainty but requires bounded recovery; the transient
# family above (fail-closed-then-recovered / never-clears-on-failed-probe /
# rebinds-recovered-carrier / runs-on-successful-refresh) replaces it, and
# preserves the original guarantee that construction never raises.


# ── QG-4: proven-coverage (named-rows) publication with a minted receipt ───


def test_proved_coverage_named_rows_publication_mints_resolvable_receipt(tmp_path):
    """QG-4: the covered_ids branch (_proved_coverage -> _archive_named_rows)
    publishes with a minted receipt and stamps the summary _row_id in BOTH
    insert paths; the receipt resolves after reopen with real coverage."""
    from agent.compression_archive import CARRIER_KEY, PREFIX, request_archive_mint, resolve

    path = tmp_path / 'named.db'
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        with db._read_ctx() as conn:
            covered_ids = [r[0] for r in conn.execute(
                "SELECT id FROM messages WHERE session_id='vertical' AND active=1 ORDER BY id")]
        carrier = _summary_row()
        request_archive_mint([carrier])
        watermark = db.get_active_message_watermark('vertical')
        published = db.archive_and_compact(
            'vertical', [carrier], watermark=watermark, covered_ids=covered_ids)
        assert published == 1
    with SessionDB(db_path=path) as reopened:
        with reopened._read_ctx() as conn:
            row = conn.execute(
                "SELECT display_metadata FROM messages WHERE session_id='vertical' "
                "AND COALESCE(_compressed_summary,0)=1 ORDER BY id DESC LIMIT 1").fetchone()
        reference = json.loads(row['display_metadata'])[CARRIER_KEY]
        assert reference.startswith(PREFIX)
        rows = resolve(reopened, 'vertical', 'research', reference)
        assert any(OMITTED in r['content'] for r in rows), \
            'named-rows publication must still mint from the real archived source'


# ── QG-5 / QG-6: failed-publication rollback of mint marker and metadata ───


def test_failed_publication_rolls_back_persisted_metadata_but_keeps_retry_markers(tmp_path):
    """QG-5/QG-6 rollback contract: a publication that fails AFTER the
    in-transaction install/refresh leaves NO durable metadata (the whole txn
    rolls back), while the caller dict keeps its mint marker (a later
    publication may legitimately mint — the store answers the trusted marker,
    never config) and the previously PUBLISHED metadata row is untouched.
    Retry with the same dicts re-mints and publishes normally."""
    from agent.compression_archive import CARRIER_KEY, PREFIX, install_carrier, request_archive_mint

    path = tmp_path / 'rollback.db'
    stale_reference = None
    with SessionDB(db_path=path) as db:
        _session(db)
        db.append_messages_batch('vertical', copy.deepcopy(_source_rows()))
        stale_reference = install_capture = None
        from agent.compression_archive import capture
        stale_reference = capture(db, 'vertical', 'research')
        carrier = _summary_row()
        install_carrier(carrier, stale_reference)

        def failing_attempt(conn):
            # Simulate a crash AFTER a hypothetical install: install fresh
            # metadata, then die — the txn must erase the durable effect.
            install_carrier(carrier, stale_reference)
            conn.execute("UPDATE messages SET display_metadata = 'poisoned' "
                         "WHERE session_id='vertical' AND id=1")
            raise RuntimeError('late transaction crash')
        with pytest.raises(RuntimeError, match='late transaction crash'):
            db._execute_write(failing_attempt)
        with db._read_ctx() as conn:
            row = conn.execute(
                "SELECT display_metadata FROM messages WHERE session_id='vertical' AND id=1"
            ).fetchone()
        assert row['display_metadata'] is None, \
            'the rolled-back txn must leave no durable metadata'
        # The trusted mint marker survives rollback so a retry may mint
        # (documented store semantics: the marker is the caller's request).
        request_archive_mint([carrier])
        watermark = db.get_active_message_watermark('vertical')
        # Presenting the stale capture while source is unchanged succeeds
        # pre- and post-body; the mint installs the fresh receipt over it.
        db.archive_and_compact('vertical', [carrier], watermark=watermark)
    with SessionDB(db_path=path) as reopened:
        with reopened._read_ctx() as conn:
            row = conn.execute(
                "SELECT display_metadata FROM messages WHERE session_id='vertical' "
                "AND COALESCE(_compressed_summary,0)=1 ORDER BY id DESC LIMIT 1").fetchone()
            originals = conn.execute(
                "SELECT display_metadata FROM messages WHERE session_id='vertical' AND id=1"
            ).fetchone()
        assert 'poisoned' not in (row['display_metadata'] or '')
        assert originals['display_metadata'] is None, \
            'original rows keep no metadata across the failed publication'
        assert json.loads(row['display_metadata'])[CARRIER_KEY].startswith(PREFIX)


# ── QG-1 / QG-2 / QG-7: the ACTUAL _commit_compaction production seam ──────


def _seam_agent(tmp_path, monkeypatch, path, *, archive_receipts):
    """An ordinary AIAgent + real ContextCompressor with a CONTROLLED summary
    adapter (never a native result) on a nonfunctional route, bound to a real
    SessionDB with a real persisted transcript — the exact construction the
    641 committed harness uses; zero native dispatch is possible."""
    from agent.context_compressor import (
        ContextCompressor, HISTORICAL_TASK_HEADING, _NO_USER_TASK_SENTINEL)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    # Exercise the adopted strategy, not the older source-custody selector
    # whose typed-output contract this controlled prose adapter cannot meet.
    extra = {'summary_strategy': 'best_effort'}
    if archive_receipts:
        extra['archive_receipts'] = True
    _write_config(tmp_path, **extra)
    from run_agent import AIAgent
    db = SessionDB(db_path=path)
    db.create_session('vertical', 'cli', profile_name='research')
    transcript = [{'role': 'system', 'content': 'Keep historical data distinct from live instructions.'}]
    for i in range(12):
        transcript.append({'role': 'user',
                           'content': f'Round {i}.\n' + 'archive detail\n' * 5})
        transcript.append({'role': 'assistant',
                           'content': (f'Result {i}.\n' + ('work detail ' + OMITTED + '\n' if i == 0
                                                          else 'routine work\n') * 80)})
    db.append_messages_batch('vertical', copy.deepcopy(transcript))
    compressor = ContextCompressor(model='fixture', config_context_length=100000, quiet_mode=True,
                                   protect_first_n=1, protect_last_n=2, abort_on_summary_failure=True)
    compressor.tail_token_budget = 10
    reply = f'{HISTORICAL_TASK_HEADING}\n{_NO_USER_TASK_SENTINEL}\n\n## Overview\nSeam summary.'
    monkeypatch.setattr(compressor, '_call_summary_llm', lambda *_: reply)
    agent = AIAgent(api_key='***', base_url='http://127.0.0.1:9/v1',
                    provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                    skip_context_files=True, skip_memory=True, max_iterations=1,
                    session_id='vertical', session_db=db)
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = 'Existing system prompt.'
    agent.compression_in_place = True
    return db, agent


def _summary_metadata(db):
    with db._read_ctx() as conn:
        rows = conn.execute(
            "SELECT id,display_metadata FROM messages WHERE session_id='vertical' "
            "AND COALESCE(_compressed_summary,0)=1 ORDER BY id").fetchall()
    return {r['id']: (json.loads(r['display_metadata']) if r['display_metadata'] else {})
            for r in rows}


def test_real_commit_compaction_seam_mints_persists_and_refreshes(tmp_path, monkeypatch):
    """QG-1 (production seam): the ACTUAL _commit_compaction path — via
    agent._compress_context — with archive_receipts enabled requests the
    mint, publishes, PERSISTS a resolvable carrier on the summary row, and
    the post-publication refresh binds the live agent to it. Non-native
    summary fixture only."""
    from agent.compression_archive import CARRIER_KEY, PREFIX, resolve
    path = tmp_path / 'seam1.db'
    db, agent = _seam_agent(tmp_path, monkeypatch, path, archive_receipts=True)
    try:
        original = db.get_messages_as_conversation('vertical')
        result, _ = agent._compress_context(copy.deepcopy(original), 'Existing system prompt.',
                                            force=True)
        assert result != original, 'the controlled compressor must produce a compaction'
        meta = _summary_metadata(db)
        assert meta, 'the real commit path must publish a summary row'
        reference = next((m[CARRIER_KEY] for m in meta.values() if CARRIER_KEY in m), None)
        assert reference is not None and reference.startswith(PREFIX), \
            'the seam must mint and persist the receipt at publication'
        rows = resolve(db, 'vertical', 'research', reference)
        assert any(OMITTED in r['content'] for r in rows), \
            'receipt coverage must include the real archived source'
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None and binding[1] == reference, \
            'the post-publication refresh at the seam must bind the new carrier'
        # Post-refresh recall works through the actual inline route.
        from agent.inline_tool_executors import _session_search
        response = json.loads(_session_search(agent, {'query': OMITTED}, None))
        assert response['success'] is True and OMITTED in json.dumps(response)
    finally:
        db.close()


def test_real_commit_compaction_second_cycle_carries_old_presented_carrier(tmp_path, monkeypatch):
    """QG-2 (production seam): a SECOND real compaction whose candidate list
    carries the cycle-1 summary dict WITH its installed typed carrier — the
    store gate validates the re-presented carrier inside the transaction and
    the newer mint becomes the newest trusted carrier."""
    from agent.compression_archive import CARRIER_KEY, PREFIX, resolve
    path = tmp_path / 'seam2.db'
    db, agent = _seam_agent(tmp_path, monkeypatch, path, archive_receipts=True)
    try:
        first = db.get_messages_as_conversation('vertical')
        after1, _ = agent._compress_context(copy.deepcopy(first), 'Existing system prompt.',
                                            force=True)
        assert after1 != first
        meta1 = _summary_metadata(db)
        first_ref = next((m[CARRIER_KEY] for m in meta1.values() if CARRIER_KEY in m), None)
        assert first_ref and first_ref.startswith(PREFIX)
        # Reload (durable projection), add a turn, and compact AGAIN through
        # the same seam: the cycle-1 carrier rides forward on the summary
        # dict carried into the new candidate.
        db.append_message('vertical', 'user', 'Later seam turn ZINC-GLASS.')
        second = db.get_messages_as_conversation('vertical')
        assert any(isinstance(m.get('display_metadata'), dict)
                   and CARRIER_KEY in m['display_metadata'] for m in second), \
            'the durable typed carrier must project back onto the reloaded summary dict'
        # Confirm the store gate actually sees a PRESENTED carrier this time
        # by watching the resolve path: with a valid carrier no failure is
        # expected, so instead assert the RESULT semantics.
        after2, _ = agent._compress_context(copy.deepcopy(second), 'Existing system prompt.',
                                            force=True)
        assert after2 != second
        meta2 = _summary_metadata(db)
        refs = [m[CARRIER_KEY] for m in meta2.values() if CARRIER_KEY in m]
        assert len(refs) == 2, 'both cycles persist their own carriers'
        assert refs[-1] != first_ref, 'the second cycle must mint a NEWER receipt'
        rows = resolve(db, 'vertical', 'research', refs[-1])
        assert any('ZINC-GLASS' in r['content'] for r in rows) and \
            any(OMITTED in r['content'] for r in rows), \
            'the newest receipt covers both cycles (cumulative source)'
        binding = getattr(agent, '_compression_archive_recall', None)
        assert binding is not None and binding[1] == refs[-1], \
            'the live binding must move to the newest carrier after the second real commit'
    finally:
        db.close()

    # The second publication must survive a new DB connection and agent,
    # not merely the committing process's in-memory binding.
    from run_agent import AIAgent
    from agent.inline_tool_executors import _session_search
    with SessionDB(db_path=path) as reopened:
        resumed = AIAgent(api_key='fixture', base_url='http://127.0.0.1:9/v1',
                          provider='custom', model='gxtd814-vertical-model', quiet_mode=True,
                          skip_context_files=True, skip_memory=True, max_iterations=1,
                          session_id='vertical', session_db=reopened)
        assert resumed._compression_archive_recall[1] == refs[-1]
        response = json.loads(_session_search(resumed, {'query': OMITTED}, None))
        assert response['success'] is True and OMITTED in json.dumps(response)
        # Fresh protected-tail content remains live, not an archived search
        # hit: archive recall intentionally searches compacted originals only.
        assert any('ZINC-GLASS' in str(m.get('content', ''))
                   for m in reopened.get_messages_as_conversation('vertical'))


def test_real_commit_compaction_config_disabled_stays_carrier_free(tmp_path, monkeypatch):
    """Config-disabled production seam: the real _commit_compaction path
    publishes with ZERO typed carrier keys and leaves the agent unbound —
    the accepted publication shape, unchanged."""
    import agent.compression_archive as archive
    path = tmp_path / 'seam-off.db'
    db, agent = _seam_agent(tmp_path, monkeypatch, path, archive_receipts=False)
    try:
        original = db.get_messages_as_conversation('vertical')
        result, _ = agent._compress_context(copy.deepcopy(original), 'Existing system prompt.',
                                            force=True)
        assert result != original, 'accepted compaction still works with the opt-in absent'
        meta = _summary_metadata(db)
        assert meta, 'the accepted path still publishes a summary row'
        for row_meta in meta.values():
            assert archive.CARRIER_KEY not in row_meta
            assert archive.AVAILABILITY_KEY not in row_meta
        assert getattr(agent, '_compression_archive_recall', None) is None
    finally:
        db.close()


def test_commit_compaction_snapshots_receipts_config_once(tmp_path, monkeypatch):
    """QG-7: the production seam must snapshot the opt-in config ONCE per
    compaction (mint request and refresh from one decision), never via two
    independent reads that can disagree mid-commit."""
    import agent.conversation_compression as cc
    path = tmp_path / 'seam-cfg.db'
    db, agent = _seam_agent(tmp_path, monkeypatch, path, archive_receipts=True)
    calls = {'n': 0}
    original_enabled = cc._archive_receipts_enabled

    def counting_enabled():
        calls['n'] += 1
        return original_enabled()
    monkeypatch.setattr(cc, '_archive_receipts_enabled', counting_enabled)
    try:
        original = db.get_messages_as_conversation('vertical')
        result, _ = agent._compress_context(copy.deepcopy(original), 'Existing system prompt.',
                                            force=True)
        assert result != original
        assert calls['n'] == 1, \
            f'one compaction commit must read the opt-in exactly once, saw {calls["n"]}'
    finally:
        db.close()
