"""Versioned archive receipts and explicitly granted same-owner session recall.

Isolated integration primitive, not a compressor strategy or authorization source.
The caller binds the already-authorized SessionDB, session, profile and tool grant.

GXTD-814 additions: the publication gate. A receipt is authority only through
this module's typed surface — the ``display_metadata`` carrier key, never
summary prose — and the store verifies every presented carrier inside its own
``_execute_transcript_write`` transaction before publishing.
"""
import hashlib
import json

PREFIX = 'HERMES_CONVERSATION_ARCHIVE_V1 '
MAX_ROWS = 4096
MAX_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 8192
# Typed carrier metadata key on a compaction summary row (structured, never prose).
CARRIER_KEY = 'compression_archive_receipt'
# Caller-dict-only trusted mint request; never a persisted schema key.
MINT_REQUEST_KEY = '_archive_receipt_mint'
# Poison binding marker: a present-but-invalid typed receipt refuses narrowly
# instead of falling back to the broad search path.
REJECTION_PREFIX = 'archive presentation restricted: '
# Typed machine-owned recall-availability status on a compaction summary row.
# Only this module's writers persist it and only this module's readers trust
# it; a degraded mint records the ONLY degraded fact (availability) without
# claiming anything about coverage completeness.
AVAILABILITY_KEY = 'compression_archive_recall_availability'
AVAILABILITY_UNAVAILABLE = 'unavailable'
AVAILABILITY_REASONS = frozenset({'archive_source_budget_exceeded'})


class ArchiveSourceBudgetError(ValueError):
    """The KNOWN source budget cap (frozen MAX_ROWS/MAX_BYTES) was exceeded.
    The only mint-side failure class approved to degrade (never abort) the
    compaction; identity, owner, and store faults keep the atomic refusal."""


class ArchiveStoreUnavailable(ValueError):
    """Transient owner-store fault (BD-2 class): the binding refused because
    the store could not answer — fail-closed, but a later bounded probe may
    clear the restriction on a POSITIVE observation from the same
    owner/store. Never cleared by a failed probe."""


class ArchiveAvailabilityUnavailable(ValueError):
    """Durable machine-owned typed status establishes archive recall
    unavailable (mint degraded at the known budget cap). Final restricted
    binding for the recall surface — no recovery probe can manufacture a
    receipt — until a NEWER carrier positively re-establishes it."""


# Bounded recovery for transient store faults (BD-2): each future recall or
# successful refresh may trigger at most one probe per backoff interval;
# RECOVERY_MAX_ATTEMPTS total. Expiry only stops auto-retry; it NEVER clears
# the restriction.
RECOVERY_PROBE_BACKOFF_S = 2.0
RECOVERY_MAX_ATTEMPTS = 10


def _read_latest_row(db, session_id, key):
    """Newest row of THIS session carrying a typed key in display_metadata."""
    with db._read_ctx() as conn:
        return conn.execute(
            'SELECT id,display_metadata FROM messages WHERE session_id=? '
            'AND (active=1 OR compacted=1) AND display_metadata IS NOT NULL '
            # json_extract conflates JSON null with missing. json_type
            # preserves present-null as 'null', so validation can refuse it.
            "AND json_type(CASE WHEN json_valid(display_metadata) "
            "THEN display_metadata ELSE '{}' END, ?) IS NOT NULL "
            'ORDER BY id DESC LIMIT 1', (session_id, '$.' + key)).fetchone()


def _decode_meta(row):
    raw = row['display_metadata']
    try:
        meta = json.loads(raw, object_pairs_hook=_pairs) if isinstance(raw, str) else raw
    except (TypeError, ValueError, RecursionError):
        raise ValueError('archive presentation restricted: carrier metadata malformed')
    if not isinstance(meta, dict):
        raise ValueError('archive presentation restricted: carrier metadata malformed')
    return meta


def validate_availability_status(status):
    """Exact grammar of our machine-owned status; a foreign shape (unknown
    reason, extra/missing fields, wrong types) is NOT an availability fact
    and never reaches an availability decision."""
    if not isinstance(status, dict) or set(status) != {'state', 'reason_code'}:
        raise ValueError('archive availability status malformed')
    if not isinstance(status['state'], str) or status['state'] != AVAILABILITY_UNAVAILABLE:
        raise ValueError('archive availability status malformed')
    if not isinstance(status['reason_code'], str) or status['reason_code'] not in AVAILABILITY_REASONS:
        raise ValueError('archive availability reason unknown')
    return status



def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate archive receipt field')
        result[key] = value
    return result


def _matches_source_generation(original, generation):
    """Recognize only the byte-exact original or its known replay projection.

    SessionDB replay strips boundary whitespace from user/assistant strings
    while retaining their logical UID. Keep the earliest raw original in the
    digest; never normalize both sides, elect a later clone, or accept role,
    interior-text, tool-body, or arbitrary summarizer changes as originals.
    """
    if generation['role'] != original['role']:
        return False
    if generation['content'] == original['content']:
        return True
    return (original['role'] in ('user', 'assistant')
            and generation['content'] == original['content'].strip())


def _snapshot(db, session_id, profile, through=None):
    if not isinstance(session_id, str) or not session_id or len(session_id)>128:
        raise ValueError('archive owner session invalid')
    if not isinstance(profile, str) or not profile or len(profile)>128:
        raise ValueError('archive owner profile invalid')
    try:
        with db._read_ctx() as connection:
            connection.execute('SAVEPOINT conversation_archive')
            try:
                owner=connection.execute('SELECT profile_name FROM sessions WHERE id=?',(session_id,)).fetchone()
                if owner is None or owner['profile_name']!=profile:
                    raise ValueError('archive owner mismatch')
                if through is None:
                    through=connection.execute('SELECT COALESCE(MAX(id),0) FROM messages WHERE session_id=?',(session_id,)).fetchone()[0]
                rows=connection.execute(
                    'SELECT id,message_uid,role,active,compacted, '
                    'CASE WHEN length(CAST(content AS BLOB))<=? THEN content END AS content, '
                    'length(CAST(content AS BLOB)) AS source_bytes FROM messages '
                    'WHERE session_id=? AND id<=? AND COALESCE(_compressed_summary,0)=0 '
                    # Frozen source definition (GXTD-814): regenerated
                    # system-prompt scaffolding is NOT conversational source.
                    # The live system row is rewritten (compression note) on
                    # every compaction under its stable uid, so including it
                    # would make the SECOND real cycle's cumulative snapshot
                    # a false identity conflict. Conversational-row uid/
                    # content conflicts remain a hard refusal.
                    "AND COALESCE(role,'')<>'system' "
                    'ORDER BY id LIMIT ?', (MAX_BYTES,session_id,through,MAX_ROWS+1)).fetchall()
            finally:
                connection.execute('RELEASE conversation_archive')
    except ValueError:
        raise
    except Exception as error:
        raise ValueError('archive store unavailable') from error
    if not rows:
        raise ValueError('archive source budget or coverage invalid')
    if len(rows) > MAX_ROWS or sum(r['source_bytes'] or 0 for r in rows) > MAX_BYTES:
        # BD-1 truthful recall semantics: exceeding the KNOWN frozen cap is a
        # distinct, nameable class. Callers keep identical fail-closed
        # behavior (this is a ValueError), but the recall surface can state
        # the honest reason instead of a generic mismatch.
        raise ArchiveSourceBudgetError('archive source budget or coverage invalid')

    unique={}
    for row in rows:
        if not isinstance(row['message_uid'],str) or not row['message_uid'] or not isinstance(row['content'],str):
            raise ValueError('archive source identity unavailable')
        uid=row['message_uid']
        if uid in unique:
            previous=unique[uid]
            if not _matches_source_generation(previous, row):
                raise ValueError('archive clone identity conflict')
        else:
            unique[uid]=dict(row)
    sources=[{key:row[key] for key in ('message_uid','role','content')} for row in unique.values()]
    digest=hashlib.sha256(_json({'session_id':session_id,'profile':profile,'sources':sources}).encode('ascii')).hexdigest()
    value={'session_id':session_id,'profile':profile,'first':rows[0]['id'],'through':through,'count':len(sources),'sha256':digest}
    return value,[dict(row) for row in rows]


def capture(db, session_id, profile):
    """Capture fixed original coverage; later appends do not invalidate it."""
    value,_rows=_snapshot(db,session_id,profile)
    reference=PREFIX+_json(value)
    if len(reference.encode())>1024:
        raise ValueError('archive receipt too large')
    return reference


def resolve(db, session_id, profile, reference):
    """Resolve only source identity, never grant access from receipt contents."""
    if not isinstance(reference,str) or not reference.startswith(PREFIX) or len(reference.encode())>1024:
        raise ValueError('archive receipt invalid')
    try:
        value=json.loads(reference[len(PREFIX):],object_pairs_hook=_pairs)
        if not isinstance(value,dict) or set(value)!={'session_id','profile','first','through','count','sha256'}:
            raise ValueError('archive receipt fields invalid')
        if value['session_id']!=session_id or value['profile']!=profile:
            raise ValueError('archive owner mismatch')
        if any(type(value[k]) is not int or value[k]<=0 for k in ('first','through','count')) or value['first']>value['through']:
            raise ValueError('archive receipt coverage invalid')
        if not isinstance(value['sha256'],str) or len(value['sha256'])!=64 or reference!=PREFIX+_json(value):
            raise ValueError('archive receipt encoding invalid')
        actual,rows=_snapshot(db,session_id,profile,value['through'])
        if actual!=value:
            raise ValueError('archive original source changed or missing')
        return rows
    except (TypeError,KeyError,RecursionError) as error:
        raise ValueError('archive receipt malformed') from error


class ArchiveRecall:
    """Bind trusted owner/grants once; model arguments cannot choose another DB."""

    def __init__(self, db, session_id, profile, *, allowed_tools):
        self._db=db
        self._session_id=session_id
        self._profile=profile
        self._allowed_tools=frozenset(allowed_tools)

    def invoke(self, reference, arguments):
        if 'session_search' not in self._allowed_tools:
            raise ValueError('archive recall requires session_search grant')
        if not isinstance(arguments,dict) or set(arguments)-{'query','session_id','profile'}:
            raise ValueError('archive recall arguments invalid')
        if arguments.get('session_id',self._session_id)!=self._session_id or arguments.get('profile',self._profile)!=self._profile:
            raise ValueError('archive recall owner override refused')
        query=arguments.get('query')
        if not isinstance(query,str) or not query.strip() or len(query.encode())>256:
            raise ValueError('archive recall query invalid')
        rows=resolve(self._db,self._session_id,self._profile,reference)
        matches=[r for r in rows if query.casefold() in r['content'].casefold() and r['active']==0 and r['compacted']==1]
        if not matches:
            return _json({'success':True,'mode':'archive-recall','messages':[]})
        # Use the real session tool; never accept a caller-supplied profile DB.
        from tools.session_search_tool import session_search
        from agent.context_compressor import _redact_compaction_text
        result=session_search(db=self._db,current_session_id=self._session_id,session_id=self._session_id,
                              around_message_id=matches[0]['id'],window=1)
        resolve(self._db,self._session_id,self._profile,reference)
        if len(result.encode())>MAX_RESULT_BYTES:
            raise ValueError('archive recall result budget exceeded')
        parsed=json.loads(result)
        if not parsed.get('success'):
            raise ValueError('archive session reader refused')
        # Sanitize source-bearing response strings, preserving JSON structure.
        def sanitize(value):
            if isinstance(value,str):
                return _redact_compaction_text(value)
            if isinstance(value,list):
                return [sanitize(item) for item in value]
            if isinstance(value,dict):
                return {key:sanitize(item) for key,item in value.items()}
            return value
        output=_json(sanitize(parsed))
        if len(output.encode())>MAX_RESULT_BYTES:
            raise ValueError('archive recall result budget exceeded')
        return output


# ── GXTD-814: trusted publication gate and resume binding ────────────────────
#
# Authority model: the ONLY trusted receipt entry points are (a) the
# caller-dict mint request the store answers inside its own transaction from
# the REAL rows being archived, and (b) the typed ``display_metadata`` carrier
# the store reads back on resume. Summary prose is never parsed as a receipt
# source; a model-authored string can never mint or present authority.


def request_archive_mint(messages):
    """Trusted opt-in: mark the compaction candidate dicts so the store mints
    a receipt from the actual archived source INSIDE its own transaction and
    installs it on the published summary row's typed metadata. The request
    marker itself is never persisted (it is not a schema column and every
    write path serializes only known columns)."""
    if not isinstance(messages, list) or not messages:
        raise ValueError('archive mint request requires compaction messages')
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError('archive mint request requires message dicts')
        message[MINT_REQUEST_KEY] = True


def mint_for_publication(db, conn, session_id, profile, through):
    """Capture a receipt for the session's non-summary source as it exists on
    ``conn`` INSIDE the publication transaction. ``through`` is the
    pre-publication MAX(id) of the session: the freshly archived originals and
    any pre-existing rows are addressable, while the just-inserted summary row
    and watermark clones above it can never leak into coverage. Coverage is
    keyed by message uid, so re-sequenced clones stay resolvable (frozen
    resolver semantics)."""
    if not isinstance(profile, str) or not profile:
        raise ValueError('archive mint requires the owning profile')
    _value, _rows = _snapshot_on_conn(conn, session_id, profile, through)
    return PREFIX + _json(_value)


def _snapshot_on_conn(conn, session_id, profile, through=None):
    """Same frozen coverage/resolution semantics as ``_snapshot``, on the
    connection already held by the caller's transaction (no new SAVEPOINT:
    the publication txn owns atomicity)."""
    owner = conn.execute('SELECT profile_name FROM sessions WHERE id=?', (session_id,)).fetchone()
    if owner is None or owner['profile_name'] != profile:
        raise ValueError('archive owner mismatch')
    if through is None:
        through = conn.execute(
            'SELECT COALESCE(MAX(id),0) FROM messages WHERE session_id=?', (session_id,)).fetchone()[0]
    rows = conn.execute(
        'SELECT id,message_uid,role,active,compacted, '
        'CASE WHEN length(CAST(content AS BLOB))<=? THEN content END AS content, '
        'length(CAST(content AS BLOB)) AS source_bytes FROM messages '
        'WHERE session_id=? AND id<=? AND COALESCE(_compressed_summary,0)=0 '
        # Same frozen source definition as ``_snapshot``: regenerated system-
        # prompt scaffolding rows are excluded (they re-render under a stable
        # uid every compaction); conversational conflicts still refuse.
        "AND COALESCE(role,'')<>'system' "
        'ORDER BY id LIMIT ?', (MAX_BYTES, session_id, through, MAX_ROWS + 1)).fetchall()
    if not rows:
        raise ValueError('archive source budget or coverage invalid')
    if len(rows) > MAX_ROWS or sum(r['source_bytes'] or 0 for r in rows) > MAX_BYTES:
        # GXTD-814 BD-1: the ONLY mint-side class approved to degrade the
        # receipt surface instead of aborting the compaction. Callers that
        # must keep the atomic refusal (identity/owner integrity, resolve)
        # discriminate this exact type.
        raise ArchiveSourceBudgetError('archive source budget or coverage invalid')

    unique = {}
    for row in rows:
        if not isinstance(row['message_uid'], str) or not row['message_uid'] or not isinstance(row['content'], str):
            raise ValueError('archive source identity unavailable')
        uid = row['message_uid']
        if uid in unique:
            previous = unique[uid]
            if not _matches_source_generation(previous, row):
                raise ValueError('archive clone identity conflict')
        else:
            unique[uid] = dict(row)
    sources = [{key: row[key] for key in ('message_uid', 'role', 'content')} for row in unique.values()]
    digest = hashlib.sha256(_json({'session_id': session_id, 'profile': profile, 'sources': sources}).encode('ascii')).hexdigest()
    value = {'session_id': session_id, 'profile': profile, 'first': rows[0]['id'], 'through': through,
             'count': len(sources), 'sha256': digest}
    return value, [dict(row) for row in rows]


def validate_carriers(messages):
    """Extract every typed receipt presented by the candidate (never prose).
    A malformed typed carrier is a presentation fault at the trust boundary:
    it refuses BEFORE any durable effect, preserving the original rows."""
    presented = []
    for message in messages or ():
        if not isinstance(message, dict):
            continue
        meta = message.get('display_metadata')
        if isinstance(meta, str):
            try:
                meta = json.loads(meta, object_pairs_hook=_pairs)
            except (TypeError, ValueError, RecursionError):
                raise ValueError('archive carrier metadata malformed')
        if meta is None:
            continue
        if not isinstance(meta, dict):
            raise ValueError('archive carrier metadata malformed')
        if AVAILABILITY_KEY in meta:
            validate_availability_status(meta[AVAILABILITY_KEY])
        if CARRIER_KEY not in meta:
            continue
        reference = meta[CARRIER_KEY]
        if not isinstance(reference, str):
            raise ValueError('archive receipt carrier must be text')
        validate_reference(reference)
        presented.append(reference)
    return presented


def validate_reference(reference):
    """Grammar-only validation (never source resolution): an externally
    presented string must parse under the frozen v1 grammar to reach the
    in-transaction resolver at all."""
    if not isinstance(reference, str) or not reference.startswith(PREFIX) or len(reference.encode()) > 1024:
        raise ValueError('archive receipt invalid')
    try:
        value = json.loads(reference[len(PREFIX):], object_pairs_hook=_pairs)
        if not isinstance(value, dict) or set(value) != {'session_id', 'profile', 'first', 'through', 'count', 'sha256'}:
            raise ValueError('archive receipt fields invalid')
        if any(type(value[k]) is not int or value[k] <= 0 for k in ('first', 'through', 'count')) or value['first'] > value['through']:
            raise ValueError('archive receipt coverage invalid')
        if not isinstance(value['sha256'], str) or len(value['sha256']) != 64 or reference != PREFIX + _json(value):
            raise ValueError('archive receipt encoding invalid')
    except (TypeError, KeyError, RecursionError) as error:
        raise ValueError('archive receipt malformed') from error
    return value


def install_carrier(message, reference):
    """Install the minted receipt on the summary dict's typed metadata (the
    only persisted carrier shape); never mutates ``content``."""
    meta = _normalize_metadata(message)
    meta[CARRIER_KEY] = reference
    message['display_metadata'] = meta


def degrade_carrier(message):
    """GXTD-814 BD-1 approved degradation: remove any typed carrier and
    stamp the machine-owned UNAVAILABLE status on the summary dict's typed
    metadata. Truthful semantics: the status records ONLY that archive recall
    is unavailable and why (reason code); it claims nothing about coverage
    completeness. Machine-owned — a degraded publication never keeps a
    verifiable receipt, and prose can never write or clear this key at any
    ordinary path."""
    meta = _normalize_metadata(message)
    meta.pop(CARRIER_KEY, None)
    meta[AVAILABILITY_KEY] = {'state': AVAILABILITY_UNAVAILABLE,
                             'reason_code': 'archive_source_budget_exceeded'}
    message['display_metadata'] = meta


def _normalize_metadata(message):
    meta = message.get('display_metadata')
    if isinstance(meta, str):
        try:
            meta = json.loads(meta, object_pairs_hook=_pairs)
        except (TypeError, ValueError, RecursionError):
            raise ValueError('archive carrier metadata malformed')
    return dict(meta) if isinstance(meta, dict) else {}


def carrier_is_unavailable(message):
    """True when a summary dict's typed metadata carries our trusted
    machine-owned unavailable status (exact grammar)."""
    meta = _normalize_metadata(message)
    status = meta.get(AVAILABILITY_KEY)
    return (isinstance(status, dict) and set(status) == {'state', 'reason_code'}
            and status['state'] == AVAILABILITY_UNAVAILABLE
            and isinstance(status['reason_code'], str)
            and status['reason_code'] in AVAILABILITY_REASONS)



def validate_presentation(db, session_id, profile, reference):
    """Full store-bound resolution for the resume path (never model text):
    owner, grammar AND source identity must all hold."""
    return resolve(db, session_id, profile, reference)


def latest_presentation(db, session_id, profile):
    """Trusted resume construction: read the NEWEST typed carrier from the
    durable rows this session's resume lineage projects, and resolve it fully
    against the same store. Absent carrier -> None (ordinary sessions keep
    existing behavior); present-but-invalid/unresolvable -> ValueError so the
    caller binds the restricted-refusal marker rather than falling back broad.
    Only the structured carrier is read: summary prose is never a source."""
    try:
        row = _read_latest_row(db, session_id, CARRIER_KEY)
    except Exception as error:
        raise ArchiveStoreUnavailable('archive store unavailable') from error
    if row is None:
        return None
    meta = _decode_meta(row)
    if CARRIER_KEY not in meta:
        raise ValueError('archive presentation restricted: carrier metadata malformed')
    reference = meta[CARRIER_KEY]
    if not isinstance(reference, str):
        raise ValueError('archive presentation restricted: receipt carrier must be text')
    try:
        validate_reference(reference)
        resolve(db, session_id, profile, reference)
    except ArchiveSourceBudgetError as error:
        # BD-1 truthful recall semantics: a receipt whose (cumulative) source
        # now exceeds the frozen source cap can never be VERIFIED. Claiming a
        # truncated slice as complete coverage is forbidden, so this is a
        # narrow refusal with the budget reason, never a broad fallback.
        raise ValueError(f'{REJECTION_PREFIX}archive source budget exceeded; '
                         f'receipt coverage not verifiable: {error}') from error
    except ValueError as error:
        raise ValueError(f'{REJECTION_PREFIX}{error}') from error
    return reference


def resolve_presentation(db, session_id, profile):
    """Recall-surface decision over BOTH typed keys, newest row wins:
      - valid carrier -> reference (bind the narrow recall path)
      - machine-owned unavailable status -> ArchiveAvailabilityUnavailable
      - malformed/unresolvable carrier or malformed status -> final ValueError
      - NO typed key anywhere -> None (absence POSITIVELY established — the
        only fact that may clear a transient restriction)
    Store faults raise ArchiveStoreUnavailable (transient class)."""
    _owner_profile(db, session_id)  # store fault -> ArchiveStoreUnavailable
    try:
        carrier_row = _read_latest_row(db, session_id, CARRIER_KEY)
        status_row = _read_latest_row(db, session_id, AVAILABILITY_KEY)
    except Exception as error:
        raise ArchiveStoreUnavailable('archive store unavailable') from error
    if carrier_row is not None and (status_row is None
                                    or carrier_row['id'] >= status_row['id']):
        # A newer carrier re-establishes recall over an older degraded
        # publication: a later cycle that minted successfully supersedes it.
        meta = _decode_meta(carrier_row)
        if AVAILABILITY_KEY in meta:
            validate_availability_status(meta[AVAILABILITY_KEY])
        if CARRIER_KEY not in meta:
            raise ValueError('archive presentation restricted: carrier metadata malformed')
        reference = meta[CARRIER_KEY]
        if not isinstance(reference, str):
            raise ValueError('archive presentation restricted: receipt carrier must be text')
        try:
            validate_reference(reference)
            resolve(db, session_id, profile, reference)
        except ArchiveSourceBudgetError as error:
            raise ValueError(f'{REJECTION_PREFIX}archive source budget exceeded; '
                             f'receipt coverage not verifiable: {error}') from error
        except ValueError as error:
            raise ValueError(f'{REJECTION_PREFIX}{error}') from error
        return reference
    if status_row is not None:
        meta = _decode_meta(status_row)
        if AVAILABILITY_KEY not in meta:
            raise ValueError('archive presentation restricted: availability metadata malformed')
        # Exact grammar only; a foreign lookalike shape is never an
        # availability fact.
        validate_availability_status(meta[AVAILABILITY_KEY])
        raise ArchiveAvailabilityUnavailable(
            f"archive recall unavailable: {meta[AVAILABILITY_KEY]['reason_code']}")
    return None


def _try_recovery_probe(agent, session_id):
    """BD-2 bounded recovery for a transient store-fault restriction. Runs
    ONLY on a future recall or a successful compaction refresh, at most
    RECOVERY_MAX_ATTEMPTS times, at least RECOVERY_PROBE_BACKOFF_S apart
    (auto-retry only: a successful refresh is itself a liveness proof).
    Clears ONLY when the SAME owner/store positively establishes no carrier
    and no availability status; a failed probe NEVER clears."""
    import time as _time
    state = getattr(agent, '_compression_archive_recovery', None)
    if not isinstance(state, dict):
        state = {'attempts': 0, 'last': 0.0}
        agent._compression_archive_recovery = state
    if state['attempts'] >= RECOVERY_MAX_ATTEMPTS:
        return None  # expired: stays restricted, no automatic retry, never cleared
    now = _time.monotonic()
    if now - state['last'] < RECOVERY_PROBE_BACKOFF_S:
        return None
    state['attempts'] += 1
    state['last'] = now
    db = getattr(agent, '_session_db', None)
    if db is None:
        return None
    try:
        profile = _owner_profile(db, session_id)
        reference = resolve_presentation(db, session_id, profile)
    except (ArchiveStoreUnavailable, ArchiveAvailabilityUnavailable, ValueError):
        # Uncertain or final-negative: keep the restricted binding exactly.
        return None
    if reference is None:
        # Positive absence established by the same owner/store -> restore the
        # accepted broad surface.
        agent._compression_archive_recall = None
    else:
        agent._compression_archive_recall = (
            ArchiveRecall(db, session_id, profile,
                          allowed_tools=_grant_from_agent(agent)), reference)
    return agent._compression_archive_recall


def refresh_after_publication(agent, session_id):
    """Ordinary refresh after a SUCCESSFUL compaction publication: rebuild the
    live agent's binding from the durable typed carrier on the SAME store
    generation. Absent carrier keeps existing behavior (unbound stays
    unbound); present-but-invalid -> restricted marker, never fallback.

    BD-2: when the live binding is a transient store-fault restriction, the
    successful publication itself proves the store answered — the bounded
    recovery probe runs here so a carrier-less session is not stuck
    restricted after recovery."""
    binding = getattr(agent, '_compression_archive_recall', None)
    if (isinstance(binding, tuple) and len(binding) == 3
            and binding[2] == 'store-unavailable'):
        _try_recovery_probe(agent, session_id)
        return
    _bind_from_store(agent, session_id, require_carrier=False)



def _receipts_enabled():
    """Same opt-in as the publication seam
    (``auxiliary.compression.archive_receipts: true``); absent/false/unreadable
    = this module is entirely inert on the ordinary paths."""
    try:
        from agent.auxiliary_client import _get_auxiliary_task_config
        config = _get_auxiliary_task_config('compression')
    except Exception:
        return False
    return isinstance(config, dict) and config.get('archive_receipts') is True


def bind_resume_binding(agent):
    """GXTD-814 producer on the ordinary construction path (every surface
    that passes a bound ``session_db``): reconstruct the archive binding from
    the LATEST trusted typed key of this agent's own session row — the
    actual bound SessionDB, the ``sessions.profile_name`` owner row, and
    ``agent.valid_tool_names`` as the tool grant. Never parses prose, model
    arguments, or arbitrary strings. No DB / no session row / no typed key ->
    leave UNBOUND, i.e. exactly the accepted behavior. Present-but-invalid,
    unresolvable, or an established-unavailable status -> bind the
    restricted-refusal marker so the ordinary inline route refuses narrowly
    instead of falling back broad. A TRANSIENT store fault binds a tagged
    marker that bounded recovery may clear. Opt-in: config-disabled
    construction performs zero archive reads."""
    if not _receipts_enabled():
        return
    _bind_from_store(agent, getattr(agent, "session_id", ""), require_carrier=True)


def _is_transient_restriction(binding):
    return (isinstance(binding, tuple) and len(binding) == 3
            and binding[2] == 'store-unavailable')


def attempt_transient_recovery_probe(agent) -> bool:
    """Bounded-probe entry for the recall route: True ONLY when a transient
    restriction was JUST resolved (binding rebound or cleared) by this call.
    Failed, rate-limited, expired, or non-transient bindings return False and
    change nothing — a failed probe never clears a restriction."""
    binding = getattr(agent, '_compression_archive_recall', None)
    if not _is_transient_restriction(binding):
        return False
    _try_recovery_probe(agent, getattr(agent, 'session_id', '') or '')
    return getattr(agent, '_compression_archive_recall', None) is not binding


def _bind_from_store(agent, session_id, *, require_carrier):
    if not session_id:
        return
    db = getattr(agent, "_session_db", None)
    if db is None:
        return
    grant = _grant_from_agent(agent)
    try:
        profile = _owner_profile(db, session_id)
        reference = resolve_presentation(db, session_id, profile)
    except ArchiveStoreUnavailable as error:
        # An enabled consumer cannot establish absence of a carrier while its
        # owner store is unavailable. Preserve ordinary agent construction,
        # but refuse archive search rather than silently taking the broad
        # path. TAGGED as transient: the SAME owner/store may later clear it
        # through a bounded recovery probe — a probe failure never clears.
        agent._compression_archive_recall = (
            None, f'{REJECTION_PREFIX}{error}', 'store-unavailable')
        return
    except ArchiveAvailabilityUnavailable as error:
        # Durable machine-owned status: recall is established-unavailable.
        # Restricted, and no probe changes it — only a NEWER typed carrier
        # from a later successful mint can re-establish recall.
        agent._compression_archive_recall = (None, f'{REJECTION_PREFIX}{error}')
        return
    except ValueError as error:
        message = str(error)
        agent._compression_archive_recall = (
            None, message if message.startswith(REJECTION_PREFIX) else f'{REJECTION_PREFIX}{message}')
        return
    if reference is None:
        if not require_carrier:
            # A compaction refresh for a session that never carried a receipt
            # must not clear or bind anything; ordinary sessions are untouched.
            return
        agent._compression_archive_recall = None
        return
    agent._compression_archive_recall = (
        ArchiveRecall(db, session_id, profile, allowed_tools=grant), reference)



def participation(messages):
    """True when this compaction candidate is archive-participating (typed
    carrier presented or mint requested). Plain compactions stay exactly as
    accepted — no scan side effects beyond this cheap structured probe."""
    if any(isinstance(m, dict) and m.get(MINT_REQUEST_KEY) for m in messages or ()):
        return True
    try:
        return bool(validate_carriers(messages))
    except ValueError:
        return True  # a malformed typed carrier participates as a refusal


def _owner_profile(db, session_id):
    try:
        with db._read_ctx() as conn:
            row = conn.execute('SELECT profile_name FROM sessions WHERE id=?', (session_id,)).fetchone()
    except Exception as error:
        # Transient store fault class (BD-2): may be cleared later ONLY by a
        # positive observation from the same owner/store, never by a probe
        # that fails again.
        raise ArchiveStoreUnavailable('archive store unavailable') from error
    profile = row['profile_name'] if row is not None else None
    if not isinstance(profile, str) or not profile:
        raise ValueError('archive owner profile unbound')
    return profile


def _grant_from_agent(agent):
    return {'session_search'} if 'session_search' in (getattr(agent, 'valid_tool_names', None) or set()) else set()
