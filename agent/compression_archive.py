"""Versioned archive receipts and explicitly granted same-owner session recall.

Isolated integration primitive, not a compressor strategy or authorization source.
The caller binds the already-authorized SessionDB, session, profile and tool grant.
"""
import hashlib
import json

PREFIX = 'HERMES_CONVERSATION_ARCHIVE_V1 '
MAX_ROWS = 4096
MAX_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 8192


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate archive receipt field')
        result[key] = value
    return result


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
                    'ORDER BY id LIMIT ?', (MAX_BYTES,session_id,through,MAX_ROWS+1)).fetchall()
            finally:
                connection.execute('RELEASE conversation_archive')
    except ValueError:
        raise
    except Exception as error:
        raise ValueError('archive store unavailable') from error
    if not rows or len(rows)>MAX_ROWS or sum(r['source_bytes'] or 0 for r in rows)>MAX_BYTES:
        raise ValueError('archive source budget or coverage invalid')
    unique={}
    for row in rows:
        if not isinstance(row['message_uid'],str) or not row['message_uid'] or not isinstance(row['content'],str):
            raise ValueError('archive source identity unavailable')
        uid=row['message_uid']
        if uid in unique:
            previous=unique[uid]
            if (row['role'],row['content'])!=(previous['role'],previous['content']):
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
