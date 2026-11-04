"""Canonical archive bytes survive the bounded SessionDB replay projection."""
import json

import pytest

from agent.compression_archive import PREFIX, _snapshot_on_conn, capture, resolve
from hermes_state import SessionDB


@pytest.mark.parametrize('role', ['user', 'assistant'])
def test_projected_generation_keeps_raw_canonical_digest(tmp_path, role):
    with SessionDB(db_path=tmp_path / 'generation.db') as db:
        db.create_session('s', 'cli', profile_name='research')
        raw = ' \nOriginal  interior spacing.\n\t'
        db.append_message('s', role, raw)
        before = capture(db, 's', 'research')
        # Actual durable replay, not a hand-normalized fixture.
        live = db.get_messages_as_conversation('s')
        assert live[0]['content'] == raw.strip()
        db.archive_and_compact('s', live)
        after = capture(db, 's', 'research')
        with db._read_ctx() as conn:
            in_txn, _ = _snapshot_on_conn(conn, 's', 'research')
        old = json.loads(before[len(PREFIX):])
        new = json.loads(after[len(PREFIX):])
        assert new == in_txn
        assert old['sha256'] == new['sha256']
        assert new['count'] == 1
        rows = resolve(db, 's', 'research', after)
        assert rows[0]['content'] == raw
        # Canonical bytes remain pinned even if replay would normalize them.
        db._write_sql('UPDATE messages SET content=? WHERE id=?',
                      (raw.strip(), rows[0]['id']))
        with pytest.raises(ValueError, match='changed or missing'):
            resolve(db, 's', 'research', after)


@pytest.mark.parametrize('fault', ['interior', 'role', 'added-boundary', 'tool'])
def test_generation_conflicts_still_refuse_both_snapshot_paths(tmp_path, fault):
    with SessionDB(db_path=tmp_path / 'conflict.db') as db:
        db.create_session('s', 'cli', profile_name='research')
        role = 'tool' if fault == 'tool' else 'user'
        raw = ' \nOriginal  interior spacing.\n'
        db.append_message('s', role, raw)
        db.append_message('s', role, raw)
        with db._read_ctx() as conn:
            rows = conn.execute('SELECT id,message_uid FROM messages ORDER BY id').fetchall()
        content = raw.strip()
        changed_role = role
        if fault == 'interior':
            content = content.replace('  ', ' ')
        elif fault == 'role':
            changed_role = 'assistant'
        elif fault == 'added-boundary':
            content = raw + '\n'
        db._write_sql('UPDATE messages SET message_uid=?,role=?,content=? WHERE id=?',
                      (rows[0]['message_uid'], changed_role, content, rows[1]['id']))
        with pytest.raises(ValueError, match='clone identity conflict'):
            capture(db, 's', 'research')
        with db._read_ctx() as conn:
            with pytest.raises(ValueError, match='clone identity conflict'):
                _snapshot_on_conn(conn, 's', 'research')
