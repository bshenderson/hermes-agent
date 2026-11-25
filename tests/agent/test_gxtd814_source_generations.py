"""Canonical archive bytes survive the bounded replay projection."""
import json

import pytest

from agent.compression_archive import PREFIX, _snapshot_on_conn, capture, resolve
from tests._fixtures.store_support import (
    append, compact, conversation, create_session, open_store, raw, write_sql)


@pytest.mark.parametrize('role', ['user', 'assistant'])
def test_projected_generation_keeps_raw_canonical_digest(tmp_path, role):
    with open_store(tmp_path) as store:
        create_session(store, 's', profile_name='research')
        raw_content = ' \nOriginal  interior spacing.\n\t'
        append(store, 's', role, raw_content)
        before = capture(store, 's', 'research')
        # Actual durable replay, not a hand-normalized fixture.
        live = conversation(store, 's')
        assert live[0]['content'] == raw_content.strip()
        compact(store, 's', live)
        after = capture(store, 's', 'research')
        with raw(tmp_path) as conn:
            in_txn, _ = _snapshot_on_conn(conn, 's', 'research')
        old = json.loads(before[len(PREFIX):])
        new = json.loads(after[len(PREFIX):])
        assert new == in_txn
        assert old['sha256'] == new['sha256']
        assert new['count'] == 1
        rows = resolve(store, 's', 'research', after)
        assert rows[0]['content'] == raw_content
        # Canonical bytes remain pinned even if replay would normalize them.
        write_sql(tmp_path, 'UPDATE messages SET content=? WHERE id=?',
                  (raw_content.strip(), rows[0]['id']))
        with pytest.raises(ValueError, match='changed or missing'):
            resolve(store, 's', 'research', after)


@pytest.mark.parametrize('fault', ['interior', 'role', 'added-boundary', 'tool'])
def test_generation_conflicts_still_refuse_both_snapshot_paths(tmp_path, fault):
    with open_store(tmp_path) as store:
        create_session(store, 's', profile_name='research')
        role = 'tool' if fault == 'tool' else 'user'
        raw_content = ' \nOriginal  interior spacing.\n'
        append(store, 's', role, raw_content)
        append(store, 's', role, raw_content)
        with raw(tmp_path) as conn:
            rows = conn.execute('SELECT id,message_uid FROM messages ORDER BY id').fetchall()
        content = raw_content.strip()
        changed_role = role
        if fault == 'interior':
            content = content.replace('  ', ' ')
        elif fault == 'role':
            changed_role = 'assistant'
        elif fault == 'added-boundary':
            content = raw_content + '\n'
        write_sql(tmp_path, 'UPDATE messages SET message_uid=?,role=?,content=? WHERE id=?',
                  (rows[0]['message_uid'], changed_role, content, rows[1]['id']))
        with pytest.raises(ValueError, match='clone identity conflict'):
            capture(store, 's', 'research')
        with raw(tmp_path) as conn:
            with pytest.raises(ValueError, match='clone identity conflict'):
                _snapshot_on_conn(conn, 's', 'research')
