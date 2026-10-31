"""Quota amplification regressions for GXTD-765; no real vault calls."""
import json
from unittest import mock

from agent.secret_sources import onepassword as op


def test_identical_references_are_read_once(monkeypatch, tmp_path):
    calls = []

    def read(*args, **kwargs):
        calls.append(args[1])
        return 'fixture-value'

    monkeypatch.setattr(op, '_run_op_read', read)
    values, warnings = op.fetch_onepassword_secrets(
        references={'A': 'op://V/I/credential', 'B': 'op://V/I/credential'},
        binary=tmp_path / 'op', use_cache=False, home_path=tmp_path,
    )
    assert values == {'A': 'fixture-value', 'B': 'fixture-value'}
    assert warnings == []
    assert calls == ['op://V/I/credential']


def test_same_item_fields_are_fetched_once(monkeypatch, tmp_path):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        assert cmd[1:3] == ['item', 'get']
        assert '--vault' in cmd and '--format' in cmd
        return mock.Mock(returncode=0, stdout=json.dumps({'fields': [
            {'id': 'credential', 'label': 'credential', 'value': 'fixture-key'},
            {'id': 'url', 'label': 'url', 'value': 'https://fixture.invalid'},
            {'id': 'unrequested', 'value': 'must-not-export'},
        ]}), stderr='')

    monkeypatch.setattr(op.subprocess, 'run', run)
    values, warnings = op.fetch_onepassword_secrets(
        references={'A': 'op://V/I/credential', 'B': 'op://V/I/url',
                    'C': 'op://V/I/credential'},
        binary=tmp_path / 'op', use_cache=False, home_path=tmp_path,
    )
    assert values == {'A': 'fixture-key', 'B': 'https://fixture.invalid', 'C': 'fixture-key'}
    assert not warnings
    assert len(calls) == 1


def test_same_item_ambiguous_field_is_not_exported(monkeypatch, tmp_path):
    def run(cmd, **kwargs):
        return mock.Mock(returncode=0, stdout=json.dumps({'fields': [
            {'id': 'one', 'label': 'credential', 'value': 'first'},
            {'id': 'two', 'label': 'credential', 'value': 'second'},
            {'id': 'url', 'value': 'https://fixture.invalid'},
        ]}), stderr='')

    monkeypatch.setattr(op.subprocess, 'run', run)
    values, warnings = op.fetch_onepassword_secrets(
        references={'A': 'op://V/I/credential', 'B': 'op://V/I/url'},
        binary=tmp_path / 'op', use_cache=False, home_path=tmp_path,
    )
    assert 'A' not in values
    assert warnings
    assert all('first' not in w and 'second' not in w for w in warnings)
