"""Offline batching boundary regressions; never invoke the vault CLI."""
import json
from unittest.mock import Mock

import pytest

from agent.secret_sources import onepassword as op


@pytest.mark.parametrize('ref', [
    'op://V/I/section/field', 'op://V/I/a%20b',
    'op://V/I/field?attribute=otp', 'op://V/I/field#selector',
])
def test_complex_reference_deduplicates_without_item_fetch(monkeypatch, tmp_path, ref):
    read = Mock(return_value='fixture')
    monkeypatch.setattr(op, '_run_op_read', read)
    monkeypatch.setattr(op, '_run_op_item_fields', Mock(side_effect=AssertionError('unexpected item fetch')))
    values, warnings = op.fetch_onepassword_secrets(
        references={'A': ref, 'B': ref}, binary=tmp_path / 'op', use_cache=False)
    assert values == {'A': 'fixture', 'B': 'fixture'}
    assert not warnings
    read.assert_called_once()


@pytest.mark.parametrize('rows', [None, {}, [None], [],
    [{'id': 'a', 'value': ''}, {'id': 'b', 'value': 'fixture'}],
    [{'id': 'a', 'value': 123}, {'id': 'b', 'value': 'fixture'}],
])
def test_invalid_group_refuses_all_and_does_not_cache(monkeypatch, tmp_path, rows):
    monkeypatch.setattr(op, 'run_cli', Mock(return_value=Mock(
        returncode=0, stdout=json.dumps({'fields': rows}))))
    store = Mock()
    monkeypatch.setattr(op._STORE, 'store', store)
    monkeypatch.setattr(op._STORE, 'lookup', Mock(return_value=None))
    values, warnings = op.fetch_onepassword_secrets(
        references={'A': 'op://V/I/a', 'B': 'op://V/I/b'},
        binary=tmp_path / 'op', home_path=tmp_path)
    assert not values
    assert len(warnings) == 2
    store.assert_not_called()


def test_group_failure_preserves_other_item_but_does_not_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(op, '_run_op_item_fields', Mock(side_effect=RuntimeError('group refused')))
    monkeypatch.setattr(op, '_run_op_read', Mock(return_value='separate'))
    monkeypatch.setattr(op._STORE, 'lookup', Mock(return_value=None))
    store = Mock()
    monkeypatch.setattr(op._STORE, 'store', store)
    values, warnings = op.fetch_onepassword_secrets(
        references={'A': 'op://V/I/a', 'B': 'op://V/I/b', 'C': 'op://V/J/c'},
        binary=tmp_path / 'op', home_path=tmp_path)
    assert values == {'C': 'separate'}
    assert len(warnings) == 2
    store.assert_not_called()


def test_group_passes_account_and_restricted_environment(monkeypatch, tmp_path):
    run = Mock(return_value=Mock(returncode=0, stdout=json.dumps({'fields': [
        {'id': 'a', 'value': 'a-value'}, {'id': 'b', 'value': 'b-value'},
    ]})))
    monkeypatch.setattr(op, 'run_cli', run)
    monkeypatch.setattr(op, 'get_source_environment', lambda: {
        'PATH': '/fixture', 'UNRELATED_SECRET': 'must-not-export'})
    values, errors = op._fetch_unique_refs(
        tmp_path / 'op', ['op://V/I/a', 'op://V/I/b'], account='fixture-account', token_value='fixture-token')
    assert not errors and len(values) == 2
    args, kwargs = run.call_args
    assert args[0][-3:] == ['fixture-account', '--', 'I']
    assert kwargs['env']['OP_SERVICE_ACCOUNT_TOKEN'] == 'fixture-token'
    assert 'UNRELATED_SECRET' not in kwargs['env']
