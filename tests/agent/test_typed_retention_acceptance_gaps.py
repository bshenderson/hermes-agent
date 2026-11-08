"""GXTD-810 unresolved acceptance discriminators; controlled output, not native.

These tests intentionally expose the recovered lexical candidate's limits.
No authority-state producer is fabricated to make them pass.
"""
import copy
import json

import pytest

from tests.agent.test_gxtd641_checkpoint_preservation import compressor, history, ledger


@pytest.fixture(autouse=True)
def source_bound_contract(compressor, tmp_path):
    """Exercise the explicitly selected successor, not the legacy lexical strategy."""
    path = tmp_path / 'config.yaml'
    config = json.loads(path.read_text())
    config['auxiliary']['compression']['summary_strategy'] = 'source_bound'
    path.write_text(json.dumps(config))


@pytest.mark.parametrize('source', [
    'Leave the dispatch button untouched until Barry gives the green light.',
    'The acknowledgement is still missing. Another submission risks duplication.',
], ids=['heldout-hold', 'heldout-uncertain-delivery'])
def test_heldout_historical_constraint_survives_lossy_consumer(compressor, source):
    original = history(source)
    snapshot = copy.deepcopy(original)
    result = compressor.compress(original, force=True)
    assert original == snapshot
    assert result != snapshot, compressor._last_summary_error
    assert source in ledger(result), 'lexical rescue is not complete constraint retention'


def test_cross_line_hold_keeps_its_scope_and_release_condition(compressor):
    lines = ['Shipment: Cedar archive.', 'Hold', 'until Barry approves the reconciled delivery receipt.']
    original = history('\n'.join(lines))
    snapshot = copy.deepcopy(original)
    result = compressor.compress(original, force=True)
    assert original == snapshot
    assert result != snapshot, compressor._last_summary_error
    assert set(lines).issubset(ledger(result)), 'isolated keyword loses scope/release condition'
