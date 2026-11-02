"""Bounded historical-custody coverage, not authorization inference."""
import copy
import pytest
from tests.agent.test_gxtd641_checkpoint_preservation import compressor, history, ledger


@pytest.mark.parametrize('record', [
    "Do not send Cedar's email, delete source data, contact Sparky, or retry an unknown delivery.",
    'Private research marker PVT-313 stays in the research domain; it is not public/shared chat material.',
    'Never replay the upload while its delivery outcome is unknown.',
    'Hold publication until the operator authorizes release.',
    'The delivery status remains uncertain; reconcile before resubmission.',
    'Correction: the private marker belongs to media, not research.',
])
def test_undated_constraint_survives_lossy_summary(compressor, record):
    original = history(record)
    before = copy.deepcopy(original)
    result = compressor.compress(original, force=True)
    assert original == before
    assert result != before, compressor._last_summary_error
    assert record in ledger(result)


def test_protected_overflow_refuses_without_mutating_history(compressor):
    original = history('\n'.join(f'Do not replay unknown delivery number {i}.' for i in range(400)))
    before = copy.deepcopy(original)
    result = compressor.compress(original, force=True)
    assert result == before
    assert original == before
    assert compressor._last_compress_aborted
    assert 'preservation budget' in compressor._last_summary_error


def test_repeated_compaction_keeps_conflicting_history_without_granting_authority(compressor):
    first_record = 'Hold delivery; the upload outcome is unknown.'
    correction = 'Correction: private marker belongs to media, not research.'
    original = history(first_record)
    first = compressor.compress(copy.deepcopy(original), force=True)
    assert first != original, compressor._last_summary_error
    second_input = first + history(correction)[1:]
    second = compressor.compress(copy.deepcopy(second_input), force=True)
    assert second != second_input, compressor._last_summary_error
    assert {first_record, correction}.issubset(ledger(second))
