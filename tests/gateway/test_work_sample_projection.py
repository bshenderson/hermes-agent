"""GXTD-755: diagnostics never disclose arbitrary runtime fields."""
import copy
import pytest
from gateway import run_work_sample as samples


def sample():
    return {'schema': 1, 'pid': 123, 'start_time': 456, 'sequence': 1,
            'observed_at': '2026-11-16T00:00:00+00:00',
            'monotonic_start': 1.0, 'monotonic_end': 2.0,
            'components': dict(chat=1, managed=0, cron=0, api=0, deferred=0), 'total': 1,
            'availability': dict(chat=True, managed=True, cron=True, api=True, deferred=True)}


def test_projection_strips_private_fields():
    assert hasattr(samples, 'project_work_sample')
    value = sample()
    value['prompt'] = 'PRIVATE_SENTINEL'
    result = samples.project_work_sample(value, pid=123, start_time=456, total=1)
    assert result == sample()


@pytest.mark.parametrize('key,value', [('pid', 999), ('total', True), ('sequence', -1),
    ('monotonic_end', float('nan')), ('observed_at', 'PRIVATE_SENTINEL'), ('start_time', None)])
def test_invalid_sample_is_unavailable(key, value):
    assert hasattr(samples, 'project_work_sample')
    data = copy.deepcopy(sample())
    data[key] = value
    assert samples.project_work_sample(data, pid=123, start_time=456, total=1) is None
