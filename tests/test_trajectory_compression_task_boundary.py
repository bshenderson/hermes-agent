"""Trajectory summarization must retain the compression task's empty-chain boundary."""
import asyncio
import logging

import pytest

from tests.agent.test_auxiliary_recovery_policy import peer
from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics


@pytest.mark.parametrize('mode', ['sync', 'async'])
def test_trajectory_compression_does_not_escape_to_primary(peer, mode):
    calls, state, configure, _ = peer
    state['status'] = 429
    configure('compression', recovery_policy=None)
    compressor = object.__new__(TrajectoryCompressor)
    compressor.config = CompressionConfig(summarization_model='fixture-aux', summary_target_tokens=32, max_retries=1)
    compressor._use_call_llm = True
    compressor._llm_provider = 'fixture-aux'
    compressor.logger = logging.getLogger('trajectory-fixture')
    metrics = TrajectoryMetrics()
    if mode == 'sync':
        result = compressor._generate_summary('Synthetic trajectory only.', metrics)
    else:
        result = asyncio.run(compressor._generate_summary_async('Synthetic trajectory only.', metrics))
    assert calls
    assert all(path.startswith('/aux/') for path, _ in calls)
    assert 'PRIMARY' not in result
    assert metrics.summarization_errors == 1
