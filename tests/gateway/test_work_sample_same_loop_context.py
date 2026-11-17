"""Direct same-loop sampling must retain the startup owner's context."""
import asyncio
from contextvars import ContextVar
from unittest.mock import patch

import pytest
from gateway import run_work_sample as samples, run_runtime
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_same_loop_counts_and_writer_use_captured_owner():
    runner, _ = make_restart_runner()
    scope = ContextVar('sample_owner', default='foreign')
    token = scope.set('launch')
    samples.bind_work_sample_owner(runner)
    scope.reset(token)
    seen = []

    def count(*args):
        seen.append(('counter', scope.get()))
        return 0

    def write(**fields):
        seen.append(('writer', scope.get()))
        assert fields['active_agents'] == fields['work_sample']['total'] == 0

    with patch.object(runner, '_running_agent_count', count), \
         patch.object(run_runtime, 'managed_turn_count', count), \
         patch.object(runner, '_running_cron_job_count', count), \
         patch.object(runner, '_api_server_hook', count), \
         patch.object(runner, '_active_deferred_agent_worker_count', count), \
         patch('gateway.run._write_runtime_status_quiet', write):
        assert runner._work_sample_loop is asyncio.get_running_loop()
        samples.publish_work_sample(runner)
    assert seen == [('counter', 'launch')] * 5 + [('writer', 'launch')]
    assert scope.get() == 'foreign'
