"""Structural boundary regressions; no model/network or live profile mutations."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tools.delegate_tool_boundary import supplied_context_only, constrain_child


def test_no_profile_does_not_consult_target_configuration():
    with patch('hermes_cli.config.load_config_readonly') as load:
        assert supplied_context_only(None) is False
        load.assert_not_called()


@pytest.mark.parametrize('value', [None, False, {}, [], 'legacy', 'SUPPLIED_CONTEXT_ONLY'])
def test_invalid_explicit_policy_fails_closed(value):
    with patch('hermes_cli.config.load_config_readonly', return_value={'delegation': {'child_boundary': value}}), \
         patch('tools.delegate_tool_routes.delegated_profile_scope', return_value=nullcontext()):
        with pytest.raises(ValueError, match='Unsupported'):
            supplied_context_only(Path('/synthetic-target'))


def test_enrollment_is_target_scoped_and_explicit():
    with patch('hermes_cli.config.load_config_readonly', return_value={'delegation': {'child_boundary': 'supplied_context_only'}}), \
         patch('tools.delegate_tool_routes.delegated_profile_scope', return_value=nullcontext()) as scope:
        assert supplied_context_only(Path('/synthetic-target')) is True
        scope.assert_called_once_with(Path('/synthetic-target'))


def test_absent_policy_preserves_legacy_contract():
    with patch('hermes_cli.config.load_config_readonly', return_value={'delegation': {}}), \
         patch('tools.delegate_tool_routes.delegated_profile_scope', return_value=nullcontext()):
        assert supplied_context_only(Path('/synthetic-target')) is False


def test_fresh_schema_cannot_expose_context_or_discovery_tools():
    from toolsets import agent_tool_surface
    child = SimpleNamespace(tools=[{'function': {'name': 'write_file'}}], valid_tool_names={'write_file'})
    constrain_child(child)
    forged = [{'function': {'name': n}} for n in ['future_private_tool', 'tool_call', 'tool_search']]
    assert agent_tool_surface(child, forged) == []
    assert child.valid_tool_names == set()


def test_refresh_final_merge_drops_unknown_capabilities():
    from tools.mcp_tool_agent import _drop_session_tools
    child = SimpleNamespace(_delegation_context_only=True)
    definitions = [{'function': {'name': 'future_private_tool'}}]
    assert _drop_session_tools(child, definitions, {'future_private_tool'}) == ([], set())


def test_middleware_denies_before_execution():
    from agent.tool_executor import _run_agent_tool_execution_middleware
    execute = Mock(side_effect=AssertionError('forbidden effect'))
    result = _run_agent_tool_execution_middleware(
        SimpleNamespace(_delegation_context_only=True), function_name='write_file',
        function_args={'path': '/must-not-be-written'}, effective_task_id='synthetic',
        tool_call_id='synthetic-call', execute=execute,
    )
    assert result.blocked and not result.dispatched
    assert json.loads(result.result)['error'] == 'tool_denied_for_supplied_context_child'
    execute.assert_not_called()


def test_direct_invoke_cannot_bypass_boundary():
    from agent.agent_runtime_helpers import invoke_tool
    result = invoke_tool(SimpleNamespace(_delegation_context_only=True), 'tool_call', {}, 'synthetic',
                         pre_tool_block_checked=True, skip_tool_execution_middleware=True)
    assert json.loads(result)['error'] == 'tool_denied_for_supplied_context_child'


def test_child_run_does_not_seed_parent_workspace():
    from tools.delegate_tool_child_run import _ChildRun
    child = SimpleNamespace(_delegation_context_only=True)
    parent = SimpleNamespace(_current_task_id='synthetic-private-parent')
    run = _ChildRun(child, parent, 0, 'supplied text task', 'synthetic-child', None)
    with patch('tools.terminal_tool.get_session_cwd') as cwd:
        run.seed_workspace()
        cwd.assert_not_called()
    assert run.parent_task_id is None
    assert run.worktree_info is None
    assert run.parent_reads_snapshot == []
    assert run.wall_start > 0
