"""Native bounded-child construction and dispatch; no live model or profile I/O."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_native_child_intersects_session_grants_and_refuses_bypasses(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'parent'))
    (tmp_path / 'parent').mkdir()
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'config.yaml').write_text(
        'delegation:\n  child_boundary: bounded_tools\n'
        '  child_tool_ceiling: [web_search, web_extract]\n', encoding='utf-8')
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    from tools import delegate_tool_config as dtc
    from tools import web_tools
    from toolsets import agent_tool_surface
    from tools.mcp_tool_agent import _drop_session_tools
    from agent.agent_runtime_helpers import invoke_tool
    from agent.tool_executor import _run_agent_tool_execution_middleware

    monkeypatch.setattr(dt, '_load_config', dict)
    monkeypatch.setattr(dtc, '_load_config', dict)
    monkeypatch.setattr(web_tools, 'check_web_api_key', lambda: True)
    parent = AIAgent(api_key='fixture-only', base_url='http://127.0.0.1:9/v1',
                     provider='custom', api_mode='chat_completions', model='fixture',
                     session_id='bounded-parent', platform='cli', quiet_mode=True,
                     skip_context_files=True, skip_memory=True, save_trajectories=False,
                     enabled_toolsets=['web'])
    child = None
    try:
        # Narrow the real parent's session, independently of its configured toolsets.
        parent.tools = [td for td in parent.tools if td['function']['name'] == 'web_search']
        parent.valid_tool_names = {'web_search'}
        assert parent.tools, 'fixture requires the registered native search schema'
        parent.prefill_messages = [{'role': 'user', 'content': 'PRIVATE-PREFILL-SENTINEL'}]
        child = dt._build_child_agent(
            task_index=0, goal='Public retrieval only', context='supplied brief', toolsets=['all'],
            model='fixture', max_iterations=2, task_count=1, parent_agent=parent,
            override_provider='custom', override_base_url='http://127.0.0.1:9/v1',
            override_api_key='fixture-target', override_api_mode='chat_completions', profile_home=target)
        assert child.valid_tool_names == {'web_search'}
        assert not child.prefill_messages
        from tools.delegate_tool_child_run import _ChildRun
        run = _ChildRun(child, parent, 0, 'supplied brief', 'bounded-child', None)
        run.seed_workspace()
        assert run.parent_task_id is None and run.worktree_info is None
        assert run.parent_reads_snapshot == []
        forged = [{'type': 'function', 'function': {'name': n}} for n in
                  ('web_search', 'web_extract', 'terminal', 'tool_call', 'delegate_task', 'unknown')]
        assert [t['function']['name'] for t in agent_tool_surface(child, forged)] == ['web_search']
        defs, names = _drop_session_tools(child, forged, {t['function']['name'] for t in forged})
        assert names == {'web_search'} and len(defs) == 1
        for name in ('web_extract', 'terminal', 'tool_call', 'delegate_task', 'unknown'):
            result = invoke_tool(child, name, {}, 'bounded-test',
                                 pre_tool_block_checked=True, skip_tool_execution_middleware=True)
            assert 'denied' in json.loads(result)['error']
            execute = Mock(side_effect=AssertionError('forbidden effect'))
            outcome = _run_agent_tool_execution_middleware(
                child, function_name=name, function_args={}, effective_task_id='bounded-test',
                tool_call_id='bounded-call', execute=execute)
            assert outcome.blocked and not outcome.dispatched
            execute.assert_not_called()
        from agent import web_search_registry
        from hermes_constants import get_hermes_home
        from tools.delegate_tool_routes import delegated_profile_scope
        observed = []

        def search(query, limit):
            observed.append((get_hermes_home(), query))
            return {'success': True, 'data': {'web': [{'url': 'https://example.org/',
                    'title': 'Offline provider fixture', 'description': query}]}}

        provider = SimpleNamespace(name='bounded-fixture', search=search)
        monkeypatch.setattr(web_tools, '_get_search_backend', lambda: '')
        monkeypatch.setattr(web_search_registry, 'get_active_search_provider', lambda: provider)
        with delegated_profile_scope(target):
            outcome = _run_agent_tool_execution_middleware(
                child, function_name='web_search', function_args={'query': 'bounded-test-query'},
                effective_task_id='bounded-test', tool_call_id='bounded-positive',
                execute=lambda args: invoke_tool(child, 'web_search', args, 'bounded-test',
                    pre_tool_block_checked=True, skip_tool_execution_middleware=True))
        assert outcome.dispatched and not outcome.blocked
        assert json.loads(outcome.result)['success'] is True
        assert observed == [(target, 'bounded-test-query')]
        # Later surface/config mutations cannot widen the captured immutable ceiling.
        parent.valid_tool_names.add('web_extract')
        child.valid_tool_names.add('terminal')
        result = invoke_tool(child, 'terminal', {}, 'bounded-test',
                             pre_tool_block_checked=True, skip_tool_execution_middleware=True)
        assert 'denied' in json.loads(result)['error']
        assert isinstance(child._delegation_tool_ceiling, frozenset)
        parent.valid_tool_names.remove('web_extract')
        assert parent.valid_tool_names == {'web_search'}
    finally:
        if child is not None:
            child.close()
        parent.close()


@pytest.mark.parametrize('policy,expected', [
    ({}, None),
    ({'child_boundary': 'supplied_context_only', 'child_tool_ceiling': ['web_search']}, frozenset()),
    ({'child_boundary': 'bounded_tools'}, frozenset()),
    ({'child_boundary': 'bounded_tools', 'child_tool_ceiling': []}, frozenset()),
    ({'child_boundary': 'bounded_tools', 'child_tool_ceiling': ['web_search']}, frozenset({'web_search'})),
    ({'child_boundary': 'bounded_tools', 'child_tool_ceiling': ['web_extract']}, frozenset()),
])
def test_target_policy_uses_actual_session_not_toolsets(tmp_path, monkeypatch, policy, expected):
    import hermes_yaml as yaml
    from tools.delegate_tool_boundary import child_tool_ceiling
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'delegation': policy}), encoding='utf-8')
    parent = SimpleNamespace(tools=[{'function': {'name': 'web_search'}}],
                             valid_tool_names={'web_search'}, enabled_toolsets=['all'])
    assert child_tool_ceiling(tmp_path, parent) == expected


@pytest.mark.parametrize('ceiling', [None, 'web_search', {}, ['terminal'], ['web_search', 7]])
def test_invalid_ceiling_refuses_before_construction(tmp_path, ceiling):
    import hermes_yaml as yaml
    from tools.delegate_tool_boundary import child_tool_ceiling
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'delegation': {
        'child_boundary': 'bounded_tools', 'child_tool_ceiling': ceiling}}), encoding='utf-8')
    with pytest.raises(ValueError, match='child_tool_ceiling'):
        child_tool_ceiling(tmp_path, SimpleNamespace())


def test_hard_zero_marker_cannot_be_widened_by_a_tool_ceiling():
    from tools.delegate_tool_boundary import child_tool_denied
    child = SimpleNamespace(_delegation_context_only=True,
                            _delegation_tool_ceiling=frozenset({'web_search'}))
    assert child_tool_denied(child, 'web_search')
