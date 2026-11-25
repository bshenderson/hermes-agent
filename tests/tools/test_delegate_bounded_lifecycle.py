"""Native child worker/dispatch coverage with scripted model and web-provider bodies.

No authenticated ingress, Internet retrieval or durable no-replay claim is made.
"""
import json
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('cancel', [False, True], ids=['result', 'cooperative-timeout'])
def test_bounded_worker_scope_dispatch_and_settlement(tmp_path, monkeypatch, cancel):
    from run_agent import AIAgent
    from tools import delegate_tool as dt, delegate_tool_config as dtc, web_tools
    from tools import delegate_tool_child_run as dtcr
    from tools.delegate_tool_child_run import _ChildRun
    from agent.agent_runtime_helpers import invoke_tool
    from agent import web_search_registry
    from hermes_constants import get_hermes_home

    parent_home = tmp_path / 'parent'
    parent_home.mkdir()
    target = tmp_path / 'target'
    target.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(parent_home))
    (target / 'config.yaml').write_text(
        'delegation:\n  child_boundary: bounded_tools\n'
        '  child_tool_ceiling: [web_search]\n', encoding='utf-8')
    monkeypatch.setattr(dt, '_load_config', dict)
    monkeypatch.setattr(dtc, '_load_config', dict)
    monkeypatch.setattr(web_tools, 'check_web_api_key', lambda: True)
    monkeypatch.setattr(dtcr, '_get_child_timeout', lambda: 0.3 if cancel else 3)
    parent = AIAgent(api_key='fixture-parent', base_url='http://127.0.0.1:9/v1',
                     provider='custom', api_mode='chat_completions', model='fixture',
                     session_id='lifecycle-parent', platform='cli', quiet_mode=True,
                     skip_context_files=True, skip_memory=True, save_trajectories=False,
                     enabled_toolsets=['web'])
    child = None
    retired = threading.Event()
    observed = []
    body_calls = []
    try:
        parent.tools = [td for td in parent.tools if td['function']['name'] == 'web_search']
        parent.valid_tool_names = {'web_search'}
        assert parent.tools
        child = dt._build_child_agent(
            task_index=0, goal='Retrieve public fixture', context=None, toolsets=['all'],
            model='fixture', max_iterations=2, task_count=1, parent_agent=parent,
            override_provider='custom', override_base_url='http://127.0.0.1:9/v1',
            override_api_key='fixture-target', override_api_mode='chat_completions', profile_home=target)
        assert child.valid_tool_names == {'web_search'}

        def search(query, limit):
            observed.append((get_hermes_home(), threading.get_ident(), query))
            if cancel:
                deadline = time.monotonic() + 3
                while not child._interrupt_requested and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert child._interrupt_requested
            return {'success': True, 'data': {'web': [{'url': 'https://example.org/',
                    'title': 'Offline lifecycle fixture', 'description': query}]}}

        monkeypatch.setattr(web_tools, '_get_search_backend', lambda: '')
        monkeypatch.setattr(web_search_registry, 'get_active_search_provider',
                            lambda: SimpleNamespace(name='lifecycle-fixture', search=search))

        def scripted_body(**kwargs):
            body_calls.append(kwargs['task_id'])
            try:
                denied = invoke_tool(child, 'terminal', {}, kwargs['task_id'],
                                     pre_tool_block_checked=True, skip_tool_execution_middleware=True)
                assert 'denied' in json.loads(denied)['error']
                result = invoke_tool(child, 'web_search', {'query': f'lifecycle-{cancel}'},
                                     kwargs['task_id'], pre_tool_block_checked=True,
                                     skip_tool_execution_middleware=True)
                return {'final_response': result, 'completed': not cancel, 'interrupted': cancel}
            finally:
                retired.set()

        monkeypatch.setattr(child, 'run_conversation', scripted_body)
        run = _ChildRun(child, parent, 0, 'Retrieve public fixture', 'bounded-worker', None)
        run.seed_workspace()
        assert run.parent_task_id is None and run.parent_reads_snapshot == []
        result, error, _ = run.await_child()
        assert retired.wait(3), 'owned scripted worker failed to retire'
        assert len(observed) == len(body_calls) == 1
        assert observed[0][0] == target
        assert observed[0][1] != threading.get_ident()
        assert body_calls == [run.child_task_id]
        assert get_hermes_home() == parent_home
        if cancel:
            assert result is None and error['status'] == 'timeout'
            assert child._interrupt_requested
        else:
            assert error is None and result['completed']
            assert json.loads(result['final_response'])['success']
    finally:
        if child is not None:
            child.hard_interrupt('fixture teardown')
            if body_calls:
                assert retired.wait(3)
            child.close()
        parent.close()
