"""Exact schema and final-dispatch ceilings for authenticated action turns."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import uuid

import pytest

from agent import domain_action_context as context
from agent import domain_action_tools as tools

POLICY = {'toolsets': ['file', 'search', 'clarify', 'todo'],
          'allowed_tools': ['govcon_action', 'web_search', 'clarify', 'todo_list'],
          'required_tools': ['govcon_action']}


def origin():
    actor = 'fixture-actor'
    native = {'subject_id': actor, **{name: str(uuid.uuid4()) for name in
              ('chat_id', 'message_id', 'user_message_id')}}
    body = context.sign({'hermes_chat_context': native}, subject=actor, groups=['fixture'],
                        profile='fixture-domain', key='fixture')
    return context.accept(body, profile='fixture-domain', key='fixture')


def agent():
    names = ['govcon_action', 'terminal', 'read_file', 'tool_call', 'web_search',
             'browser_vault_fill', 'delegate_task', 'cronjob', 'skill_manage', 'mcp__mail__send']
    return SimpleNamespace(tools=[{'function': {'name': name}} for name in names], valid_tool_names=set(names))


def test_ordinary_owner_worker_turn_unchanged():
    value = agent()
    before = list(value.tools)
    assert tools.policy_for_current({}) is None
    tools.constrain_agent(value, None)
    assert value.tools == before
    assert tools.dispatch_denial(value, 'terminal') is None


def test_signed_mode_requires_explicit_trusted_policy():
    with context.bound(origin()):
        with pytest.raises(ValueError, match='policy_required'):
            tools.policy_for_current({})
        policy = tools.policy_for_current({'api_server': {'domain_action_tools': POLICY}})
        assert policy == POLICY


def test_exact_schema_filter_and_final_guard_survive_schema_mutation():
    value = agent()
    with context.bound(origin()):
        tools.constrain_agent(value, POLICY)
        assert value.valid_tool_names == {'govcon_action', 'web_search'}
        # A plugin/tool-search refresh cannot widen the captured execution ceiling.
        value.valid_tool_names.add('terminal')
        assert tools.dispatch_denial(value, 'terminal') is not None
        assert tools.dispatch_denial(value, 'govcon_action') is None
    assert tools.dispatch_denial(value, 'govcon_action') is not None


def test_missing_required_plugin_fails_before_turn():
    value = SimpleNamespace(tools=[{'function': {'name': 'terminal'}}])
    with context.bound(origin()):
        with pytest.raises(ValueError, match='unavailable'):
            tools.constrain_agent(value, POLICY)


def test_other_origin_cannot_reuse_agent():
    value = agent()
    with context.bound(origin()):
        tools.constrain_agent(value, POLICY)
    with context.bound(origin()):
        assert tools.dispatch_denial(value, 'govcon_action') is not None


def test_actual_middleware_denies_before_relay_hooks_or_execution(monkeypatch):
    from agent import tool_executor, relay_tools
    value = agent()
    monkeypatch.setattr(relay_tools, 'execute', lambda *a, **kw: pytest.fail('denied tool reached Relay'))
    with context.bound(origin()):
        tools.constrain_agent(value, POLICY)
        for name in ('terminal', 'read_file', 'tool_call', 'browser_vault_fill', 'delegate_task'):
            result = tool_executor._run_agent_tool_execution_middleware(value,
                function_name=name, function_args={}, effective_task_id='fixture', tool_call_id='fixture',
                execute=lambda *a: pytest.fail('denied tool executed'))
            assert result.blocked and not result.dispatched


def test_worker_context_propagation_and_missing_context_fail_closed(monkeypatch):
    from tools import thread_context
    monkeypatch.setattr(thread_context, '_callback_api', lambda: ())
    value = agent()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with context.bound(origin()):
            tools.constrain_agent(value, POLICY)
            wrapped = thread_context.propagate_context_to_thread(lambda: tools.dispatch_denial(value, 'govcon_action'))
            assert pool.submit(wrapped).result() is None
            assert pool.submit(tools.dispatch_denial, value, 'govcon_action').result() is not None
        assert pool.submit(tools.dispatch_denial, value, 'govcon_action').result() is not None


def test_expiry_never_falls_back_to_owner_policy(monkeypatch):
    value = origin()
    with context.bound(value):
        scoped = agent()
        tools.constrain_agent(scoped, POLICY)
        monkeypatch.setattr(context.time, 'time', lambda: value['expires_at'])
        assert tools.dispatch_denial(scoped, 'govcon_action') is not None
        with pytest.raises(ValueError):
            tools.policy_for_current({})


def test_real_api_constructor_uses_scoped_toolsets_and_ceiling(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from tests.gateway.test_api_server import _patch_create_agent_runtime
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.tools = agent().tools
            self.valid_tool_names = {'terminal', 'govcon_action', 'web_search'}

    _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
    monkeypatch.setattr('gateway.run._load_gateway_config',
                        lambda: {'api_server': {'domain_action_tools': POLICY}})
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    monkeypatch.setattr(adapter, '_ensure_session_db', lambda: None)
    monkeypatch.setattr(adapter, '_session_model_override_for', lambda *_: None)
    with context.bound(origin()):
        value = adapter._create_agent(session_id='scoped-fixture')
        assert captured['enabled_toolsets'] == POLICY['toolsets']
        assert value.valid_tool_names == {'govcon_action', 'web_search'}
        assert tools.dispatch_denial(value, 'terminal') is not None


def test_failed_scope_construction_returns_memory_owner(monkeypatch):
    from unittest.mock import Mock
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from tests.gateway.test_api_server import _patch_create_agent_runtime

    class MissingPluginAgent:
        def __init__(self, **_kwargs):
            self.tools = [{'function': {'name': 'terminal'}}]

    _patch_create_agent_runtime(monkeypatch, {}, MissingPluginAgent)
    monkeypatch.setattr('gateway.run._load_gateway_config',
                        lambda: {'api_server': {'domain_action_tools': POLICY}})
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    monkeypatch.setattr(adapter, '_ensure_session_db', lambda: None)
    monkeypatch.setattr(adapter, '_session_model_override_for', lambda *_: None)
    checkin = Mock()
    adapter._memory_sessions = SimpleNamespace(checkout=lambda _session: None, checkin=checkin)
    with context.bound(origin()):
        with pytest.raises(ValueError, match='unavailable'):
            adapter._create_agent(session_id='missing-plugin')
    checkin.assert_called_once()
