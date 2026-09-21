"""Real-import integration for the request-mode dispatch boundary."""
import json
from types import SimpleNamespace

from agent.preferred_web_policy import ResearchPolicy, bind_agent
from agent.agent_runtime_helpers import invoke_tool
from agent import tool_executor


def agent():
    value = SimpleNamespace(tools=[{'type':'function','function':{'name':n}} for n in ['web_search','web_extract','terminal']], valid_tool_names=set())
    bind_agent(value, ResearchPolicy('/research task'))
    return value


def test_direct_invoke_denies_before_inline_delegation_and_middleware():
    value = agent()
    for name in ('delegate_task','tool_call','execute_code','terminal','handoff_profile','browser_navigate','computer_use'):
        assert json.loads(invoke_tool(value, name, {}, 'test', pre_tool_block_checked=True, skip_tool_request_middleware=True, skip_tool_execution_middleware=True))['code'] == 'research_tool_denied'


def test_managed_dispatch_never_calls_underlying_bypass(monkeypatch):
    value = agent()
    value._tool_guardrails = SimpleNamespace(before_call=lambda *a: SimpleNamespace(allows_execution=True))
    monkeypatch.setattr(tool_executor, '_pre_tool_block', lambda *a: (None, {}))
    monkeypatch.setattr(tool_executor, '_begin_tool_execution', lambda *a: None)
    monkeypatch.setattr(tool_executor, '_run_with_activity_heartbeat', lambda agent, name, fn: fn())
    ref = SimpleNamespace(name='delegate_task',args={},task_id='test',call_id='test',trace=[])
    state = SimpleNamespace(args={})
    calls = []
    result = tool_executor._dispatch_authorized_once(value,state,ref,execute=lambda args: calls.append(args),scope_block=None,display_index=None,begin_execution=None,authorization_gate=None)
    assert json.loads(result)['code'] == 'research_tool_denied'
    assert calls == []
    value._preferred_web_required = False
    tool_executor._dispatch_authorized_once(value,state,ref,execute=lambda args: calls.append(args),scope_block=None,display_index=None,begin_execution=None,authorization_gate=None)
    assert calls == [{}]
