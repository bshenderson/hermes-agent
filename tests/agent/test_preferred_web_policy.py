"""Preserved GXTD-603 request-local restrictions."""
import pytest
from agent import preferred_web_policy

def module():
    return preferred_web_policy

def test_only_explicit_current_user_request_enables_mode():
    p = module()
    assert p.admit('compare these products', None, 'homelab-coordinator', True) is None
    policy = p.admit('/research compare https://example.com/a', None, 'homelab-coordinator', True)
    assert policy.allowed == {'https://example.com/a': 0}
    assert p.admit('follow up', None, 'homelab-coordinator', True) is None
    with pytest.raises(p.PolicyError):
        p.admit('/research task', None, 'default', True)
    with pytest.raises(p.PolicyError):
        p.admit('/research task', None, 'homelab-coordinator', False)
    with pytest.raises(p.PolicyError):
        p.admit('/research task', False, 'homelab-coordinator', True)


def test_dispatch_denies_wrappers_and_guesses_before_network(monkeypatch):
    import json
    p = module()
    policy = p.ResearchPolicy('/research https://example.com/a')
    calls = []
    monkeypatch.setattr(policy, '_guard', lambda cap: None, raising=False)
    monkeypatch.setattr(policy, '_extract', lambda url, allowed: calls.append(url), raising=False)
    for tool in ['terminal','execute_code','browser_exec','computer_use','delegate_task','handoff','tool_call','tool_search','skill_manage']:
        result = json.loads(policy.execute(tool, {}))
        assert result['code'] == 'research_tool_denied'
    result = json.loads(policy.execute('web_extract', {'urls':['https://example.com/a','https://example.com/guess']}))
    assert result['code'] == 'source_not_approved'
    assert calls == []


def test_successful_discovery_and_bounded_links_are_request_local(monkeypatch):
    import json
    p = module()
    policy = p.ResearchPolicy('/research sources')
    other = p.ResearchPolicy('/research sources')
    monkeypatch.setattr(policy, '_guard', lambda cap: None, raising=False)
    monkeypatch.setattr(policy, '_native_search', lambda q, n: json.dumps({'data': {'web': [{'url':'https://example.com/a','title':'A'}]}}), raising=False)
    result = json.loads(policy.execute('web_search', {'query':'sources'}))
    assert result['data']['web'][0]['url'] in policy.allowed
    assert not other.allowed
    monkeypatch.setattr(policy, '_post', lambda body: (200, {'success':True,'data':{'markdown':'Source text','links':['https://example.com/b','https://other.example/escape'],'metadata':{'policy':'gxtd603-v1','url':body['url'],'statusCode':200}}}, {}), raising=False)
    result = json.loads(policy.execute('web_extract', {'urls':['https://example.com/a']}))
    assert result['results'][0]['content'] == 'Source text'
    assert policy.allowed['https://example.com/b'] == 1
    assert 'https://other.example/escape' not in policy.allowed
    assert not other.allowed


def test_schema_and_execution_remain_closed_if_policy_state_is_missing():
    from types import SimpleNamespace
    import json
    p = module()
    agent = SimpleNamespace(tools=[{'type':'function','function':{'name':n}} for n in ['web_search','web_extract','terminal']], valid_tool_names={'terminal','web_search','web_extract'})
    assert p.dispatch_for_agent(agent, 'terminal', {}) is None
    p.bind_agent(agent, p.ResearchPolicy('/research task'))
    assert agent.valid_tool_names == {'web_search','web_extract'}
    assert [t['function']['name'] for t in agent.tools] == ['web_search','web_extract']
    assert json.loads(p.dispatch_for_agent(agent, 'delegate_task', {}))['code'] == 'research_tool_denied'
    agent._preferred_web_policy = None
    assert json.loads(p.dispatch_for_agent(agent, 'web_search', {}))['code'] == 'research_policy_missing'


def test_unsupported_routes_and_nonsticky_history():
    p = module()
    assert p.unsupported_request({'input':'/research task'}, '/v1/responses')
    assert p.unsupported_request({'preferred_web_mode':'research'}, '/v1/runs')
    assert not p.unsupported_request({'messages':[{'role':'user','content':'/research previous'},{'role':'user','content':'ordinary next'}]}, '/v1/responses')
    assert not p.unsupported_request({'messages':[{'role':'user','content':'/research task'}]}, '/p/homelab-coordinator/v1/chat/completions')


def test_mixed_success_failure_preserves_partial_and_bounded_retry(monkeypatch):
    import json
    p = module()
    policy = p.ResearchPolicy('/research https://example.com/a https://example.com/b')
    monkeypatch.setattr(policy, '_guard', lambda cap: None)
    calls = []
    def post(body):
        calls.append(body['url'])
        if body['url'].endswith('/b'):
            return 429, {'success':False}, {'retry-after':'60'}
        return 200, {'success':True,'data':{'markdown':'A source','links':[],'metadata':{'policy':'gxtd603-v1','url':body['url'],'statusCode':200}}}, {}
    monkeypatch.setattr(policy, '_post', post)
    result = json.loads(policy.execute('web_extract', {'urls':list(policy.allowed)}))
    assert result['results'][0]['content'] == 'A source'
    assert result['results'][1]['error'] == 'backend_rate_limited'
    policy.execute('web_extract', {'urls':['https://example.com/a']})
    assert len(calls) == 2
    policy.closed = True
    assert json.loads(policy.execute('web_search', {'query':'something'}))['code'] == 'research_request_closed'
