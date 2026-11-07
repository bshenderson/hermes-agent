"""GXTD-803 Tier2b through the existing real API constructor boundary."""
from unittest.mock import Mock

import pytest

from tests.gateway.test_api_server_elastic_precision import constructor, create, GEMMA, SPARKY
from gateway.platforms.api_server_precision_availability import probe_local

GPTOSS = 'gptoss-120b-openai-mxfp4-tp1-128k'


@pytest.mark.parametrize('intent,ready,expected,order', [
    ('responsive', {GEMMA}, GEMMA, [GEMMA]),
    ('responsive', {SPARKY, GPTOSS}, SPARKY, [GEMMA, SPARKY]),
    ('responsive', {GPTOSS}, GPTOSS, [GEMMA, SPARKY, GPTOSS]),
    ('substantial', {GPTOSS}, GPTOSS, [SPARKY, GPTOSS]),
])
def test_candidate_order_and_real_agent_binding(constructor, intent, ready, expected, order):
    constructor[1]['api_server']['precision_intent'] = intent
    constructor[3].side_effect = lambda base, model: {'available': model in ready}
    agent = create(constructor)
    assert agent.model == expected
    assert [call.args[1] for call in constructor[3].call_args_list] == order
    assert len(constructor[2]) == 1
    built = constructor[2][0]
    assert built['fallback_model'] is None
    assert built['api_key'] == 'fixture-local'
    assert built['enabled_toolsets'] == ['file']
    assert agent._hermes_precision_no_replay is True
    assert agent.request_overrides['extra_body']['homelab_queue_timeout_seconds'] == 0
    assert 'homelab_precision_warm_only' not in agent.request_overrides['extra_body'] or expected == GEMMA
    constructor[4].assert_not_called()


@pytest.mark.parametrize('reason', ['capacity_busy','backend_fenced','backend_cooldown',
    'backend_unavailable','observation_unavailable','unconfigured_model'])
def test_unknown_or_reserved_set_never_constructs_or_resolves_cloud(constructor, reason):
    constructor[3].return_value = {'available': False, 'reason': reason}
    with pytest.raises(RuntimeError, match='No model or tools dispatched'):
        create(constructor)
    assert [call.args[1] for call in constructor[3].call_args_list] == [GEMMA, SPARKY, GPTOSS]
    assert constructor[2] == []
    constructor[4].assert_not_called()


def test_gptoss_probe_uses_fixed_read_only_admission_transport(monkeypatch):
    import json
    response = Mock(status=200)
    response.read.return_value = json.dumps({'schema':'homelab.precision-availability.v1',
        'model':GPTOSS,'available':True,'context_length':131072}).encode()
    connection = Mock()
    connection.getresponse.return_value = response
    factory = Mock(return_value=connection)
    monkeypatch.setattr('http.client.HTTPConnection',factory)
    assert probe_local('http://127.0.0.1:8090/v1',GPTOSS)['context_length'] == 131072
    factory.assert_called_once_with('127.0.0.1',8090,timeout=2)
    connection.request.assert_called_once_with('GET','/routing/precision-availability?model='+GPTOSS)
    connection.close.assert_called_once()


def test_authenticated_executor_selects_gptoss_before_only_model_invocation(constructor, monkeypatch):
    import asyncio
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from aiohttp.test_utils import make_mocked_request
    from gateway.platforms import api_server as api
    import run_agent
    adapter, _, built, probe, resolver = constructor
    probe.side_effect = lambda base, model: {'available': model == GPTOSS, 'context_length': 131072}
    monkeypatch.setattr(adapter, '_profile_scope', lambda profile: nullcontext())
    adapter.gateway_runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    monkeypatch.setattr('hermes_cli.profiles.profiles_to_serve', lambda **_: [('homelab-coordinator', None)])
    monkeypatch.setattr(adapter, '_derive_browser_control_principal', lambda _: None)
    monkeypatch.setattr(adapter, '_browser_control_transport_family', lambda _: '')
    monkeypatch.setattr(adapter, '_expected_api_key', lambda: 'fixture-authenticated-key')
    monkeypatch.setattr(adapter, '_activate_admitted_request', lambda: None)
    monkeypatch.setattr(adapter, '_record_api_metrics', lambda *_: None)
    monkeypatch.setattr(adapter, '_finish_turn_result', lambda agent, result, *args, **kwargs: (result, {}))
    monkeypatch.setattr(api, '_publish_turn_process_ownership', lambda *_: None)
    monkeypatch.setattr(api, '_clear_turn_process_ownership', lambda *_: None)
    adapter._memory_sessions = MagicMock()
    original = run_agent.AIAgent
    generation = Mock(side_effect=RuntimeError('late admission refusal; never replay'))
    def build(**kwargs):
        agent = original(**kwargs)
        agent.run_conversation = generation
        return agent
    monkeypatch.setattr(run_agent, 'AIAgent', build)
    request = make_mocked_request('POST', '/p/homelab-coordinator/v1/chat/completions',
        headers={'Authorization': 'Bearer fixture-authenticated-key'}, match_info={'profile': 'homelab-coordinator'})
    async def handler(request):
        assert adapter._check_auth(request) is None
        assert api._api_request_profile.get() == 'homelab-coordinator'
        return await adapter._run_agent('ordinary precision request', [])
    with pytest.raises(RuntimeError, match='late admission refusal'):
        asyncio.run(adapter._make_profile_prefix_middleware()(request, handler))
    assert len(built) == 1 and built[0]['model'] == GPTOSS
    generation.assert_called_once()
    assert [call.args[1] for call in probe.call_args_list] == [GEMMA, SPARKY, GPTOSS] * 2
    resolver.assert_not_called()
