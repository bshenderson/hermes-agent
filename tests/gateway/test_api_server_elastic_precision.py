"""GXTD-743 real constructor and delegate producer boundary, offline only."""
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock

import pytest
import gateway.run as gateway
import run_agent
import asyncio
from contextlib import nullcontext
from aiohttp.test_utils import make_mocked_request

GEMMA = 'gemma4-31b-nvfp4-64k'
SPARKY = 'precision-default'


@pytest.mark.parametrize('history,options,selected', [
    ([], {}, GEMMA),
    ([{'role': 'user', 'content': 'prior'}, {'role': 'assistant', 'content': 'done'}], {}, GEMMA),
    ([{'role': 'user', 'content': 'uncertain'}], {}, SPARKY),
    ([{'role': 'assistant', 'tool_calls': [{'id': 'pending'}]}], {}, SPARKY),
    ([], {'resume_unanswered_turn': True}, SPARKY),
    ([], {'session_model': SPARKY}, SPARKY),
    ([], {'confirmed_runtime_lock': True}, SPARKY),
])
def test_authenticated_coordinator_executor_selects_startup_gemma(constructor, monkeypatch, history, options, selected):
    """Real ingress provenance -> executor -> real constructor, no model I/O."""
    from gateway.platforms import api_server as api
    adapter, _, built, _, _ = constructor
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
    monkeypatch.setattr('agent.session_persistence.adopt_unanswered_turn', lambda *_: None)
    adapter._memory_sessions = MagicMock()
    original = run_agent.AIAgent
    def build(**kwargs):
        agent = original(**kwargs)
        agent.run_conversation = Mock(return_value={'final_response': 'offline fixture'})
        return agent
    monkeypatch.setattr(run_agent, 'AIAgent', build)
    request = make_mocked_request('POST', '/p/homelab-coordinator/v1/chat/completions',
        headers={'Authorization': 'Bearer fixture-authenticated-key'}, match_info={'profile': 'homelab-coordinator'})
    async def handler(request):
        assert adapter._check_auth(request) is None
        assert api._api_request_profile.get() == 'homelab-coordinator'
        return await adapter._run_agent('ordinary homelab task', history, **options)
    asyncio.run(adapter._make_profile_prefix_middleware()(request, handler))
    assert built[0]['model'] == selected
    if selected != GEMMA:
        constructor[3].assert_not_called()


@pytest.fixture
def constructor(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    import gateway.run as gateway
    import run_agent
    from hermes_cli import tools_config
    from gateway.platforms import api_server_precision_availability as policy
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    config = {'api_server': {'precision_elastic_first': True, 'precision_intent': 'responsive'}}
    monkeypatch.setattr(gateway, '_resolve_runtime_agent_kwargs', lambda: {
        'provider': 'custom', 'base_url': 'http://127.0.0.1:8090/v1',
        'api_key': 'fixture-local', 'api_mode': 'chat_completions'})
    monkeypatch.setattr(gateway, '_resolve_gateway_model', lambda: SPARKY)
    monkeypatch.setattr(gateway, '_load_gateway_config', lambda: config)
    monkeypatch.setattr(gateway.GatewayRunner, '_load_reasoning_config', lambda _: None)
    monkeypatch.setattr(gateway.GatewayRunner, '_load_fallback_model', lambda: None)
    monkeypatch.setattr(gateway, '_checkpoint_agent_kwargs', lambda _: {})
    monkeypatch.setattr(gateway, '_current_max_iterations', lambda: 4)
    monkeypatch.setattr(tools_config, '_get_platform_tools', lambda *_: ['file'])
    monkeypatch.setattr(adapter, '_ensure_session_db', lambda: None)
    monkeypatch.setattr(adapter, '_select_agent_runtime', lambda kw, model, **_: (model, None, None, None))
    resolver = Mock(side_effect=AssertionError('no cloud or credential resolution authorized'))
    monkeypatch.setattr(adapter, '_resolve_provider_runtime', resolver)
    built = []
    def build(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(model=kwargs['model'], provider=kwargs['provider'])
    monkeypatch.setattr(run_agent, 'AIAgent', build)
    probe = Mock(return_value={'available': True, 'reason': 'ready'})
    monkeypatch.setattr(policy, 'probe_local', probe)
    return adapter, config, built, probe, resolver


def create(fixture, **kwargs):
    return fixture[0]._create_agent(precision_availability_eligible=True,
        precision_elastic_eligible=True,
        precision_availability_profile='homelab-coordinator', **kwargs)


def test_real_constructor_selects_gemma_and_keeps_provider_bundle(constructor):
    agent = create(constructor)
    assert agent.model == GEMMA
    kwargs = constructor[2][0]
    assert (kwargs['provider'], kwargs['api_key'], kwargs['base_url'], kwargs['api_mode']) == (
        'custom', 'fixture-local', 'http://127.0.0.1:8090/v1', 'chat_completions')
    assert kwargs['fallback_model'] is None
    assert kwargs['enabled_toolsets'] == ['file']
    assert agent.request_overrides['extra_body']['homelab_queue_timeout_seconds'] == 0
    constructor[4].assert_not_called()


@pytest.mark.parametrize('reason', ['capacity_busy', 'backend_fenced', 'backend_unavailable',
    'observation_unavailable', 'elastic_not_warm', 'backend_cooldown'])
def test_unready_gemma_uses_sparky_before_construction(constructor, reason):
    constructor[3].side_effect = lambda base, model: {
        'available': model == SPARKY, 'reason': 'ready' if model == SPARKY else reason}
    assert create(constructor).model == SPARKY
    assert len(constructor[2]) == 1
    assert constructor[2][0]['fallback_model'] is None
    constructor[4].assert_not_called()


def test_both_unavailable_refuses_before_agent_or_tools(constructor):
    constructor[3].return_value = {'available': False, 'reason': 'capacity_busy'}
    with pytest.raises(RuntimeError, match='No model or tools dispatched'):
        create(constructor)
    assert constructor[2] == []
    constructor[4].assert_not_called()


def test_capacity_refusal_keeps_failed_outcome_through_real_executor(constructor, monkeypatch):
    from gateway.platforms import api_server as api

    adapter, _, built, probe, resolver = constructor
    monkeypatch.setattr(adapter, '_profile_scope', lambda _: nullcontext())
    probe.return_value = {'available': False, 'reason': 'capacity_busy'}
    token = api._api_request_profile.set('homelab-coordinator')
    try:
        result, usage = asyncio.run(adapter._run_agent('capacity fixture', []))
    finally:
        api._api_request_profile.reset(token)
    assert result.get('failed') is True
    assert result.get('completed') is False
    assert result.get('error_code') == 'precision_unavailable'
    assert result['api_calls'] == 0 and result['tools'] == []
    assert 'No model or tools dispatched' in result['final_response']
    assert 'authentication' not in result['final_response'].lower()
    assert usage['total_tokens'] == 0
    assert built == []
    assert probe.call_count == 3
    resolver.assert_not_called()


@pytest.mark.parametrize('stream', [False, True])
def test_capacity_refusal_is_failed_on_authenticated_http_ingress(constructor, monkeypatch, stream):
    import json
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.platforms import api_server as api

    adapter, _, built, probe, resolver = constructor
    probe.return_value = {'available': False, 'reason': 'capacity_busy'}
    monkeypatch.setattr(adapter, '_profile_scope', lambda _: nullcontext())
    adapter.gateway_runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    monkeypatch.setattr('hermes_cli.profiles.profiles_to_serve', lambda **_: [('homelab-coordinator', None)])
    monkeypatch.setattr(adapter, '_derive_browser_control_principal', lambda _: None)
    monkeypatch.setattr(adapter, '_browser_control_transport_family', lambda _: '')
    monkeypatch.setattr(adapter, '_expected_api_key', lambda: 'fixture-authenticated-key')
    monkeypatch.setattr(adapter, '_activate_admitted_request', lambda: None)
    monkeypatch.setattr(adapter, '_record_api_metrics', lambda *_: None)
    monkeypatch.setattr(adapter, '_select_request_route', lambda *_, **__: (None, {}, None))
    adapter._memory_sessions = MagicMock()

    async def handle(request):
        assert api._api_request_profile.get() == 'homelab-coordinator'
        rejected = adapter._check_auth(request)
        if rejected is not None:
            return rejected
        return await adapter._handle_chat_completions(request)

    async def trial():
        app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
        app.router.add_post('/p/{profile}/v1/chat/completions', handle)
        async with TestClient(TestServer(app)) as client:
            response = await client.post('/p/homelab-coordinator/v1/chat/completions',
                headers={'Authorization': 'Bearer fixture-authenticated-key'},
                json={'model': 'homelab-coordinator', 'stream': stream,
                      'messages': [{'role': 'user', 'content': 'capacity fixture'}]})
            assert response.status == 200  # Existing in-band HTTP/SSE failure contract.
            raw = await response.text()
            if stream:
                chunks = [json.loads(line[6:]) for line in raw.splitlines()
                          if line.startswith('data: ') and line != 'data: [DONE]']
                payload = next(c for c in chunks if c.get('hermes', {}).get('failed'))
                assert 'data: [DONE]' in raw
            else:
                payload = json.loads(raw)
                assert response.headers['X-Hermes-Completed'] == 'false'
            assert payload['choices'][0]['finish_reason'] == 'error'
            assert payload['hermes']['failed'] is True
            assert payload['hermes']['completed'] is False
            assert 'authentication failed' not in raw.lower()

    asyncio.run(trial())
    assert built == [] and probe.call_count == 3
    resolver.assert_not_called()


def test_substantial_is_explicit_producer_intent_not_prompt_guessing(constructor):
    constructor[1]['api_server']['precision_intent'] = 'substantial'
    assert create(constructor).model == SPARKY
    assert all(call.args[1] == SPARKY for call in constructor[3].call_args_list)


def test_existing_history_and_other_profiles_do_not_export_or_reselect(constructor):
    adapter = constructor[0]
    assert adapter._create_agent(precision_availability_eligible=False,
        precision_availability_profile='homelab-coordinator').model == SPARKY
    assert adapter._create_agent(precision_availability_eligible=True,
        precision_availability_profile='personal-barry').model == SPARKY
    constructor[3].assert_not_called()


@pytest.mark.parametrize('option', ['requested_model', 'requested_provider', 'route',
    'session_model', 'confirmed_runtime_lock'])
def test_explicit_selections_do_not_probe(constructor, option):
    values = {'requested_model': 'qwen38-27b-nvfp4-256k', 'requested_provider': 'custom:other',
        'route': {'model': 'qwen36-27b-nvfp4-256k'}, 'session_model': GEMMA,
        'confirmed_runtime_lock': True}
    # The precedence resolver is separately tested; expose its explicit-selection outputs.
    if option in {'requested_model', 'requested_provider'}:
        constructor[0]._select_agent_runtime = lambda kw, model, **_: (
            model, None, values.get('requested_model') if option == 'requested_model' else None,
            values.get('requested_provider') if option == 'requested_provider' else None)
    create(constructor, **{option: values[option]})
    constructor[3].assert_not_called()


@pytest.mark.parametrize('branch', ['inherit', 'direct', 'provider', 'direct-provider'])
def test_real_delegate_producer_pins_substantial_sparky_not_parent_gemma(monkeypatch, branch):
    from tools.delegate_tool import _build_child_agent
    from tools.delegate_tool_config import _resolve_delegation_credentials
    from tests.tools.test_delegate import _make_mock_parent
    import run_agent
    parent = _make_mock_parent(depth=0)
    parent.model = GEMMA
    parent.provider = 'custom'
    parent.base_url = 'http://127.0.0.1:8090/v1'
    parent.api_key = 'fixture-local'
    parent._hermes_precision_no_replay = True
    parent.request_overrides = {'extra_body': {'homelab_precision_warm_only': True,
        'homelab_queue_timeout_seconds': 0, 'unrelated': 'preserve'}}
    cfg = {'model': SPARKY, 'fallback_providers': []}
    if branch in {'direct', 'direct-provider'}:
        cfg['base_url'] = parent.base_url
    if branch in {'provider', 'direct-provider'}:
        cfg['provider'] = 'llamacpp-main'
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_provider', lambda **_: {
        'provider': 'custom', 'base_url': parent.base_url, 'api_key': 'fixture-child',
        'api_mode': 'chat_completions', 'request_overrides': {'extra_body': {'unrelated': 'preserve'}}})
    credentials = _resolve_delegation_credentials(cfg, parent)
    monkeypatch.setattr('tools.delegate_tool._load_config', lambda: cfg)
    built = Mock(return_value=MagicMock())
    monkeypatch.setattr(run_agent, 'AIAgent', built)
    child = _build_child_agent(task_index=0, goal='substantial producer fixture', context=None,
        toolsets=None, model=credentials['model'], max_iterations=4, parent_agent=parent,
        task_count=1, override_request_overrides=credentials['request_overrides'],
        override_provider=credentials['provider'], override_base_url=credentials['base_url'],
        override_api_key=credentials['api_key'], override_api_mode=credentials['api_mode'])
    assert built.call_args.kwargs['model'] == SPARKY
    assert built.call_args.kwargs['fallback_model'] is None
    extra = built.call_args.kwargs['request_overrides'].get('extra_body', {})
    assert 'homelab_precision_warm_only' not in extra
    assert 'homelab_queue_timeout_seconds' not in extra
    assert child._hermes_precision_no_replay is True
    assert built.call_args.kwargs['skip_memory'] is True
    assert parent.request_overrides['extra_body']['homelab_precision_warm_only'] is True


@pytest.mark.parametrize('model', ['qwen38-27b-nvfp4-256k', 'qwen36-27b-nvfp4-256k',
    'qwen36-35b-nvfp4-256k', 'qwen36-hauhau-35b-lyf-nvfp4-256k', GEMMA,
    'gemma4-31b-nvidia-nvfp4-mlp-64k', 'chat-default'])
def test_explicit_alternatives_and_lyf_never_enter_precision(model):
    from gateway.platforms.api_server_precision_availability import select_elastic_runtime
    probe = Mock(side_effect=AssertionError('no selector normalization'))
    assert select_elastic_runtime(enabled=True, profile='homelab-coordinator', model=model,
        explicit_selection=False, intent='responsive', probe=probe) is None
    probe.assert_not_called()


@pytest.mark.parametrize('intent', [None, '', 'auto', 'short', True])
def test_unknown_intent_fails_before_readiness_or_credentials(constructor, intent):
    constructor[1]['api_server']['precision_intent'] = intent
    with pytest.raises(RuntimeError, match='producer intent'):
        create(constructor)
    constructor[3].assert_not_called()
    constructor[4].assert_not_called()
    assert not constructor[2]


def test_probe_exception_uses_only_remaining_local_candidate(constructor):
    def probe(base, model):
        if model == GEMMA:
            raise OSError('fixture observation unknown')
        return {'available': True}
    constructor[3].side_effect = probe
    agent = create(constructor)
    assert agent.model == SPARKY
    assert agent._hermes_precision_no_replay is True
    constructor[4].assert_not_called()


@pytest.mark.parametrize('profile', ['homelab-coordinator', 'homelab-ops', 'homelab-coding',
    'ops-admin', 'ops-engineer', 'ops-monitor', 'default', 'operator', 'personal-barry',
    'govcon', 'research', 'media', None])
@pytest.mark.parametrize('enabled', [True, False, 'true'])
def test_local_grant_is_exact_six_profiles_and_strict_optin(profile, enabled):
    from gateway.platforms.api_server_precision_availability import select_elastic_runtime, PROFILES
    probe = Mock(return_value={'available': True})
    result = select_elastic_runtime(enabled=enabled, profile=profile, model=SPARKY,
        explicit_selection=False, intent='responsive', probe=probe)
    if enabled is True and profile in PROFILES:
        assert result is not None
        assert result['model'] == GEMMA
    else:
        assert result is None
        probe.assert_not_called()
