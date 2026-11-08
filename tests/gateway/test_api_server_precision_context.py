"""Estimated context selection; offline, no model or live admission calls."""
from unittest.mock import Mock

import pytest
from tests.gateway.test_api_server_elastic_precision import constructor, create

from gateway.platforms.api_server_precision_availability import (
    select_elastic_runtime, ELASTIC_MODEL, GPTOSS_MODEL, AvailabilityRefused,
)


def test_ready_small_window_is_skipped_before_execution():
    seen = []
    def probe(model):
        seen.append(model)
        return {'available': True, 'context_length': {
            ELASTIC_MODEL: 65536, 'precision-default': 262144, GPTOSS_MODEL: 131072,
        }[model]}
    result = select_elastic_runtime(
        enabled=True, profile='homelab-coordinator', model='precision-default',
        explicit_selection=False, intent='responsive', probe=probe,
        required_context=65537,
    )
    assert result['model'] == 'precision-default'
    assert result['context_length'] == 262144
    assert seen == [ELASTIC_MODEL, 'precision-default']


def test_prompt_preview_does_not_persist_or_invoke_session_hooks(monkeypatch):
    from unittest.mock import Mock
    from tests.agent.test_system_prompt_restore import _make_agent
    from agent import conversation_loop as loop
    agent = _make_agent()
    agent._persist_disabled = False
    persist = Mock(side_effect=AssertionError('preview must not persist'))
    hook = Mock(side_effect=AssertionError('preview must not invoke lifecycle'))
    monkeypatch.setattr(loop, '_persist_system_prompt', persist)
    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', hook)
    loop._restore_or_build_system_prompt(agent, None, [], prepare_only=True)
    assert agent._cached_system_prompt == 'BUILT_PROMPT'
    persist.assert_not_called()
    hook.assert_not_called()


def test_preparation_prices_system_tools_history_prefill_and_output_without_mutation():
    from copy import deepcopy
    from tests.agent.test_system_prompt_restore import _make_agent
    from agent.model_metadata import estimate_request_tokens_rough
    from gateway.platforms.api_server_precision_context import prepare_requirement
    agent = _make_agent(prebuilt_prompt='system ' * 100)
    agent.max_tokens = 2048
    agent.tools = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {}, 'description': 'schema ' * 100}}]
    agent.prefill_messages = [{'role': 'assistant', 'content': 'prefill ' * 20}]
    history = [{'role': 'user', 'content': 'short', 'api_content': 'injected ' * 200}, {'role': 'assistant', 'content': 'done'}]
    before = deepcopy(history)
    expected = estimate_request_tokens_rough(
        agent.prefill_messages + [{'role': 'user', 'content': 'injected ' * 200}, history[1], {'role': 'user', 'content': 'next'}],
        system_prompt='system ' * 100, tools=agent.tools,
    ) + 2048
    assert prepare_requirement(agent, 'next', history) == expected
    assert history == before
    assert agent._cached_system_prompt is None  # ordinary turn still owns lifecycle


def test_constructor_uses_prepared_context_before_dispatch(constructor, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import run_agent
    from gateway.platforms import api_server_precision_context as context
    _adapter, _, built, probe, _ = constructor
    switched = []
    def build(**kw):
        built.append(kw)
        agent = SimpleNamespace(model=kw['model'], provider=kw['provider'], max_tokens=8192)
        def switch_model(**destination):
            switched.append(destination['new_model'])
            agent.model = destination['new_model']
        agent.switch_model = switch_model
        return agent
    monkeypatch.setattr(run_agent, 'AIAgent', build)
    prepare = Mock(return_value=70000)
    monkeypatch.setattr(context, 'prepare_requirement', prepare)
    probe.side_effect = lambda base, model: {'available': True, 'context_length': {
        ELASTIC_MODEL: 65536, 'precision-default': 262144, GPTOSS_MODEL: 131072,
    }[model]}
    agent = create(constructor, precision_context=('new user', [{'role': 'assistant', 'content': 'complete'}]))
    assert agent.model == 'precision-default'
    assert switched == ['precision-default']
    assert agent.request_overrides['extra_body']['homelab_required_context'] == 70000
    assert agent._hermes_precision_context_window == 262144
    assert 'homelab_precision_warm_only' not in agent.request_overrides['extra_body']
    assert prepare.call_count == 2  # reprices final model after in-place setup


def test_prepare_only_does_not_persist_tool_pin():
    from unittest.mock import Mock
    from tests.agent.test_system_prompt_restore import _make_agent
    from agent import conversation_loop as loop
    db = Mock()
    db.get_session.return_value = {'system_prompt': None, 'tool_names': None}
    agent = _make_agent(session_db=db)
    agent._persist_disabled = False
    persist_tools = Mock(side_effect=AssertionError('preparation must not write tool pins'))
    history = [{'role': 'user', 'content': 'prior'}, {'role': 'assistant', 'content': 'done'}]
    import tools.mcp_tool_agent as mcp
    from unittest.mock import patch
    with patch.object(mcp, 'persist_agent_tool_names', persist_tools), \
         patch.object(loop, '_persist_system_prompt'):
        loop._restore_or_build_system_prompt(agent, None, history, prepare_only=True)
    persist_tools.assert_not_called()
    # The ordinary turn still owns the pin writes (restore + persist tail).
    with patch.object(mcp, 'persist_agent_tool_names') as persist_tools_normal:
        loop._restore_or_build_system_prompt(agent, None, history)
    assert persist_tools_normal.call_count == 2


def test_stored_valid_restore_with_missing_pin_respects_prepare_only():
    """GXTD-803-IR-1: the runtime-valid stored-prompt branch must not write a tool pin.

    A legacy/swept row (tool_names absent) on a continuing session reaches
    ``_restore_pinned_tools`` from the STORED-VALID branch. Without prepare_only
    propagation there, a preview pass persists the pin for a turn that may never
    be sent and freezes tools[] for later turns. This drives the real helper:
    the sentinel sits on ``persist_agent_tool_names`` inside the helper body.
    """
    from unittest.mock import Mock, patch
    from tests.agent.test_system_prompt_restore import _make_agent
    from agent import conversation_loop as loop
    import tools.mcp_tool_agent as mcp
    stored = ('SYSTEM BODY\n\nModel: test-model\nProvider: openrouter\nPlatform: cli')
    db = Mock()
    # No 'tool_names' key at all: the missing-pin (legacy/swept) row state.
    db.get_session.return_value = {'system_prompt': stored}
    agent = _make_agent(session_db=db)
    agent._persist_disabled = False
    agent._platform_hint_overrides = None
    agent._surface_switch_note = ""
    agent._gateway_turn_context_notes = ""
    history = [{'role': 'user', 'content': 'prior'}, {'role': 'assistant', 'content': 'done'}]
    persist_tools = Mock()
    with patch.object(mcp, 'persist_agent_tool_names', persist_tools):
        loop._restore_or_build_system_prompt(agent, None, history, prepare_only=True)
    # Stored bytes priced, no rebuild, and NO pin write from the preview pass.
    assert agent._cached_system_prompt == stored
    assert agent._build_system_prompt.call_count == 0
    persist_tools.assert_not_called()
    # The ordinary turn still owns the pin on this same branch/row state.
    persist_normal = Mock()
    with patch.object(mcp, 'persist_agent_tool_names', persist_normal):
        loop._restore_or_build_system_prompt(agent, None, history)
    assert persist_normal.call_count == 1


def test_prepared_prompt_build_is_consumed_once_by_the_normal_turn():
    from tests.agent.test_system_prompt_restore import _make_agent
    from agent import conversation_loop as loop
    from unittest.mock import Mock, patch
    agent = _make_agent()
    import tools.mcp_tool_agent as mcp
    with patch.object(mcp, 'persist_agent_tool_names'), patch.object(loop, '_persist_system_prompt'):
        loop._restore_or_build_system_prompt(agent, None, [], prepare_only=True)
    assert agent._cached_system_prompt == 'BUILT_PROMPT'
    assert agent._prepared_system_prompt_build is not None
    agent._cached_system_prompt = None  # same clear prepare_requirement performs
    with patch.object(mcp, 'persist_agent_tool_names'), patch.object(loop, '_persist_system_prompt'):
        loop._restore_or_build_system_prompt(agent, None, [])
    # One builder callback total: the normal turn consumed the stashed bytes.
    assert agent._build_system_prompt.call_count == 1
    assert agent._cached_system_prompt == 'BUILT_PROMPT'
    assert agent._prepared_system_prompt_build is None  # single-consume


def test_stash_invalidates_when_model_identity_moves():
    from tests.agent.test_system_prompt_restore import _make_agent
    from agent import conversation_loop as loop
    from unittest.mock import patch
    agent = _make_agent()
    import tools.mcp_tool_agent as mcp
    with patch.object(mcp, 'persist_agent_tool_names'), patch.object(loop, '_persist_system_prompt'):
        loop._restore_or_build_system_prompt(agent, None, [], prepare_only=True)
    agent._cached_system_prompt = None
    agent.model = 'switched-model'  # switch_model invalidates _cached_system_prompt
    with patch.object(mcp, 'persist_agent_tool_names'), patch.object(loop, '_persist_system_prompt'):
        loop._restore_or_build_system_prompt(agent, None, [])
    assert agent._build_system_prompt.call_count == 2  # stale stash must not survive the switch


def test_bot_chat_stale_capability_prices_stored_prompt_without_refresh(monkeypatch):
    from unittest.mock import Mock, patch
    from tests.agent.test_system_prompt_restore import _make_agent
    from agent import conversation_loop as loop
    stored = ('SYSTEM BODY\n\nModel: test-model\nProvider: openrouter\nPlatform: cli')
    db = Mock()
    db.get_session.return_value = {'system_prompt': stored, 'tool_names': None}
    agent = _make_agent(session_db=db)
    monkeypatch.setattr(loop, '_bot_chat_prompt_stale', lambda a, p: True)
    refresh = Mock(side_effect=AssertionError('preparation must not refresh tools'))
    clear = Mock(side_effect=AssertionError('preparation must not clear the global skills cache'))
    record = Mock(side_effect=AssertionError('preparation must not record a cache break'))
    import tools.mcp_tool_agent as mcp
    with patch.object(loop, '_refresh_bot_chat_tools', refresh), \
         patch('agent.prompt_builder.clear_skills_system_prompt_cache', clear), \
         patch.object(loop, 'record_cache_break', record), \
         patch.object(mcp, 'persist_agent_tool_names'), \
         patch.object(loop, '_persist_system_prompt'):
        loop._restore_or_build_system_prompt(agent, None,
            [{'role': 'user', 'content': 'prior'}, {'role': 'assistant', 'content': 'done'}],
            prepare_only=True)
    # Fails closed to the runtime-valid stored bytes; the ordinary turn owns the
    # once-per-change refresh, and the request-assembly guard reprices it.
    assert agent._cached_system_prompt == stored
    refresh.assert_not_called()
    clear.assert_not_called()
    record.assert_not_called()
    assert agent._build_system_prompt.call_count == 0


def test_rebuild_telemetry_recorded_once_across_preparation_and_normal_turn():
    from tests.agent.test_system_prompt_restore import _make_agent
    from agent import conversation_loop as loop
    from unittest.mock import patch
    agent = _make_agent()
    history = [{'role': 'user', 'content': 'prior'}, {'role': 'assistant', 'content': 'done'}]
    import tools.mcp_tool_agent as mcp
    with patch.object(mcp, 'persist_agent_tool_names'), patch.object(loop, '_persist_system_prompt'), \
         patch.object(loop, 'record_prompt_rebuild') as record:
        loop._restore_or_build_system_prompt(agent, None, history, prepare_only=True)
        record.assert_not_called()
        agent._cached_system_prompt = None
        loop._restore_or_build_system_prompt(agent, None, history)
    record.assert_called_once()


def _precision_ingress_app(adapter, api):
    from aiohttp import web

    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])

    async def handle(request):
        rejected = adapter._check_auth(request)
        if rejected is not None:
            return rejected
        return await adapter._handle_chat_completions(request)

    app.router.add_post('/p/{profile}/v1/chat/completions', handle)
    return app


@pytest.mark.parametrize('stream', [False, True])
def test_real_ingress_history_size_moves_route_through_prepared_accounting(monkeypatch, tmp_path, stream):
    """Authenticated ingress -> real constructor -> real AIAgent -> real preparation.

    Scripted: loopback readiness windows only. Blocked: model dispatch
    (``run_conversation`` replaced at the boundary). History size alone must flip
    the chosen route when the prepared request crosses the elastic window.
    """
    import asyncio
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms import api_server as api
    from gateway.platforms import api_server_precision_availability as policy
    import run_agent
    from hermes_cli import tools_config
    import gateway.run as gateway

    (tmp_path / 'config.yaml').write_text('model:\n  max_tokens: 4096\n', encoding='utf-8')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('TERMINAL_CWD', str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    config = {'api_server': {'precision_elastic_first': True, 'precision_intent': 'responsive'},
              'model': {'max_tokens': 4096}}
    monkeypatch.setattr(gateway, '_resolve_runtime_agent_kwargs', lambda: {
        'provider': 'custom', 'base_url': 'http://127.0.0.1:8090/v1',
        'api_key': 'fixture-local', 'api_mode': 'chat_completions'})
    monkeypatch.setattr(gateway, '_resolve_gateway_model', lambda: 'precision-default')
    monkeypatch.setattr(gateway, '_load_gateway_config', lambda: config)
    monkeypatch.setattr(gateway.GatewayRunner, '_load_reasoning_config', lambda _: None)
    monkeypatch.setattr(gateway.GatewayRunner, '_load_fallback_model', lambda: None)
    monkeypatch.setattr(gateway, '_checkpoint_agent_kwargs', lambda _: {})
    monkeypatch.setattr(gateway, '_current_max_iterations', lambda: 4)
    monkeypatch.setattr(tools_config, '_get_platform_tools', lambda *_: [])
    monkeypatch.setattr(adapter, '_ensure_session_db', lambda: None)
    monkeypatch.setattr(adapter, '_select_agent_runtime', lambda kw, model, **_: (model, None, None, None))
    monkeypatch.setattr(adapter, '_profile_scope', lambda _: nullcontext())
    monkeypatch.setattr(adapter, '_expected_api_key', lambda: 'fixture-authenticated-key')
    monkeypatch.setattr(adapter, '_activate_admitted_request', lambda: None)
    monkeypatch.setattr(adapter, '_record_api_metrics', lambda *_: None)
    monkeypatch.setattr(adapter, '_select_request_route', lambda *_, **__: (None, {}, None))
    monkeypatch.setattr(api, '_publish_turn_process_ownership', lambda *_: None)
    monkeypatch.setattr(api, '_clear_turn_process_ownership', lambda *_: None)
    adapter._memory_sessions = MagicMock()
    adapter.gateway_runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    monkeypatch.setattr('hermes_cli.profiles.profiles_to_serve', lambda **_: [('homelab-coordinator', None)])

    # Scripted readiness only: every candidate ready, distinct windows.
    monkeypatch.setattr(policy, 'probe_local', lambda base, model: {'available': True,
        'context_length': {policy.ELASTIC_MODEL: 65536, 'precision-default': 262144,
                           policy.GPTOSS_MODEL: 131072}[model]})
    # Blocked model dispatch: the boundary that would send tokens to any engine.
    generation = Mock(return_value={'final_response': 'offline fixture', 'messages': []})
    built = []
    agents = []
    original = run_agent.AIAgent

    def build(**kwargs):
        built.append(kwargs)
        agent = original(**kwargs)
        agent.run_conversation = generation
        agents.append(agent)
        return agent
    monkeypatch.setattr(run_agent, 'AIAgent', build)

    async def trial():
        async with TestClient(TestServer(_precision_ingress_app(adapter, api))) as client:
            small = await client.post('/p/homelab-coordinator/v1/chat/completions',
                headers={'Authorization': 'Bearer fixture-authenticated-key'},
                json={'model': 'homelab-coordinator', 'stream': stream, 'messages': [
                    {'role': 'user', 'content': 'short'}]})
            assert small.status == 200
            small_body = await small.text()
            if stream:
                assert 'text/event-stream' in small.headers['Content-Type']
                assert 'data: [DONE]' in small_body
            assert 'offline fixture' in small_body
            # Ingress caps each content part at 65,536 chars; 7 capped rows
            # (4 user + 3 assistant, ~16.4k tokens each) price past the 65536
            # elastic window through the production estimator, inside the 262144
            # Sparky window. The final user row is the live user message.
            big = await client.post('/p/homelab-coordinator/v1/chat/completions',
                headers={'Authorization': 'Bearer fixture-authenticated-key'},
                json={'model': 'homelab-coordinator', 'stream': stream, 'messages': [
                    {'role': 'user' if i % 2 == 0 else 'assistant', 'content': 'u' * 65536}
                    for i in range(7)]})
            assert big.status == 200
            big_body = await big.text()
            if stream:
                assert 'text/event-stream' in big.headers['Content-Type']
                assert 'data: [DONE]' in big_body
            assert 'offline fixture' in big_body
    asyncio.run(trial())
    assert built[0]['model'] == policy.ELASTIC_MODEL
    assert agents[0].model == policy.ELASTIC_MODEL
    assert generation.call_count == 2
    # The large turn first constructed the elastic-preferred row, then the REAL
    # preparation repriced past 65536 and switched to the only sufficient window.
    assert built[1]['model'] == policy.ELASTIC_MODEL
    assert agents[1].model == 'precision-default'
    assert agents[1]._hermes_precision_context_window == 262144
    declared = agents[1].request_overrides['extra_body']['homelab_required_context']
    assert 65536 < declared <= 262144
    assert agents[0].request_overrides['extra_body']['homelab_required_context'] <= 65536


def test_prompt_tool_history_and_output_each_move_the_selected_route():
    """Each contribution changes the REAL prepared requirement, and the requirement
    alone changes the route: base fits the small window, each grown variant does not."""
    from tests.agent.test_system_prompt_restore import _make_agent
    from gateway.platforms import api_server_precision_context as ctx
    from gateway.platforms.api_server_precision_availability import (
        ELASTIC_MODEL, GPTOSS_MODEL, AvailabilityRefused)

    probe = lambda model: {'available': True, 'context_length': {
        ELASTIC_MODEL: 65536, 'precision-default': 262144, GPTOSS_MODEL: 131072}[model]}

    def required(**changes):
        agent = _make_agent(prebuilt_prompt='sys ' + 'x' * (300000 if changes.get('system') else 4))
        agent.max_tokens = 66000 if changes.get('output') else 1200
        agent.tools = ([{'type': 'function', 'function': {
            'name': 'wide', 'description': 'y' * 300000, 'parameters': {}}}]
            if changes.get('tools') else [])
        agent.prefill_messages = []
        history = ([{'role': 'user', 'content': 'z' * 300000}] if changes.get('history') else [])
        return ctx.prepare_requirement(agent, 'next', history)

    def route(value):
        selected = select_elastic_runtime(enabled=True, profile='homelab-coordinator',
            model='precision-default', explicit_selection=False, intent='responsive',
            probe=probe, required_context=value)
        assert selected is not None
        return selected['model']

    base = required()
    assert route(base) == ELASTIC_MODEL
    for change in ('system', 'tools', 'history', 'output'):
        grown = required(**{change: True})
        assert grown > base, change
        assert grown > 65536, change
        assert route(grown) == 'precision-default', change


def test_real_assembly_late_growth_is_terminal_break_with_fresh_estimate():
    """Full request-assembly pass through the guard: growth refuses, the outer loop
    breaks, and the declared estimate tracks the final assembled request."""
    import copy
    import logging
    from agent.turn_loop_errors import PrecisionContextRefused, handle_outer_loop_error
    from agent.turn_request_assembly import assemble_api_request
    from agent.model_metadata import estimate_request_tokens_rough
    from gateway.platforms.api_server_precision_context import guard_prepared_request

    def agent_with(toolsets):
        agent = Mock()
        agent.model = 'm'
        agent.provider = 'custom'
        agent.api_mode = 'chat_completions'
        agent.base_url = 'http://127.0.0.1:8090/v1'
        agent._use_prompt_caching = False
        agent.prefill_messages = []
        agent.tools = toolsets
        agent.max_tokens = 2048
        agent._usage_anchor = None
        agent._current_turn_timestamp = 1.0
        agent.ephemeral_system_prompt = ''
        agent._hermes_precision_context_window = 65536
        agent._hermes_precision_context_guard = guard_prepared_request
        agent.request_overrides = {'extra_body': {'homelab_queue_timeout_seconds': 0}}
        agent._copy_reasoning_content_for_api = lambda s, t: None
        agent._outbound_retired_image_ids = None
        agent._should_sanitize_tool_calls = lambda: False
        agent._sanitize_api_messages = lambda v: v
        agent._emit_warning = agent._vprint = lambda *a, **k: None
        agent._drop_thinking_only_and_merge_users = lambda v, **k: v
        return agent

    history = [{'role': 'user', 'content': 'hello'},
               {'role': 'assistant', 'content': 'done'},
               {'role': 'user', 'content': 'next'}]
    small = agent_with([])
    with _patched_assembly():
        first = assemble_api_request(
            small, messages=copy.deepcopy(history), current_turn_user_idx=2,
            _ext_prefetch_cache='', _plugin_user_context='', moa_config=None,
            active_system_prompt='', original_user_message='next',
            pending_moa_prepared_request=None,
            request_logger=logging.getLogger('gxtd803-assembly'))
    declared = small.request_overrides['extra_body']['homelab_required_context']
    expected_small = estimate_request_tokens_rough(
        first.api_messages, system_prompt='', tools=small.tools) + 2048
    assert declared == expected_small

    # Same turn, injected context growth before the wire: the guard refuses at assembly.
    grown = agent_with([])
    grown._hermes_precision_context_window = 65536
    with _patched_assembly(grow=True):
        with pytest.raises(PrecisionContextRefused):
            assemble_api_request(
                grown, messages=copy.deepcopy(history), current_turn_user_idx=2,
                _ext_prefetch_cache='', _plugin_user_context='', moa_config=None,
                active_system_prompt='', original_user_message='next',
                pending_moa_prepared_request=None,
                request_logger=logging.getLogger('gxtd803-assembly'))
    verdict = handle_outer_loop_error(grown, e=PrecisionContextRefused('grown'),
        _outer_error_count=0, api_call_count=0, messages=list(history),
        conversation_history=[], _turn_exit_reason='', failed=False, final_response='')
    assert verdict.action == 'break' and verdict.failed is True
    assert verdict._turn_exit_reason == 'precision_context_unavailable'
    # No route replay: nothing re-selected, the declared estimate never advanced.
    assert 'homelab_required_context' not in grown.request_overrides['extra_body']


class _patched_assembly:
    """Neutralize the optional hooks the precision path does not own."""

    def __init__(self, grow=False):
        self.grow = grow

    def __enter__(self):
        import agent.conversation_loop as loop
        from unittest.mock import patch
        self._patches = [
            patch.object(loop, '_apply_context_engine_selection',
                         self._selector),
            patch.object(loop, '_canonicalize_api_tool_calls', lambda messages: None),
            patch.object(loop, '_midturn_request_pressure_tokens',
                         lambda agent, messages, system, approx: approx),
            patch.object(loop, '_pressure_with_real_floor', lambda compressor, value: value),
        ]
        for p in self._patches:
            p.start()
        return self

    def _selector(self, agent, api, history, incoming, logger=None):
        if self.grow:
            return api + [{'role': 'user', 'content': 'injected ' * 60000}]
        return api

    def __exit__(self, *exc):
        from unittest.mock import patch
        for p in reversed(self._patches):
            p.stop()
        return False


def test_late_context_growth_refuses_without_reselection():
    from types import SimpleNamespace
    from gateway.platforms.api_server_precision_context import guard_prepared_request
    from agent.turn_loop_errors import PrecisionContextRefused, handle_outer_loop_error
    agent = SimpleNamespace(max_tokens=8192, _hermes_precision_context_window=65536,
                            request_overrides={'extra_body': {'homelab_queue_timeout_seconds': 0}})
    guard_prepared_request(agent, [{'role': 'system', 'content': 'small'}], [])
    assert agent.request_overrides['extra_body']['homelab_required_context'] > 8192
    with pytest.raises(PrecisionContextRefused) as caught:
        guard_prepared_request(agent, [{'role': 'user', 'content': 'x' * 1000000}], [])
    verdict = handle_outer_loop_error(agent, e=caught.value, _outer_error_count=0,
        api_call_count=0, messages=[], conversation_history=[], _turn_exit_reason='',
        failed=False, final_response='')
    assert verdict.action == 'break' and verdict.failed is True
    assert verdict._turn_exit_reason == 'precision_context_unavailable'
