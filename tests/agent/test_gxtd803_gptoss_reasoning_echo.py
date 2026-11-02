"""GPT-OSS Harmony requires current tool-turn reasoning on the owned local route."""
from copy import deepcopy

import pytest

from agent.message_sanitization import (
    apply_reasoning_content_policy,
    needs_reasoning_echo,
    reapply_reasoning_echo,
)

MODEL = 'gptoss-120b-openai-mxfp4-tp1-128k'
BASE = 'http://127.0.0.1:8090/v1'


def test_owned_gptoss_tool_turn_keeps_captured_reasoning_on_wire():
    source = {'role': 'assistant', 'content': '',
              'reasoning_content': 'Use the requested terminal tool.',
              'tool_calls': [{'id': 'call_owned', 'type': 'function',
                              'function': {'name': 'terminal', 'arguments': '{"command":"printf marker"}'}}]}
    wire = deepcopy(source)
    echo = needs_reasoning_echo('custom', MODEL, BASE)
    apply_reasoning_content_policy(source, wire, echo)
    assert wire['reasoning_content'] == source['reasoning_content']
    reapply_reasoning_echo([wire], echo)
    assert wire['reasoning_content'] == source['reasoning_content']
    assert wire['tool_calls'] == source['tool_calls']


@pytest.mark.parametrize('provider,model,base', [
    ('custom', MODEL, BASE),
    ('custom:llamacpp-main', MODEL, BASE + '/'),
])
def test_owned_route_requires_echo(provider, model, base):
    assert needs_reasoning_echo(provider, model, base)


def test_actual_agent_policy_preserves_tool_continuation_and_invalidates_on_switch():
    from agent.reasoning_params import ReasoningParamsMixin
    from agent.turn_context import build_api_messages

    class Agent(ReasoningParamsMixin):
        provider = 'custom'
        model = MODEL
        base_url = BASE
        api_mode = 'chat_completions'
        ephemeral_system_prompt = None
        _current_turn_timestamp = 10000.0

        def _should_sanitize_tool_calls(self):
            return False

    agent = Agent()
    history = [{'role': 'user', 'content': 'Run the marker.'},
               {'role': 'assistant', 'content': '', 'reasoning_content': 'Use the terminal.',
                'tool_calls': [{'id': 'c1', 'type': 'function',
                                'function': {'name': 'terminal', 'arguments': '{}'}}]},
               {'role': 'tool', 'content': 'marker', 'tool_call_id': 'c1'}]
    wire, _ = build_api_messages(agent, history, current_turn_user_idx=0,
        ext_prefetch_cache='', plugin_user_context='', moa_config=None, active_system_prompt='')
    assert wire[1]['reasoning_content'] == 'Use the terminal.'
    assert history[1]['reasoning_content'] == 'Use the terminal.'
    agent.model = 'precision-default'
    wire, _ = build_api_messages(agent, history, current_turn_user_idx=0,
        ext_prefetch_cache='', plugin_user_context='', moa_config=None, active_system_prompt='')
    assert 'reasoning_content' not in wire[1]


@pytest.mark.parametrize('provider,model,base', [
    ('groq', MODEL, 'https://api.groq.com/openai/v1'),
    ('custom', MODEL, 'https://127.0.0.1:8090/v1'),
    ('custom', MODEL, 'http://127.0.0.1:8090/v1/foreign'),
    ('custom', MODEL, 'http://127.0.0.1:8090.evil/v1'),
    ('custom', 'precision-default', BASE),
    ('custom', 'gemma4-31b-nvfp4-64k', BASE),
    ('custom', 'not-' + MODEL, BASE),
    ('groq', MODEL, BASE),
])
def test_other_destinations_keep_strict_strip_policy(provider, model, base):
    source = {'role': 'assistant', 'content': '', 'reasoning_content': 'private reasoning'}
    wire = deepcopy(source)
    echo = needs_reasoning_echo(provider, model, base)
    assert not echo
    apply_reasoning_content_policy(source, wire, echo)
    assert 'reasoning_content' not in wire
