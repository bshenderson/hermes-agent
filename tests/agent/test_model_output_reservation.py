"""Route-scoped reservations must follow identity, never leak across swaps."""

import json

import pytest
from openai import OpenAI
from run_agent import AIAgent

QWEN = 'qwen38-unsloth-27b-q6-96k'
LOCAL = 'http://127.0.0.1:8090/v1'


def configure(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = {
        'model': {'default': 'other', 'provider': 'openai', 'context_length': 262144},
        'providers': {'llamacpp-main': {'api': LOCAL, 'context_length': 98304,
                      'models': {QWEN: {'context_length': 98304, 'max_tokens': 16384},
                                 'other-local': {'context_length': 98304}}}},
        'compression': {'threshold': .85}, 'memory': {'memory_enabled': False},
        'auxiliary': {'compression': {'provider': 'auto', 'model': ''}},
    }
    (tmp_path / 'config.yaml').write_text(json.dumps(config))
    return config


def make_agent(model, provider, base_url, max_tokens=None):
    return AIAgent(model=model, provider=provider, requested_provider=provider,
                   base_url=base_url, api_key='test-only-not-a-credential',
                   api_mode='chat_completions', max_tokens=max_tokens,
                   enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
                   skip_memory=True, skip_background_review=True)


def assert_limit(agent, expected):
    assert agent.max_tokens == expected
    assert agent.context_compressor.max_tokens == expected


def test_init_and_switch_follow_exact_route(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    agent = make_agent(QWEN, 'custom', LOCAL)
    assert_limit(agent, 16384)
    assert agent.context_compressor.context_length == 98304
    assert agent.context_compressor.threshold_tokens == 69632
    assert agent._session_init_model_config['max_tokens'] is None
    # Same endpoint, different model must drop the local reservation.
    agent.switch_model('other-local', new_provider='custom', base_url=LOCAL,
                       api_key='test-only-not-a-credential', api_mode='chat_completions')
    assert_limit(agent, None)
    agent.switch_model(QWEN, new_provider='custom', base_url=LOCAL,
                       api_key='test-only-not-a-credential', api_mode='chat_completions')
    assert_limit(agent, 16384)
    # Same model name on a different custom endpoint must also drop it.
    agent.switch_model(QWEN, new_provider='custom', base_url='http://127.0.0.1:9/v1',
                       api_key='test-only-not-a-credential', api_mode='chat_completions')
    assert_limit(agent, None)
    # Codex/native Responses remains provider-default, even after returning from Qwen.
    agent.switch_model(QWEN, new_provider='custom', base_url=LOCAL,
                       api_key='test-only-not-a-credential', api_mode='chat_completions')
    assert_limit(agent, 16384)
    agent.switch_model('gpt-5.4', new_provider='openai-codex', base_url='https://chatgpt.com/backend-api/codex',
                       api_key='test-only-not-a-credential', api_mode='codex_responses')
    assert_limit(agent, None)
    assert agent._primary_runtime['max_tokens'] is None


def test_explicit_precedence_fallback_restore_and_rollback(tmp_path, monkeypatch):
    config = configure(tmp_path, monkeypatch)
    agent = make_agent(QWEN, 'custom', LOCAL, max_tokens=8192)
    assert_limit(agent, 8192)
    agent.switch_model('other-local', new_provider='custom', base_url=LOCAL,
                       api_key='test-only-not-a-credential', api_mode='chat_completions')
    assert_limit(agent, 8192)
    agent = make_agent(QWEN, 'custom', LOCAL)
    # Actual fallback path; only client/auth resolution is isolated from external services.
    agent._fallback_chain = [{'provider': 'openai', 'model': 'gpt-4.1',
                              'base_url': 'http://127.0.0.1:9/v1'}]
    agent._fallback_index = 0
    client = OpenAI(api_key='test-only-not-a-credential', base_url='http://127.0.0.1:9/v1')
    monkeypatch.setattr('agent.auxiliary_client.resolve_provider_client', lambda *a, **kw: (client, 'gpt-4.1'))
    assert agent._try_activate_fallback()
    assert_limit(agent, None)
    assert agent._restore_primary_runtime()
    assert_limit(agent, 16384)
    # Failed compressor transition must restore both output cap and reserved threshold.
    original_update = agent.context_compressor.update_model
    def fail_once(**kwargs):
        if kwargs.get('model') == 'other-local':
            raise ValueError('injected compressor failure')
        return original_update(**kwargs)
    monkeypatch.setattr(agent.context_compressor, 'update_model', fail_once)
    with pytest.raises(ValueError, match='injected'):
        agent.switch_model('other-local', new_provider='custom', base_url=LOCAL,
                           api_key='test-only-not-a-credential', api_mode='chat_completions')
    assert agent.model == QWEN
    assert_limit(agent, 16384)
    assert agent.context_compressor.threshold_tokens == 69632
    # A missing profile setting must not borrow the default profile's reservation.
    config['providers']['llamacpp-main']['models'][QWEN].pop('max_tokens')
    (tmp_path / 'config.yaml').write_text(json.dumps(config))
    other_profile_agent = make_agent(QWEN, 'custom', LOCAL)
    assert_limit(other_profile_agent, None)
