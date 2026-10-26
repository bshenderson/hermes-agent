"""Agent-backed auxiliary slots retain their declared wire request overrides."""
import copy
from types import SimpleNamespace

from agent import background_review, review_engine


def test_review_delegation_credentials_carry_auxiliary_extra_body(monkeypatch):
    slot = {'provider': 'named-child', 'model': 'child-model', 'extra_body': {'max_tokens': 1024, 'chat_template_kwargs': {'enable_thinking': False}}}
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: {'auxiliary': {'review': slot}})
    cfg = review_engine._load_review_credentials_cfg()
    assert cfg is not None
    assert cfg['request_overrides']['extra_body'] == slot['extra_body']
    cfg['request_overrides']['extra_body']['max_tokens'] = 2048
    assert slot['extra_body']['max_tokens'] == 1024


def test_review_inherited_route_carries_explicit_ceiling(monkeypatch):
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: {
        'auxiliary': {'review': {'provider': 'auto', 'extra_body': {'max_tokens': 1024}}}})
    cfg = review_engine._load_review_credentials_cfg()
    assert cfg is not None
    assert cfg['provider'] == ''
    assert cfg['request_overrides']['extra_body']['max_tokens'] == 1024


def _parent():
    return SimpleNamespace(provider='parent', model='parent-model', max_tokens=None,
        request_overrides={'extra_body': {'parent_flag': True}},
        _current_main_runtime=lambda: {'api_key': 'fixture', 'base_url': 'http://parent.invalid/v1'})


def test_background_review_pinned_route_merges_slot_ceiling(monkeypatch):
    runtime = {'provider': 'child', 'model': 'child-model', 'request_overrides': {'extra_body': {'runtime_flag': True}}}
    before = copy.deepcopy(runtime)
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_provider', lambda **kwargs: runtime)
    slot = {'provider': 'child', 'model': 'child-model', 'extra_body': {'max_tokens': 1024, 'chat_template_kwargs': {'enable_thinking': False}}}
    result = background_review._resolve_review_runtime(_parent(), slot)
    assert result['request_overrides']['extra_body'] == {'runtime_flag': True, **slot['extra_body']}
    result['request_overrides']['extra_body']['max_tokens'] = 2048
    assert runtime == before
    assert slot['extra_body']['max_tokens'] == 1024


def test_background_review_inherited_route_applies_slot_ceiling():
    parent = _parent()
    result = background_review._resolve_review_runtime(parent, {'provider': 'auto', 'extra_body': {'max_tokens': 1024}})
    assert result['request_overrides']['extra_body'] == {'parent_flag': True, 'max_tokens': 1024}
    assert parent.request_overrides == {'extra_body': {'parent_flag': True}}
