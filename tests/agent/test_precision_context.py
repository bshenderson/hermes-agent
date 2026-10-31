"""Exact deployed public Gemma metadata reaches the actual compressor, offline."""
import pytest
from types import SimpleNamespace
from agent import model_metadata as metadata
from agent.context_compressor import ContextCompressor
from hermes_cli.config import get_custom_provider_context_length
from agent.agent_init import config_context_length_for_runtime

URL = 'http://127.0.0.1:8090/v1'
GEMMA = 'gemma4-31b-nvfp4-64k'


def routes(models=None):
    return [{'name': 'llamacpp-main', 'base_url': URL, 'context_length': 262144,
             'models': models or {}}]


def test_exact_metadata_precedes_stale_cache_and_provider_default(monkeypatch):
    monkeypatch.setattr(metadata, 'get_cached_context_length', lambda *_: pytest.fail('stale cache must not win'))
    cp = routes()
    assert get_custom_provider_context_length(GEMMA, URL, cp) == 65536
    assert metadata.get_model_context_length(GEMMA, URL, provider='custom', custom_providers=cp) == 65536
    compressor = ContextCompressor(model=GEMMA, base_url=URL, provider='custom',
        custom_providers=cp, max_tokens=8192, quiet_mode=True)
    assert compressor.context_length == 65536
    assert compressor.max_tokens == 8192
    assert 0 < compressor.threshold_tokens < compressor.context_length


def test_explicit_model_and_runtime_pins_keep_precedence():
    cp = routes({GEMMA: {'context_length': 49152}})
    assert get_custom_provider_context_length(GEMMA, URL, cp) == 49152
    assert metadata.get_model_context_length(GEMMA, URL, custom_providers=cp) == 49152
    assert metadata.get_model_context_length(GEMMA, URL, config_context_length=32768, custom_providers=[]) == 32768
    assert get_custom_provider_context_length('precision-default', URL, routes()) == 262144


def test_startup_pin_scoping_then_compressor_uses_exact_gemma():
    cp = routes()
    cfg = {'model': {'default': 'operator-default', 'provider': 'custom:llamacpp-main',
                     'base_url': URL, 'context_length': 262144}, 'custom_providers': cp}
    agent = SimpleNamespace(model=GEMMA, provider='custom', base_url=URL)
    scoped = config_context_length_for_runtime(agent, config=cfg)
    assert scoped is None
    compressor = ContextCompressor(model=GEMMA, base_url=URL, provider='custom',
        custom_providers=cp, config_context_length=scoped, quiet_mode=True)
    assert compressor.context_length == 65536


@pytest.mark.parametrize('model,url', [('gemma4-other', URL), (GEMMA, 'http://127.0.0.1:8086/v1'),
    (GEMMA, 'https://other.example/v1'), (GEMMA, URL + '?other=1')])
def test_binding_never_changes_other_models_or_native_endpoints(model, url):
    assert metadata._endpoint_scoped_context_length(model, url) is None
