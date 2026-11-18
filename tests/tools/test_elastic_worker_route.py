"""Offline GXTD-850 plugin-to-child construction; never launches a worker."""
import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.tools.test_delegation_route_fields import _construct, _spawn

_CAPTURED_PLUGIN = Path(__file__).resolve().parents[3] / 'homelab/infra/hermes/specialist-support/advanced-delegation'
PLUGIN = (_CAPTURED_PLUGIN if _CAPTURED_PLUGIN.is_dir() else Path(os.environ.get('GXTD850_PLUGIN_ROOT',
    '/usr/local/share/git/repos/homelab/infra/hermes/specialist-support/advanced-delegation'))).resolve()


@pytest.fixture
def provider(monkeypatch):
    spec = importlib.util.spec_from_file_location('g850_plugin', PLUGIN / '__init__.py', submodule_search_locations=[str(PLUGIN)])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, package)
    spec.loader.exec_module(package)
    spec = importlib.util.spec_from_file_location('g850_plugin.model_map', PLUGIN / 'model_map.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve().parent == PLUGIN
    return module.ModelMapDelegationProvider()


@pytest.fixture
def resolver():
    state = {'schema': 'homelab.elastic-worker.v1', 'selector': 'elastic-worker', 'available': True,
             'epoch_kind': 'admission_route_fingerprint',
             'pin': {'public_id': 'fixture-exact', 'native_model': 'fixture-native',
                     'context_length': 65536, 'route_epoch': 'a' * 64}}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == '/routing/elastic-worker'
            body = json.dumps(state).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def config(url, model='elastic-worker'):
    return {'backend': 'model_map', 'fallback_providers': [{'provider': 'openrouter', 'model': 'must-not-run'}],
            'backends': {'model_map': {'default': 'worker', 'options': [
                {'name': 'worker', 'description': 'Fixture only', 'base_url': url, 'model': model}]}}}


def test_alias_resolves_and_pins_through_child_construction(provider, resolver):
    url, state = resolver
    def construct(**kwargs):
        child = _construct(**kwargs)
        child.context_compressor = SimpleNamespace(context_length=65536)
        return child
    _result, built, children = _spawn([None], cfg=config(url), provider=provider, construct=construct)
    assert len(children) == 1
    child = children[0]
    assert child.model == 'fixture-exact'
    assert built[0]['fallback_model'] is None
    assert built[0]['api_key'] == 'no-key-required'
    assert child.request_overrides['extra_body']['homelab_elastic_worker'] == state['pin']
    manifest = child._delegate_execution_manifest
    assert manifest['requested']['model'] == 'elastic-worker'
    assert manifest['native_model'] == 'fixture-native'
    assert manifest['route_epoch'] == state['pin']['route_epoch']
    assert manifest['epoch_kind'] == 'admission_route_fingerprint'


def test_exact_pin_does_not_consult_alias_resolver(provider):
    route = provider.route_task({}, config('http://127.0.0.1:1/v1', 'explicit-native'))
    assert route['model'] == 'explicit-native'
    assert 'execution_identity' not in route


def test_unavailable_alias_refuses_before_constructing_child(provider, resolver):
    url, state = resolver
    state['available'] = False
    result, built, children = _spawn([None], cfg=config(url), provider=provider)
    assert not built and not children
    assert 'elastic-worker is unavailable' in result['error']


@pytest.mark.parametrize('field,value', [('context_length', True), ('context_length', 8192),
                                        ('route_epoch', 'guessed'), ('native_model', None)])
def test_malformed_identity_never_constructs_child(provider, resolver, field, value):
    url, state = resolver
    state['pin'][field] = value
    result, built, children = _spawn([None], cfg=config(url), provider=provider)
    assert not built and not children
    assert 'elastic-worker is unavailable' in result['error']


def test_identity_is_detached_from_later_resolution(provider, resolver):
    url, state = resolver
    first = provider.route_task({}, config(url))
    state['pin']['public_id'] = 'fixture-successor'
    state['pin']['route_epoch'] = 'b' * 64
    second = provider.route_task({}, config(url))
    assert first['model'] == 'fixture-exact'
    assert second['model'] == 'fixture-successor'
    assert first['request_overrides']['extra_body']['homelab_elastic_worker']['route_epoch'] == 'a' * 64
