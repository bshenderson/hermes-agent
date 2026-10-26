"""Profile-scoped pre-dispatch selection; no model, credentials or live I/O."""
import importlib.util
from pathlib import Path
import unittest
import ast
import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / 'gateway/platforms/api_server_precision_availability.py'


class PrecisionAvailabilityTests(unittest.TestCase):
    def test_unavailable_selects_only_approved_cloud_before_construction(self):
        self.assertTrue(SOURCE.exists(), 'precision availability selector not implemented')
        spec = importlib.util.spec_from_file_location('precision_availability_test', SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = []
        def probe(model):
            calls.append(('probe', model))
            return {'available': False, 'reason': 'backend_fenced'}
        def resolve(provider, model):
            calls.append(('resolve', provider, model))
            return {'provider': provider, 'model': model, 'base_url': 'https://cloud.invalid'}
        selected = module.select_runtime(
            enabled=True, profile='homelab-coordinator', model='operator-default',
            explicit_selection=False, probe=probe, resolve=resolve)
        self.assertEqual(selected['model'], 'gpt-6.1-sol')
        self.assertEqual(selected['runtime']['provider'], 'openai-codex')
        self.assertEqual(calls, [('probe', 'operator-default'), ('resolve', 'openai-codex', 'gpt-6.1-sol')])
        self.assertIn('backend_fenced', selected['notice'])

    def test_actual_agent_constructor_selects_cloud_without_local_credentials(self):
        spec = importlib.util.spec_from_file_location('gateway.platforms.api_server_precision_availability', SOURCE)
        assert spec is not None and spec.loader is not None
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper.probe_local = lambda *_: {'available': False, 'reason': 'capacity_busy'}
        tree = ast.parse((SOURCE.parent / 'api_server.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(isinstance(x, ast.FunctionDef) and x.name == '_create_agent' for x in n.body))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_create_agent')
        constructed = []
        def constructor(**kw):
            constructed.append(kw)
            return SimpleNamespace(model=kw['model'], provider=kw['provider'])
        gateway = ModuleType('gateway.run')
        gateway._resolve_runtime_agent_kwargs = lambda: {'provider': 'custom', 'api_key': 'local-secret-sentinel', 'base_url': 'http://127.0.0.1:8090/v1'}
        gateway._resolve_gateway_model = lambda: 'operator-default'
        gateway._load_gateway_config = lambda: {'api_server': {'precision_availability': True}}
        gateway._checkpoint_agent_kwargs = lambda _: {}
        gateway._current_max_iterations = lambda: 8
        gateway.GatewayRunner = SimpleNamespace(_load_fallback_model=lambda: None, _load_reasoning_config=lambda _: None)
        env = {'Any': object, 'Optional': __import__('typing').Optional,
               '_request_reasoning_config': lambda _: None,
               '_request_service_tier': lambda _: None, '_REQUEST_OPTION_MISSING': None,
               'resolve_display_setting': lambda *a: True,
               '_api_request_profile': SimpleNamespace(get=lambda: 'homelab-coordinator'),
               '_RUNTIME_AGENT_OVERRIDE_KEYS': ('api_key', 'base_url', 'provider', 'api_mode', 'command', 'args', 'credential_pool'),
               '_ProviderAuthResolutionError': RuntimeError,
               '_apply_runtime_agent_overrides': lambda a,b: a.update(b)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE.parent / 'api_server.py'), 'exec'), env)
        owner = SimpleNamespace(_select_agent_runtime=lambda kw,m,**k: (m,None,None,None),
            _resolve_provider_runtime=lambda *a,**k: {'provider': 'openai-codex', 'api_key': 'cloud-secret-sentinel', 'base_url': 'https://cloud.invalid'},
            _ensure_session_db=lambda: None, _memory_sessions=SimpleNamespace(checkout=lambda _: None))
        imports = {'gateway.run': gateway, 'run_agent': SimpleNamespace(AIAgent=constructor),
            'hermes_cli.tools_config': SimpleNamespace(_get_platform_tools=lambda *a: ['file']),
            'gateway.platforms.api_server_precision_availability': helper}
        with patch.dict(sys.modules, imports):
            retained = env['_create_agent'](owner)
            self.assertEqual(retained.model, 'operator-default')
            agent = env['_create_agent'](owner, precision_availability_eligible=True)
        self.assertEqual(agent.model, 'gpt-6.1-sol')
        self.assertEqual(constructed[1]['api_key'], 'cloud-secret-sentinel')
        self.assertEqual(constructed[1]['base_url'], 'https://cloud.invalid')
        self.assertIsNone(constructed[1]['fallback_model'])
        self.assertNotIn('runtime', agent._hermes_precision_availability)

    def test_scope_and_explicit_choices_do_not_probe_or_resolve(self):
        spec = importlib.util.spec_from_file_location('precision_policy_scope', SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def forbidden(*_):
            self.fail('unapproved readiness or credential access')
        cases = [(False, 'homelab-coordinator', 'operator-default', False),
                 ('true', 'homelab-coordinator', 'operator-default', False),
                 (True, 'homelab-coordinator', 'operator-default', True),
                 (True, 'homelab-coordinator', 'chat-default', False)]
        cases += [(True, p, 'operator-default', False) for p in
                  ['personal-barry', 'govcon', 'research', 'media', 'default', 'operator', None]]
        for enabled, profile, model, explicit in cases:
            with self.subTest(profile=profile, model=model, explicit=explicit):
                self.assertIsNone(module.select_runtime(enabled=enabled, profile=profile,
                    model=model, explicit_selection=explicit, probe=forbidden, resolve=forbidden))

    def test_ready_local_never_resolves_cloud_and_denial_is_sanitized(self):
        spec = importlib.util.spec_from_file_location('precision_policy_local', SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for profile in module.PROFILES:
            selected = module.select_runtime(enabled=True, profile=profile, model='operator-default',
                explicit_selection=False, probe=lambda _: {'available': True},
                resolve=lambda *_: self.fail('cloud resolution on local ready'))
            self.assertTrue(selected['local_no_wait'])
            self.assertIsNone(selected['runtime'])
        def denied(*_):
            raise ValueError('synthetic-private-token')
        with self.assertRaises(module.AvailabilityRefused) as error:
            module.select_runtime(enabled=True, profile='homelab-coordinator', model='operator-default',
                explicit_selection=False, probe=lambda _: {'available': False}, resolve=denied)
        self.assertNotIn('synthetic-private-token', str(error.exception))
        self.assertIn('No model or tools dispatched', str(error.exception))


if __name__ == '__main__':
    unittest.main()
