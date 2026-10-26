"""Bootstrapped API constructor with only configuration/auth/model I/O replaced.

Run with the existing managed Hermes interpreter, not the obsolete .venv.
No model request, live home or production database is used.
"""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class NativeConstructorTest(unittest.TestCase):
    def test_real_import_and_runtime_selection(self):
        with tempfile.TemporaryDirectory(prefix='g743-native-', dir=os.environ['TMPDIR']) as home:
            with patch.dict(os.environ, {'HERMES_HOME': home, 'HERMES_DISABLE_PM_RELAUNCH': '1'}):
                from gateway.config import PlatformConfig
                from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
                from gateway.platforms import api_server_precision_availability as policy
                import gateway.run as gateway
                from hermes_cli import tools_config
                import run_agent
                adapter = APIServerAdapter(PlatformConfig(enabled=True))
                built = []
                def construct(**kwargs):
                    built.append(kwargs)
                    return SimpleNamespace(model=kwargs['model'], provider=kwargs['provider'])
                token = _api_request_profile.set('homelab-coordinator')
                try:
                    with patch.object(gateway, '_resolve_runtime_agent_kwargs', side_effect=lambda: {
                            'provider': 'custom', 'base_url': 'http://127.0.0.1:8090/v1', 'api_key': 'local-sentinel'}), \
                         patch.object(gateway, '_resolve_gateway_model', return_value='operator-default'), \
                         patch.object(gateway, '_load_gateway_config', return_value={'api_server': {'precision_availability': True}}), \
                         patch.object(gateway.GatewayRunner, '_load_reasoning_config', return_value=None), \
                         patch.object(gateway, '_checkpoint_agent_kwargs', return_value={}), \
                         patch.object(gateway, '_current_max_iterations', return_value=4), \
                         patch.object(tools_config, '_get_platform_tools', return_value=['file']), \
                         patch.object(adapter, '_ensure_session_db', return_value=None), \
                         patch.object(adapter, '_resolve_provider_runtime', return_value={
                             'provider': 'openai-codex', 'api_key': 'cloud-sentinel',
                             'base_url': 'https://cloud.invalid', 'api_mode': 'codex_responses'}), \
                         patch.object(policy, 'probe_local', return_value={'available': False, 'reason': 'backend_fenced'}) as probe, \
                         patch.object(run_agent, 'AIAgent', side_effect=construct):
                        agent = adapter._create_agent(precision_availability_eligible=True)
                        probe.return_value = {'available': True, 'reason': 'ready'}
                        local = adapter._create_agent(precision_availability_eligible=True)
                    self.assertEqual(agent.model, 'gpt-6.1-sol')
                    self.assertEqual(built[0]['api_key'], 'cloud-sentinel')
                    self.assertEqual(built[0]['base_url'], 'https://cloud.invalid')
                    self.assertIsNone(built[0]['fallback_model'])
                    self.assertEqual(local.model, 'operator-default')
                    self.assertEqual(local.request_overrides['extra_body']['homelab_queue_timeout_seconds'], 0)
                    self.assertIsNone(built[1]['fallback_model'])
                finally:
                    _api_request_profile.reset(token)


if __name__ == '__main__':
    unittest.main()
