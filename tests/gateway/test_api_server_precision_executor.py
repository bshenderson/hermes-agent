"""Exercise real executor -> constructor; no model, tools or live state."""
import asyncio
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class StopBeforeConversation(Exception):
    pass


class PrecisionExecutorTests(unittest.TestCase):
    def test_executor_retains_server_profile_for_selection(self):
        with tempfile.TemporaryDirectory(prefix='g743-executor-', dir=os.environ['TMPDIR']) as home:
            with patch.dict(os.environ, {'HERMES_HOME': home, 'HERMES_DISABLE_LAZY_INSTALLS': '1'}):
                from gateway.config import PlatformConfig
                from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
                from gateway.platforms import api_server_precision_availability as policy
                import gateway.run as gateway
                from hermes_cli import tools_config
                import run_agent
                adapter = APIServerAdapter(PlatformConfig(enabled=True))
                built = []
                selections = []
                def construct(**kwargs):
                    built.append(kwargs)
                    return SimpleNamespace(model=kwargs['model'], provider=kwargs['provider'])
                real_create = adapter._create_agent
                def stop_after_create(**kwargs):
                    agent = real_create(**kwargs)
                    selections.append(agent._hermes_precision_availability)
                    raise StopBeforeConversation()
                async def exercise(profile='homelab-coordinator', history=None, **kwargs):
                    token = _api_request_profile.set(profile)
                    try:
                        with self.assertRaises(StopBeforeConversation):
                            await adapter._run_agent('fixture input', history or [], session_id='fixture-session', **kwargs)
                    finally:
                        _api_request_profile.reset(token)
                with patch.object(gateway, '_resolve_runtime_agent_kwargs', side_effect=lambda: {
                        'provider': 'custom', 'base_url': 'http://127.0.0.1:8090/v1', 'api_key': 'local-sentinel'}), \
                     patch.object(gateway, '_resolve_gateway_model', return_value='operator-default'), \
                     patch.object(gateway, '_load_gateway_config', return_value={'api_server': {'precision_availability': True}}), \
                     patch.object(gateway.GatewayRunner, '_load_reasoning_config', return_value=None), \
                     patch.object(gateway.GatewayRunner, '_load_fallback_model', return_value=None), \
                     patch.object(gateway, '_checkpoint_agent_kwargs', return_value={}), \
                     patch.object(gateway, '_current_max_iterations', return_value=4), \
                     patch.object(tools_config, '_get_platform_tools', return_value=['file']), \
                     patch.object(adapter, '_profile_scope', side_effect=lambda _: nullcontext()), \
                     patch.object(adapter, '_ensure_session_db', return_value=None), \
                     patch.object(adapter._memory_sessions, 'checkout', return_value=None), \
                     patch.object(adapter, '_resolve_provider_runtime', return_value={
                         'provider': 'openai-codex', 'api_key': 'cloud-sentinel',
                         'base_url': 'https://cloud.invalid', 'api_mode': 'codex_responses'}) as resolve, \
                     patch.object(adapter, '_create_agent', side_effect=stop_after_create), \
                     patch.object(policy, 'probe_local', return_value={'available': True, 'reason': 'ready'}) as probe, \
                     patch.object(run_agent, 'AIAgent', side_effect=construct):
                    asyncio.run(exercise())
                    probe.assert_called_once()
                    self.assertIsNotNone(selections[0], 'executor lost server-owned profile before availability selection')
                    self.assertTrue(selections[0]['local_no_wait'])
                    resolve.assert_not_called()
                    probe.return_value = {'available': False, 'reason': 'capacity_busy'}
                    asyncio.run(exercise())
                    self.assertEqual(built[-1]['model'], 'gpt-6.1-sol')
                    self.assertEqual(built[-1]['api_key'], 'cloud-sentinel')
                    self.assertEqual(built[-1]['base_url'], 'https://cloud.invalid')
                    self.assertIsNone(built[-1]['fallback_model'])
                    self.assertIn('openai-codex/gpt-6.1-sol', selections[-1]['notice'])
                    resolve.assert_called_once_with('openai-codex', target_model='gpt-6.1-sol', required=True)
                    for kwargs in [
                            {'history': [{'role': 'user', 'content': 'prior private turn'}]},
                            {'resume_unanswered_turn': True},
                            {'requested_model': 'operator-default'},
                            {'profile': 'personal-barry'},
                            {'profile': 'govcon'}, {'profile': 'research'}, {'profile': 'media'}]:
                        probe.reset_mock()
                        resolve.reset_mock()
                        asyncio.run(exercise(**kwargs))
                        probe.assert_not_called()
                        resolve.assert_not_called()
                        self.assertIsNone(selections[-1])
                        self.assertEqual(built[-1]['model'], 'operator-default')


if __name__ == '__main__':
    unittest.main()
