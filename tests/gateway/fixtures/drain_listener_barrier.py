"""Pause ordinary startup immediately after platform listeners connect."""
import asyncio
from pathlib import Path
import runpy
from gateway.run_startup import GatewayStartupMixin
from hermes_constants import get_hermes_home

original = GatewayStartupMixin._start_aggregate_connect_results


async def pause_after_connect(self, *args):
    result = await original(self, *args)
    (get_hermes_home() / 'listeners-connected').write_text('paused')
    await asyncio.Event().wait()
    return result


GatewayStartupMixin._start_aggregate_connect_results = pause_after_connect
runpy.run_module('gateway.run', run_name='__main__')
