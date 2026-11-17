"""Ticket-authenticated disposable gateway: retained drain and in-process release."""
import asyncio
import json
from pathlib import Path
import sqlite3
import threading
from http.server import ThreadingHTTPServer

from gateway.drain_control import write_drain_request, clear_drain_request
from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, websocket, rpc
from tests.waiting import async_wait_until


def test_retained_drain_refuses_socket_input_then_releases_same_process(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {'gateway': {'multiplex_profiles': False},
              'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
              'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1')

    def rows():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return dict(db.execute('SELECT request_id,status FROM session_admissions'))

    async def create(desc):
        async with websocket(home, desc) as ws:
            result = await rpc(ws, 'session.create', request_id='owned', source='tui',
                               cwd=str(home), model='local-wire-stub', toolsets=[])
            assert 'result' in result, result
            return result['result']['session_id']

    async def exercise(desc, session):
        async with websocket(home, desc) as ws:
            resumed = await rpc(ws, 'session.resume', session_id=session)
            assert 'result' in resumed, resumed
            params = dict(session_id=session, input_id='release-input', text='DRAIN_RELEASE')
            refused = await rpc(ws, 'prompt.submit', **params)
            assert refused.get('error', {}).get('message') == 'runtime_draining', refused
            assert rows() == {} and not peer.requests
            clear_drain_request(home=home)
            await async_wait_until(lambda: json.loads((home / 'gateway_state.json').read_text())['gateway_state'] == 'running',
                                   timeout=15, interval=.05, what='same-process drain release')
            accepted = await rpc(ws, 'prompt.submit', **params)
            assert 'result' in accepted, accepted
            await async_wait_until(lambda: rows().get('release-input') == 'terminal', timeout=30,
                                   interval=.05, what='released input terminal')
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            session = asyncio.run(create(desc))
        write_drain_request(home=home, principal='socket-fixture')
        with daemon(root, home, env, barrier=False) as (proc, desc):
            pid = proc.pid
            asyncio.run(exercise(desc, session))
            assert proc.poll() is None and proc.pid == pid
            assert len(peer.requests) == 1
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
