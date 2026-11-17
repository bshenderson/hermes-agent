"""Signed webhook input while startup is held before background watchers."""
import hashlib
import hmac
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import pytest
from gateway.drain_control import write_drain_request
from tests.gateway.fixtures.local_recovery_probe import child_env, seed_offline_models_dev


@pytest.mark.parametrize('draining', [True, False])
def test_connected_listener_before_startup_completion(tmp_path, draining):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(); user.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    cfg = {'gateway': {'multiplex_profiles': False},
           'model': {'provider': 'custom', 'default': 'offline', 'base_url': 'http://127.0.0.1:1/v1'},
           'auxiliary': {'title_generation': {'enabled': False}},
           'platforms': {'webhook': {'enabled': True, 'extra': {'host': '127.0.0.1', 'port': port,
               'secret': 'fixture-secret', 'routes': {'fixture': {'prompt': '{text}', 'deliver': 'log'}}}}}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    seed_offline_models_dev(home)
    if draining:
        write_drain_request(home=home, principal='early-listener-fixture')
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL='http://127.0.0.1:1/v1')
    with (home / 'boot.log').open('w+') as log:
        proc = subprocess.Popen([sys.executable, str(root / 'tests/gateway/fixtures/drain_listener_barrier.py')],
                                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            while not (home / 'listeners-connected').exists():
                assert proc.poll() is None and time.monotonic() < deadline, (home / 'boot.log').read_text()
                time.sleep(.05)
            body = json.dumps({'text': 'EARLY_INPUT'}).encode()
            headers = {'X-GitHub-Delivery': 'early', 'X-GitHub-Event': 'push',
                       'X-Hub-Signature-256': 'sha256=' + hmac.new(b'fixture-secret', body, hashlib.sha256).hexdigest()}
            request = urllib.request.Request(f'http://127.0.0.1:{port}/webhooks/fixture', body, headers)
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    status = response.status
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
            assert status == (503 if draining else 202)
            with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                rows = dict(db.execute('SELECT request_id,status FROM session_admissions'))
            assert rows == ({} if draining else {'early': 'queued'})
        finally:
            proc.kill()
            proc.wait(timeout=10)
