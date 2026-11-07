"""Exercise real fresh-process cron dispatch with synthetic secret input only."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('scoped', [True, False])
def test_fresh_external_worker_no_agent_scope(tmp_path, scoped):
    root = Path(__file__).resolve().parents[2]
    home = tmp_path / 'home'
    home.mkdir()
    scripts = home / 'scripts'
    scripts.mkdir()
    sentinel = tmp_path / 'script-ran'
    (scripts / 'probe.sh').write_text(f'#!/bin/sh\ntouch {sentinel}\n')
    calls = tmp_path / 'vault-calls'
    fake = tmp_path / 'fake-op'
    fake.write_text(f'#!/bin/sh\nprintf "call\\n" >> {calls}\nprintf synthetic-value\n')
    fake.chmod(0o700)
    (home / 'config.yaml').write_text(
        'secrets:\n  onepassword:\n    enabled: true\n'
        f'    binary_path: {fake}\n'
        '    env:\n      UNUSED_KEY: op://synthetic/item/key\n'
    )
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('OP_', 'HOMELAB_OP_', 'HERMES_', '_HERMES_'))}
    env.update(HOME=str(home), HERMES_HOME=str(home), HERMES_DISABLE_LAZY_INSTALLS='1',
               PYTHONPATH=str(root), TMPDIR=str(tmp_path))
    setup = '''
import json, sys
from pathlib import Path
from cron.jobs import create_job
from cron.executions import create_execution, mark_execution_handoff_pending
home=Path(sys.argv[1])
kw={'secret_env': []} if sys.argv[2]=='1' else {}
j=create_job(name='synthetic-scope', prompt='', schedule='every 1h',
             script='probe.sh', no_agent=True, deliver='local', **kw)
e=create_execution(j['id'], source='synthetic')
assert mark_execution_handoff_pending(e['id'])
j['execution_id']=e['id']
(home/'payload.json').write_text(json.dumps({'job':j,'profile_home':str(home),
                                           'multiplex_active':False}))
'''
    prep = subprocess.run([sys.executable, '-c', setup, str(home), str(int(scoped))],
                          cwd=root, env=env, text=True, capture_output=True, timeout=30)
    assert prep.returncode == 0, prep.stderr
    # Ignore the setup entrypoint: the real worker starts cold below.
    calls.unlink(missing_ok=True)
    worker = subprocess.run(
        [sys.executable, '-m', 'cron.scheduler', '--external-worker-file',
         str(home / 'payload.json'), '--ack-file', str(home / 'ready.json')],
        cwd=root, env=env, text=True, capture_output=True, timeout=45)
    assert worker.returncode == 0, worker.stderr
    assert sentinel.exists()
    assert json.loads((home / 'ready.json').read_text())['execution_id']
    count = len(calls.read_text().splitlines()) if calls.exists() else 0
    if scoped:
        assert count == 0
    else:
        assert count > 0  # Positive control: the fake manager is reachable.
