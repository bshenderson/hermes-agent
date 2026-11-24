"""Real child expiration and explicit readiness of the CLI startup budget."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def test_startup_deadline_exits_actionably_without_secret_output():
    assert importlib.util.find_spec('hermes_cli.startup_budget') is not None, 'CLI startup deadline missing'
    code = '''
from hermes_cli import startup_budget as b
import time
b.arm(0.05)
b.stage('config')
time.sleep(2)
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=3, check=False)
    assert result.returncode == 78
    assert 'stage=config' in result.stderr
    assert 'deadline' in result.stderr
    assert 'Traceback' not in result.stderr


def test_ready_disarms_budget():
    assert importlib.util.find_spec('hermes_cli.startup_budget') is not None
    code = '''
from hermes_cli import startup_budget as b
import time
b.arm(0.05)
b.ready()
time.sleep(0.1)
print('ready')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=3, check=False)
    assert result.returncode == 0
    assert result.stdout.strip() == 'ready'
