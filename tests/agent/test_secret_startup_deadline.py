"""Single budget traverses the real registry thread and op group loop."""
from pathlib import Path
import time
import importlib.util


def test_one_deadline_stops_later_secret_groups(tmp_path, monkeypatch):
    assert importlib.util.find_spec('agent.secret_sources.deadline') is not None, 'shared source deadline missing'
    from agent.secret_sources.deadline import secret_budget
    from agent.secret_sources.onepassword import fetch_onepassword_secrets
    fake = tmp_path / 'op'
    marker = tmp_path / 'calls'
    fake.write_text('#!/usr/bin/python3\nimport time\nfrom pathlib import Path\n'
                    f'with Path({str(marker)!r}).open("a") as f: f.write("call\\n")\n'
                    'time.sleep(2)\nprint("synthetic")\n')
    fake.chmod(0o700)
    started = time.monotonic()
    with secret_budget(0.15):
        values, warnings = fetch_onepassword_secrets(
            references={'ONE': 'op://test/one/key', 'TWO': 'op://test/two/key'},
            binary=fake, use_cache=False, home_path=tmp_path)
    assert values == {}
    assert warnings
    assert time.monotonic() - started < 1
    assert marker.read_text().splitlines() == ['call']


def test_registry_timeout_does_not_pin_process_exit():
    import subprocess
    import sys
    code = '''
from pathlib import Path
import time
from agent.secret_sources.base import SecretSource, FetchResult
from agent.secret_sources.registry import _fetch_with_timeout
from agent.secret_sources.deadline import secret_budget
class Slow(SecretSource):
    name = 'synthetic'
    def fetch(self, cfg, home):
        time.sleep(5)
        return FetchResult()
with secret_budget(0.05):
    result = _fetch_with_timeout(Slow(), {}, Path('.'), {})
assert result.error
print('bounded')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=2, check=False)
    assert result.returncode == 0 and result.stdout.strip() == 'bounded'


def test_budget_timeout_reaps_ordinary_cli_process_group(tmp_path):
    import os
    import sys
    from agent.secret_sources.base import run_cli
    from agent.secret_sources.deadline import secret_budget
    import pytest
    pid_file = tmp_path / 'child.pid'
    code = ('import subprocess,sys,time; from pathlib import Path; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(15)"]); '
            f'Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(15)')
    started = time.monotonic()
    try:
        with secret_budget(0.2), pytest.raises(RuntimeError):
            run_cli([sys.executable, '-c', code], env=dict(os.environ), timeout=20,
                    label='fixture', timeout_message='fixture deadline')
        assert time.monotonic() - started < 2
        pid = int(pid_file.read_text())
        for _ in range(50):
            stat = Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                break
            time.sleep(0.01)
        else:
            pytest.fail('owned helper descendant survived the deadline')
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), 9)
            except ProcessLookupError:
                pass
