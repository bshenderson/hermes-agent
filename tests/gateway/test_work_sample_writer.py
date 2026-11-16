"""Status-writer pairing across delay, failure, and incarnation replacement."""
import threading

from gateway import status
from tests.gateway.restart_test_helpers import make_restart_runner


def test_incarnation_heartbeat_cannot_reinherit_old_sample(tmp_path, monkeypatch):
    target = tmp_path / 'status.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    runner, _ = make_restart_runner()
    runner._persist_active_agents()
    assert status.flush_runtime_status()
    identity = status._build_pid_record()
    monkeypatch.setattr(status, '_build_pid_record', lambda: {**identity, 'start_time': identity['start_time'] + 1})
    status.write_runtime_status(platform='fixture', platform_state='connected')
    assert status.flush_runtime_status()
    assert status.read_runtime_status(target)['work_sample'] is None


def test_delayed_writer_coalesces_complete_pair(tmp_path, monkeypatch):
    target = tmp_path / 'status.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    entered, release = threading.Event(), threading.Event()
    original = status._write_json_file
    calls = []
    def write(path, payload):
        calls.append(payload)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        original(path, payload)
    writer = status._RuntimeStatusWriter(write)
    monkeypatch.setattr(status, '_get_runtime_status_writer', lambda: writer)
    runner, _ = make_restart_runner()
    try:
        runner._persist_active_agents()
        assert entered.wait(5)
        monkeypatch.setattr(runner, '_running_agent_count', lambda: 3)
        runner._persist_active_agents()
        status.publish_runtime_status(platform='fixture', platform_state='connected')
    finally:
        release.set()
    assert writer.flush(5)
    result = status.read_runtime_status(target)
    assert result['active_agents'] == result['work_sample']['total'] == 3
    assert all(p['active_agents'] == p['work_sample']['total'] for p in calls)


def test_failed_writer_recovers_on_next_publication(tmp_path, monkeypatch):
    target = tmp_path / 'status.json'
    monkeypatch.setattr(status, '_get_runtime_status_path', lambda: target)
    attempts = []
    def write(path, payload):
        attempts.append(payload)
        if len(attempts) == 1:
            raise OSError('fixture storage failure')
        status._write_json_file(path, payload)
    writer = status._RuntimeStatusWriter(write)
    monkeypatch.setattr(status, '_get_runtime_status_writer', lambda: writer)
    runner, _ = make_restart_runner()
    runner._persist_active_agents()
    assert not writer.flush(5)
    monkeypatch.setattr(runner, '_running_agent_count', lambda: 2)
    runner._persist_active_agents()
    assert writer.flush(5)
    result = status.read_runtime_status(target)
    assert result['active_agents'] == result['work_sample']['total'] == 2
    assert result['work_sample']['sequence'] > attempts[0]['work_sample']['sequence']
