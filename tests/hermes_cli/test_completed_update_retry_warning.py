"""GXTD-900: old retry history is not an outstanding update obligation."""
import pytest


@pytest.mark.parametrize('current,pending,warn', [(True, False, False), (True, True, True), (False, False, True)])
def test_completion_warning_requires_owed_work(tmp_path, monkeypatch, capsys, current, pending, warn):
    from hermes_cli import venv_sync
    marker = tmp_path / 'source-completion-pending'
    attempts = marker.with_name('source-completion-attempts')
    attempts.write_text('6\n')
    if pending:
        marker.write_text('source update tail not finished\n')
    monkeypatch.setattr(venv_sync, 'completion_pending_path', lambda root: marker)
    venv_sync._may_retry_completion(tmp_path, ['chat'], current=current)
    assert bool(capsys.readouterr().err) is warn
    assert attempts.read_text() == '6\n'
    assert marker.exists() is pending
