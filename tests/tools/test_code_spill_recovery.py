"""GXTD-850: recover nested reads without replay, with honest spill limits."""

import json
from pathlib import Path

import pytest

from tools import code_execution_tool as execution
from tools import file_tools
from tools.code_kernel import shutdown_all_kernels
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture
def real_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id="default": ops)
    # Replace only dispatcher selection; the nested read and filesystem are real.
    def dispatch(name, arguments, task_id="default", **kwargs):
        assert name == "read_file"
        return file_tools.read_file_tool(task_id=task_id, **arguments)
    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


def test_nested_read_spill_recovers_hidden_content_without_rerun(tmp_path, real_reads):
    source = tmp_path / "source.txt"
    source.write_text("".join(f"row-{i:04d}: " + "x" * 70 + "\n" for i in range(900)))
    second = tmp_path / "second.txt"
    second.write_text("".join(f"second-{i:04d}: " + "y" * 70 + "\n" for i in range(900)))
    task = "spill-recovery-child-a"
    result = json.loads(execution.execute_code(
        f"from hermes_tools import read_file\n"
        f"print(read_file({str(source)!r})['content'])\n"
        f"print(read_file({str(second)!r})['content'])",
        task_id=task, enabled_tools=["read_file"],
    ))
    assert result["status"] == "success"
    assert result["stdout_truncated"]
    assert result["stdout_spill_complete"] is True
    assert result["stdout_spill_bytes_omitted"] == 0
    assert "row-0400:" not in result["output"]
    repeated = json.loads(file_tools.read_file_tool(str(source), task_id=task))
    assert repeated["content_returned"] is False
    recovered = json.loads(file_tools.read_file_tool(result["stdout_spill_path"], offset=390, limit=30, task_id=task))
    assert "row-0400:" in recovered["content"]
    recovered_second = json.loads(file_tools.read_file_tool(
        result["stdout_spill_path"], offset=1290, limit=30, task_id=task))
    assert "second-0400:" in recovered_second["content"]
    other = json.loads(file_tools.read_file_tool(str(source), task_id="spill-recovery-child-b"))
    assert "row-0400:" in other["content"]


def test_capped_spill_obeys_byte_limit_and_reports_loss(real_reads, monkeypatch):
    monkeypatch.setattr(execution, "MAX_STDOUT_BYTES", 100)
    monkeypatch.setattr(execution, "MAX_SPILLED_STDOUT_BYTES", 301)
    result = json.loads(execution.execute_code(
        "print('界' * 200)", task_id="spill-capped-child", enabled_tools=["read_file"],
    ))
    assert result["status"] == "success"
    retained = Path(result["stdout_spill_path"]).read_bytes()
    assert len(retained) <= 301
    assert retained.decode("utf-8") == "界" * 100
    assert result["stdout_spill_complete"] is False
    assert result["stdout_spill_bytes"] == 300
    assert result["stdout_spill_bytes_omitted"] == 301
    assert "FULL output" not in result["warning"]


@pytest.mark.parametrize("size", [300, 301, 302])
def test_spill_exact_byte_boundary(tmp_path, monkeypatch, size):
    monkeypatch.setattr(execution, "MAX_STDOUT_BYTES", 100)
    monkeypatch.setattr(execution, "MAX_SPILLED_STDOUT_BYTES", 301)
    _, metadata = execution._truncate_stdout_text("x" * size)
    retained = Path(metadata["stdout_spill_path"]).read_bytes()
    assert retained == b"x" * min(size, 301)
    assert metadata["stdout_spill_complete"] is (size <= 301)
    assert metadata["stdout_spill_bytes_omitted"] == max(0, size - 301)


def test_failed_spill_does_not_advise_replaying_effects(real_reads, monkeypatch):
    monkeypatch.setattr(execution, "MAX_STDOUT_BYTES", 100)
    def fail_write(*args, **kwargs):
        raise OSError("fixture storage unavailable")
    monkeypatch.setattr("tools.spill_safety.write_text_exclusive", fail_write)
    result = json.loads(execution.execute_code(
        "print('x' * 200)", task_id="spill-failed-child", enabled_tools=["read_file"],
    ))
    assert result["status"] == "success"
    assert result["stdout_truncated"]
    assert "stdout_spill_path" not in result
    assert "Spill storage failed" in result["warning"]
    assert "do not replay side effects" in result["warning"]
