#!/usr/bin/env python3
"""Run the GXTD-647 offline integration matrix with the canonical test runner.

Use HERMES_PYTHON to select a development interpreter without modifying live
packages. Each run gets a disposable HOME, never operator state.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
EXPLICIT = (
    "tests/agent/test_aux_owned_cancellation.py",
    "tests/agent/test_aux_stream_host_deadline.py",
    "tests/agent/test_tool_guardrails.py",
    "tests/agent/test_tool_call_guardrail_runtime.py",
    "tests/agent/test_web_synthesis.py",
    "tests/agent/test_preferred_web_policy.py",
    "tests/agent/test_preferred_web_dispatch.py",
    "tests/gateway/test_preferred_web_http.py",
    "tests/tools/test_web_backend_policy.py",
    "tests/agent/test_chat_location.py",
    "tests/gateway/test_homelab_overlay_integration.py",
    "tests/gateway/test_api_lifecycle_metrics.py",
    "tests/gateway/test_api_lifecycle_native.py",
    "tests/gateway/test_sse_disconnect.py",
    "tests/gateway/test_multiplex_adapter_registry.py",
    "tests/hermes_cli/test_tools_config.py",
    "tests/hermes_cli/test_manual_compress_recovery.py",
    "tests/hermes_cli/test_cli_manual_compress.py",
    "tests/hermes_cli/test_manual_compress.py",
    "tests/test_cli_command_interrupt.py",
    "tests/test_cli_signal_command_interrupt.py",
)
PATTERNS = (
    "tests/agent/*compress*.py",
    "tests/gateway/test_api_server*.py",
    "tests/tools/test_web_tools*.py",
)


def main():
    paths: set[str] = set(EXPLICIT)
    for pattern in PATTERNS:
        paths.update(str(path.relative_to(ROOT)) for path in ROOT.glob(pattern))
    missing = [path for path in paths if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required test files: {missing}")
    print(f"GXTD-647 offline matrix: {len(paths)} files", flush=True)
    with tempfile.TemporaryDirectory(prefix="gxtd647-test-home-") as home:
        env = {**os.environ, "HOME": home}
        return subprocess.call(
            ["bash", "scripts/run_tests.sh", "-j", "2", "--file-retries", "0", *sorted(paths)],
            cwd=ROOT, env=env,
        )


if __name__ == "__main__":
    sys.exit(main())
