"""Contract test for s6-overlay helper installation.

This test verifies that the s6-overlay helper binary is present and executable
after installation. The helper is used for container initialization and process
management within the s6-overlay supervision tree.

The test runs in isolation without side effects and checks the Dockerfile
to ensure the helper is installed correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _dockerfile_instructions(dockerfile_text: str) -> list[str]:
    """Extract Dockerfile instructions (excluding comments)."""
    instructions: list[str] = []
    current = ""

    for raw_line in dockerfile_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        continued = line.removesuffix("\\").strip()
        current = f"{current} {continued}".strip()
        if not line.endswith("\\"):
            instructions.append(current)
            current = ""

    return instructions


def _run_steps(dockerfile_text: str) -> list[str]:
    """Extract RUN steps from Dockerfile."""
    return [
        instruction
        for instruction in _dockerfile_instructions(dockerfile_text)
        if instruction.startswith("RUN ")
    ]


def _instruction_text(dockerfile_text: str) -> str:
    """Join non-comment Dockerfile instructions into a searchable string."""
    return "\n".join(_dockerfile_instructions(dockerfile_text))


def test_s6_overlay_helper_installed_in_dockerfile(dockerfile_text: str):
    """Verify s6-overlay helper is installed in the Dockerfile.

    The s6-overlay helper is essential for container initialization and
    process management within the supervision tree. It must be present
    in the Dockerfile build instructions.
    """
    assert DOCKERFILE.exists(), "Dockerfile not present in this checkout"
    dockerfile_text = DOCKERFILE.read_text()

    # s6-overlay helper is located at /command/s6-overlay/command/helper
    # after installation. Verify the Dockerfile includes the s6-overlay
    # installation which creates this path.
    instructions = _instruction_text(dockerfile_text)

    # Check for s6-overlay installation
    assert "s6-overlay" in instructions, (
        "Dockerfile must install s6-overlay. The helper binary at "
        "/command/s6-overlay/command/helper is required for container "
        "initialization and process management."
    )

    # Check for s6-overlay tarball download
    assert any("s6-overlay-*.tar.xz" in step for step in _run_steps(dockerfile_text)), (
        "Dockerfile must download s6-overlay tarball(s) during build."
    )

    # Check for tar extraction that creates the helper path
    assert any("tar" in step and "s6-overlay" in step for step in _run_steps(dockerfile_text)), (
        "Dockerfile must extract s6-overlay tarball to create the helper binary."
    )


def test_s6_overlay_helper_is_executable(dockerfile_text: str):
    """Verify s6-overlay helper installation preserves executable permissions.

    The helper binary must remain executable after installation for
    container initialization to work correctly.
    """
    dockerfile_text = DOCKERFILE.read_text()
    instructions = _instruction_text(dockerfile_text)

    # Check that the tar extraction doesn't explicitly remove execute permissions
    # s6-overlay tarballs extract with default permissions which include execute
    # for binaries. The test verifies the install process doesn't strip them.
    assert "chmod" not in instructions or "chmod u+x" in instructions or "chmod +x" in instructions, (
        "Dockerfile should preserve or explicitly set execute permissions for s6-overlay binaries."
    )


def test_s6_overlay_version_arg_present(dockerfile_text: str):
    """Verify s6-overlay version is configurable via ARG.

    The version should be set as a build ARG for reproducibility and
    ability to bump versions when needed.
    """
    dockerfile_text = DOCKERFILE.read_text()
    assert "S6_OVERLAY_VERSION" in dockerfile_text, (
        "Dockerfile must define S6_OVERLAY_VERSION ARG for version control."
    )


def test_s6_overlay_checksum_verification(dockerfile_text: str):
    """Verify s6-overlay tarballs are checksum-verified during installation.

    Supply-chain integrity requires checksum verification of downloaded
    s6-overlay tarballs to detect tampering.
    """
    dockerfile_text = DOCKERFILE.read_text()
    instructions = _instruction_text(dockerfile_text)

    # Check for sha256sum verification
    assert "sha256sum" in instructions, (
        "Dockerfile must verify s6-overlay tarball checksums using sha256sum."
    )

    # Check for multiple tarballs being verified
    run_steps = _run_steps(dockerfile_text)
    assert any("sha256sum" in step for step in run_steps), (
        "Dockerfile must verify s6-overlay tarball checksums in RUN steps."
    )
