"""Opt-in, signed-origin authored-file transport for streaming API consumers.

No new file-serving endpoint, model tool or Media workload lifecycle. Bytes stay
out of model history and use the existing native outbound path policy.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import stat

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_FILES = 5
MIME = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.webp': 'image/webp', '.gif': 'image/gif', '.pdf': 'application/pdf',
        '.txt': 'text/plain', '.md': 'text/markdown', '.csv': 'text/csv',
        '.zip': 'application/zip',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'}


def artifact_binding(body, *, profile, key, now=None):
    """Validate at ingress, not after a potentially long-running turn."""
    if profile == 'media' or body.get('hermes_artifact_transport') != 'inline-v1':
        return None
    from agent.chat_location import accept_context
    context = body.get('hermes_chat_context')
    accepted = accept_context(context, body=body, profile=profile, key=key, now=now)
    accepted.close()
    return {name: context[name] for name in
            ('subject_id', 'chat_id', 'message_id', 'user_message_id')}


def _read_artifact(path, root):
    # Walk from the filesystem root with borrowed directory capabilities rather
    # than reopening a checked pathname through mutable symlink ancestors.
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    fd = None
    try:
        parts = [*root.parts[1:], *path.relative_to(root).parts]
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= MAX_FILE_BYTES:
            raise ValueError('unsupported_artifact_size_or_type')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after) or len(data) != before.st_size:
            raise ValueError('artifact_changed_during_read')
        return data
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def artifact_event(text, *, binding, artifact_root, session_key=''):
    """Capture validated output bytes once; never emit paths as fetch authority."""
    if not binding or not isinstance(text, str) or 'MEDIA:' not in text:
        return None
    from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE, validate_media_delivery_path
    artifacts, seen, errors, total = [], set(), [], 0
    for match in MEDIA_TAG_CLEANUP_RE.finditer(text):
        if len(artifacts) >= MAX_FILES:
            errors.append('artifact_count_limit')
            break
        safe = validate_media_delivery_path(match.group('path'), session_key=session_key)
        if not safe:
            errors.append('artifact_path_unavailable')
            continue
        path = Path(safe)
        root = Path(artifact_root).absolute()
        if root.resolve() != root or not path.is_relative_to(root):
            errors.append('artifact_outside_profile_output_root')
            continue
        if safe in seen:
            continue
        seen.add(safe)
        mime = MIME.get(path.suffix.lower())
        if not mime:
            errors.append('unsupported_artifact_mime')
            continue
        try:
            data = _read_artifact(path, root)
        except (OSError, ValueError):
            errors.append('artifact_read_unavailable')
            continue
        if total + len(data) > MAX_TOTAL_BYTES:
            errors.append('artifact_total_limit')
            break
        total += len(data)
        digest = hashlib.sha256(data).hexdigest()
        artifacts.append({'artifact_id': digest, 'sha256': digest, 'filename': path.name,
                          'content_type': mime, 'size_bytes': len(data),
                          'data_base64': base64.b64encode(data).decode('ascii')})
    return {'schema': 'hermes.artifacts.v1', 'origin': dict(binding),
            'artifacts': artifacts, 'errors': errors[:MAX_FILES]}
