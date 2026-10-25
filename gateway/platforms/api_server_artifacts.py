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
import json
import sqlite3
import time
from contextlib import contextmanager

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_FILES = 5


class CompletedReceiptStore:
    """Finite transport retention only; never schedules or resumes producer work."""

    TTL = 6 * 3600
    CAPACITY = 256
    MAX_RECEIPT_BYTES = 15 * 1024 * 1024
    MAX_TOTAL_BYTES = 128 * 1024 * 1024

    def __init__(self, path):
        self.path = Path(path)
        if self.path.parent.is_symlink() or not self.path.parent.is_dir():
            raise ValueError('Receipt storage requires an existing trusted directory')
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
                raise ValueError('Invalid receipt storage owner or type')
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS receipts '
                       '(origin TEXT PRIMARY KEY, receipt TEXT NOT NULL, expires REAL NOT NULL)')

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _origin(origin):
        keys = {'subject_id', 'chat_id', 'message_id', 'user_message_id'}
        if (not isinstance(origin, dict) or set(origin) != keys
                or any(not isinstance(v, str) or not 0 < len(v) <= 256 for v in origin.values())):
            raise ValueError('Invalid receipt origin')
        return json.dumps(origin, sort_keys=True, separators=(',', ':'))

    def save(self, origin, receipt, *, now=None):
        key = self._origin(origin)
        if (not isinstance(receipt, dict) or receipt.get('schema') != 'hermes.artifacts.v1'
                or receipt.get('origin') != origin):
            raise ValueError('Invalid completed receipt')
        encoded = json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if len(encoded.encode()) > self.MAX_RECEIPT_BYTES:
            raise ValueError('Completed receipt exceeds retention bound')
        now = time.time() if now is None else now
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM receipts WHERE expires <= ?', (now,))
            prior = db.execute('SELECT receipt FROM receipts WHERE origin=?', (key,)).fetchone()
            if prior:
                if prior[0] != encoded:
                    raise ValueError('Completed receipt identity already sealed')
                return
            if db.execute('SELECT count(*) FROM receipts').fetchone()[0] >= self.CAPACITY:
                raise RuntimeError('Completed receipt retention is full')
            retained = db.execute('SELECT COALESCE(SUM(length(CAST(receipt AS BLOB))),0) FROM receipts').fetchone()[0]
            if retained + len(encoded.encode()) > self.MAX_TOTAL_BYTES:
                raise RuntimeError('Completed receipt retention byte budget is full')
            db.execute('INSERT INTO receipts VALUES (?,?,?)', (key, encoded, now + self.TTL))

    def get(self, origin, *, now=None):
        key = self._origin(origin)
        now = time.time() if now is None else now
        with self._connect() as db:
            row = db.execute('SELECT receipt FROM receipts WHERE origin=? AND expires>?', (key, now)).fetchone()
        return json.loads(row[0]) if row else None
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
