# ovat/agentic/atomic.py
"""Atomic text file writes using exclusive same-directory temporary files and ``os.replace``.

Ensures concurrent readers and interrupted writes never observe partially
written state (saved conversations, preferences, permission rules, workflow edits).
Defaults file mode to user-only access (``0600`` on POSIX), while callers editing
existing configs pass the preserved mode.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text(path: str | os.PathLike, text: str, *, mode: int = 0o600,
               newline: str | None = None) -> Path:
    """Write `text` to `path` atomically. Raises OSError when the disk says
    no: the caller decides how to tell the person. `newline` is open()'s:
    "" writes the text byte for byte."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.",
                                     suffix=".tmp", dir=target.parent)
    temp = Path(temp_name)
    try:
        with open(fd, "w", encoding="utf-8", newline=newline) as handle:
            handle.write(text)
        os.chmod(temp, mode)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return target
