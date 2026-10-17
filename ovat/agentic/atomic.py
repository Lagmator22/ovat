# ovat/agentic/atomic.py
"""Write a text file so that a crash leaves the old one, never half of a new one.

Note to myself: four files are rewritten while the app runs (a saved
conversation, the UI preferences, the saved permission rules, and the workflow
that /config edits). All of them go through here: the text is written whole to
a temporary file next to the target and moved over it with os.replace, which
is one step on every platform.

`mode` sets the replacement file's permissions. Conversations and rules
are owner-only, like the A2A token; workflow edits pass the existing mode.
A fresh exclusive temporary file prevents old files or symlinks from being
opened as the temporary output.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text(path: str | os.PathLike, text: str, *, mode: int = 0o600
               ) -> Path:
    """Write `text` to `path` atomically. Raises OSError when the disk says
    no: the caller decides how to tell the person."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.",
                                     suffix=".tmp", dir=target.parent)
    temp = Path(temp_name)
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(temp, mode)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return target
