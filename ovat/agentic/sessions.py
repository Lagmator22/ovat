# ovat/agentic/sessions.py
"""Saved conversations on disk: where they live, and the agent app's store.

Note to myself: three things save conversations, and they must agree on the
folder or `/resume` and `/load` would each see half of them:

    .ovat/sessions/<name>.json        the launcher's /chat, and /save in the
                                      agent apps (a plain list of messages)
    .ovat/sessions/agent/<id>.json    `ovat agent` conversations (a record,
                                      see SessionStore)

So the folder is decided here, once. This file never imports textual: the
launcher's chat screen and the agent app server both use it.

A record is plain JSON (a dict), written whole to a temporary file and moved
over the old one, so a crash mid-write leaves the previous save, never half
of one. What goes into a record is the server's business; this file only
keeps them.
"""
from __future__ import annotations

import json
from pathlib import Path

from ovat.agentic import atomic

SESSIONS_DIR = Path(".ovat") / "sessions"
AGENT_SUBDIR = "agent"
# Bumped when a record changes shape; a reader skips what it cannot read.
FORMAT = 1


def sessions_dir(cwd: str) -> Path:
    """The folder of saved conversations for the project in `cwd`."""
    return Path(cwd) / SESSIONS_DIR


def safe_name(name: str) -> str:
    """A session id or name as a bare file name: no folders, no dots."""
    return "".join(c for c in name if c.isalnum() or c in "-_")


class SessionStore:
    """The agent app's saved conversations: one JSON record per file."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    def path(self, session_id: str) -> Path:
        name = safe_name(session_id)
        if not name:
            raise ValueError(f"Not a session id: {session_id!r}")
        return self.folder / f"{name}.json"

    def save(self, record: dict) -> Path:
        """Write `record` (it carries its own "id"). Raises OSError when the
        disk says no: the caller decides how to tell the person."""
        # Owner-only, like the A2A token: a conversation holds the person's
        # documents and every tool's output.
        return atomic.write_text(
            self.path(record["id"]),
            json.dumps({**record, "format": FORMAT}, indent=1))

    def load(self, session_id: str) -> dict | None:
        """The record, or None when there is none (or it cannot be read)."""
        try:
            return self._read(self.path(session_id))
        except ValueError:
            return None

    def list(self) -> list[dict]:
        """Every readable record, the most recently updated first."""
        try:
            files = sorted(self.folder.glob("*.json"))
        except OSError:
            return []
        records = [r for r in map(self._read, files) if r is not None]
        return sorted(records, key=lambda r: r.get("updated_at") or 0,
                      reverse=True)

    def delete(self, session_id: str) -> bool:
        try:
            self.path(session_id).unlink()
        except (OSError, ValueError):
            return False
        return True

    @staticmethod
    def _read(path: Path) -> dict | None:
        # A half-written or hand-edited file must not hide every other
        # session from the list, so anything odd reads as "not there".
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            return None
        if (not isinstance(record, dict) or record.get("format") != FORMAT
                or record.get("id") != path.stem):
            return None
        return record


def store_for(cwd: str | None) -> SessionStore | None:
    """The store for the project in `cwd`; None (nothing is saved) without
    one. Tests replace this with a store in a temporary folder."""
    if not cwd:
        return None
    return SessionStore(sessions_dir(cwd) / AGENT_SUBDIR)
