# ovat/a2a/audit.py
"""The record of every answer that left this machine.

This is the evidence behind "you can see everything that left": one JSON
line per finished task, with who asked, what they asked, and what was sent
back. Never a token.

Opened when the server starts, not on the first answer, so a log that
cannot be written stops `ovat a2a serve` at once with a sentence, instead
of the server running for hours and recording nothing.
"""
import json
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        # The log holds every question and answer: owner-only. A folder that
        # already exists is the owner's choice and is not touched.
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Creating it proves it is writable now. 0600 applies only to a NEW
        # file; one that already exists keeps its mode.
        os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                         0o600))
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        """Append one line. A failure is logged, never raised: the answer
        has already been sent, and the caller should not see a crash."""
        line = json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                           **record}, ensure_ascii=False, default=str)
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            log.error("could not write the A2A audit log %s: %s",
                      self.path, exc)

    def tail(self, n: int) -> list[dict]:
        """The last n records, oldest first. Unreadable lines are skipped."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines[-n:] if n > 0 else []:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out
