# ovat/tui/server/logs.py
"""Bounded in-memory debug log buffer and OVMS log tail reader."""
from __future__ import annotations

import logging
import os
import re
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ovat.tui.vibe.app_server.models import DebugLogEntry, DebugLogPage

LOG_BUFFER_SIZE = 500
_OVMS_TAIL_BYTES = 64 * 1024
_OVAT_LOGGER_NAME = "ovat"

_TS_RE = re.compile(
    r"^\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\]?\s*"
)
_LEVEL_RE = re.compile(
    r"^\[?(debug|info|warning|warn|error|critical)\]?\s*[:-]?\s*",
    re.IGNORECASE,
)


class BufferedLogHandler(logging.Handler):
    """Logging handler storing recent ``ovat.*`` records in a ring buffer."""

    def __init__(self, maxlen: int = LOG_BUFFER_SIZE) -> None:
        super().__init__(level=logging.DEBUG)
        self._lock = threading.Lock()
        self._seq = 0
        self._entries: deque[tuple[tuple[float, int, int], DebugLogEntry]] = deque(
            maxlen=maxlen
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            raw_line = self.format(record)
        except Exception:
            raw_line = record.getMessage()
        message = record.getMessage() or raw_line
        created = float(getattr(record, "created", 0.0) or 0.0)
        ts = datetime.fromtimestamp(created, tz=timezone.utc)
        level = record.levelname.upper()
        with self._lock:
            self._seq += 1
            seq = self._seq
            entry = DebugLogEntry(
                id=f"py-{seq}",
                timestamp=ts,
                ppid=os.getppid(),
                pid=os.getpid(),
                level=level,
                message=message,
                raw_line=raw_line,
            )
            self._entries.append(((created, 1, seq), entry))

    def entries(self) -> list[tuple[tuple[float, int, int], DebugLogEntry]]:
        with self._lock:
            return list(self._entries)


def attach_ovat_log_handler(handler: BufferedLogHandler) -> None:
    logging.getLogger(_OVAT_LOGGER_NAME).addHandler(handler)


def detach_ovat_log_handler(handler: BufferedLogHandler) -> None:
    logging.getLogger(_OVAT_LOGGER_NAME).removeHandler(handler)


def _parse_timestamp(raw: str) -> datetime | None:
    norm = raw[:-1] + "+00:00" if raw.endswith("Z") else raw.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(norm)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _read_ovms_log(
    cwd: str | None,
) -> list[tuple[tuple[float, int, int], DebugLogEntry]]:
    path = Path(cwd or os.getcwd()) / ".ovat" / "ovms.log"
    try:
        stat = path.stat()
    except OSError:
        return []
    if not path.is_file() or stat.st_size <= 0:
        return []

    start = max(0, stat.st_size - _OVMS_TAIL_BYTES)
    try:
        with path.open("rb") as fh:
            fh.seek(start)
            chunk = fh.read(_OVMS_TAIL_BYTES)
    except OSError:
        return []

    base_offset = start
    if start > 0:
        nl = chunk.find(b"\n")
        if nl < 0:
            return []
        chunk = chunk[nl + 1 :]
        base_offset += nl + 1

    fallback_base = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    ppid = os.getppid()
    pid = os.getpid()
    items: list[tuple[tuple[float, int, int], DebugLogEntry]] = []
    pos = base_offset

    for idx, raw_bytes in enumerate(chunk.splitlines(keepends=True)):
        line_offset = pos
        pos += len(raw_bytes)
        line = raw_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line.strip():
            continue

        rest = line
        ts: datetime | None = None
        ts_match = _TS_RE.match(rest)
        if ts_match is not None:
            ts = _parse_timestamp(ts_match.group(1))
            if ts is not None:
                rest = rest[ts_match.end() :]
        if ts is None:
            ts = fallback_base + timedelta(microseconds=idx)

        level = "INFO"
        lvl_match = _LEVEL_RE.match(rest)
        if lvl_match is not None:
            lvl = lvl_match.group(1).upper()
            level = "WARNING" if lvl == "WARN" else lvl
            rest = rest[lvl_match.end() :]

        message = rest or line
        entry = DebugLogEntry(
            id=f"ovms-{line_offset}",
            timestamp=ts,
            ppid=ppid,
            pid=pid,
            level=level,
            message=message,
            raw_line=line,
        )
        items.append(((ts.timestamp(), 0, idx), entry))

    return items


def read_logs(
    handler: BufferedLogHandler | None,
    cwd: str | None,
    *,
    limit: int = 100,
    offset: int = 0,
) -> DebugLogPage:
    """Return newest-first debug log entries combining ``ovat.*`` and OVMS logs."""
    combined = (handler.entries() if handler is not None else []) + _read_ovms_log(cwd)
    combined.sort(key=lambda item: item[0], reverse=True)

    safe_offset = max(0, int(offset))
    safe_limit = max(1, min(int(limit), LOG_BUFFER_SIZE))
    page = [entry for _, entry in combined[safe_offset : safe_offset + safe_limit]]
    has_more = safe_offset + len(page) < len(combined)
    cursor = safe_offset + len(page) if has_more else None
    return DebugLogPage(entries=page, has_more=has_more, cursor=cursor)
