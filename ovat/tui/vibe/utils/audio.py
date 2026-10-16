from __future__ import annotations

from enum import auto

from ovat.tui.vibe._compat import StrEnum


class RecordingMode(StrEnum):
    BUFFER = auto()
    STREAM = auto()
