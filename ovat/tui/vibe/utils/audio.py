# Changed by OVAT from Mistral Vibe (commit 4ae5c59). See NOTICE.
from __future__ import annotations

from enum import auto

from ovat.tui.vibe._compat import StrEnum


class RecordingMode(StrEnum):
    BUFFER = auto()
    STREAM = auto()
