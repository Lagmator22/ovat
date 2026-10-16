from __future__ import annotations

from enum import auto

from ovat.tui.vibe._compat import StrEnum

__all__ = ["AgentSafety", "AgentType"]


class AgentSafety(StrEnum):
    SAFE = auto()
    NEUTRAL = auto()
    DESTRUCTIVE = auto()
    SMART = auto()
    YOLO = auto()


class AgentType(StrEnum):
    AGENT = auto()
    SUBAGENT = auto()
