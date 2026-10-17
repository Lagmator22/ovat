from __future__ import annotations

from typing import TYPE_CHECKING

from ovat.tui.vibe.cli.voice_manager.voice_manager_port import (
    RecordingStartError,
    VoiceManagerPort,
)

if TYPE_CHECKING:
    from ovat.tui.vibe.cli.voice_manager.voice_manager import VoiceManager

__all__ = ["RecordingStartError", "VoiceManager", "VoiceManagerPort"]


def __getattr__(name: str) -> object:
    if name == "VoiceManager":
        from ovat.tui.vibe.cli.voice_manager.voice_manager import VoiceManager

        return VoiceManager
    raise AttributeError(name)
