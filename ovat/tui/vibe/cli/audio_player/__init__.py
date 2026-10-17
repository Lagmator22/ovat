from __future__ import annotations

from typing import TYPE_CHECKING

from ovat.tui.vibe.cli.audio_player.audio_player_port import (
    AlreadyPlayingError,
    AudioBackendUnavailableError,
    AudioFormat,
    AudioPlayerPort,
    NoAudioOutputDeviceError,
    UnsupportedAudioFormatError,
)

if TYPE_CHECKING:
    from ovat.tui.vibe.cli.audio_player.audio_player import AudioPlayer

__all__ = [
    "AlreadyPlayingError",
    "AudioBackendUnavailableError",
    "AudioFormat",
    "AudioPlayer",
    "AudioPlayerPort",
    "NoAudioOutputDeviceError",
    "UnsupportedAudioFormatError",
]


def __getattr__(name: str) -> object:
    if name == "AudioPlayer":
        from ovat.tui.vibe.cli.audio_player.audio_player import AudioPlayer

        return AudioPlayer
    raise AttributeError(name)
