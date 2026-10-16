from __future__ import annotations

from typing import TYPE_CHECKING

from ovat.tui.vibe.app_server.config import AudioProviderView, TranscribeModelConfigView
from ovat.tui.vibe.cli.transcribe.transcribe_client_port import (
    TranscribeClientPort,
    TranscribeDone,
    TranscribeError,
    TranscribeEvent,
    TranscribeSessionCreated,
    TranscribeTextDelta,
)

def make_transcribe_client(
    provider: AudioProviderView, model: TranscribeModelConfigView
) -> TranscribeClientPort:
    from ovat.tui.vibe.cli.transcribe.factory import make_transcribe_client as factory

    return factory(provider, model)


__all__ = [
    "TranscribeClientPort",
    "TranscribeDone",
    "TranscribeError",
    "TranscribeEvent",
    "TranscribeSessionCreated",
    "TranscribeTextDelta",
    "make_transcribe_client",
]
