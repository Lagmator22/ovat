from __future__ import annotations

from typing import TYPE_CHECKING

from ovat.tui.vibe.app_server.config import AudioProviderView, TTSModelConfigView
from ovat.tui.vibe.cli.tts.tts_client_port import TTSClientPort, TTSResult

if TYPE_CHECKING:
    from ovat.tui.vibe.cli.tts.mistral_tts_client import MistralTTSClient


def make_tts_client(
    provider: AudioProviderView, model: TTSModelConfigView
) -> TTSClientPort:
    from ovat.tui.vibe.cli.tts.factory import make_tts_client as factory

    return factory(provider, model)


__all__ = ["MistralTTSClient", "TTSClientPort", "TTSResult", "make_tts_client"]


def __getattr__(name: str) -> object:
    if name == "MistralTTSClient":
        from ovat.tui.vibe.cli.tts.mistral_tts_client import MistralTTSClient

        return MistralTTSClient
    raise AttributeError(name)
