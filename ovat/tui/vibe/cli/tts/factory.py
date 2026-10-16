from __future__ import annotations

from ovat.tui.vibe.app_server.config import AudioProviderView, TTSModelConfigView
from ovat.tui.vibe.cli.audio_request_metadata import RequestMetadataGetter
from ovat.tui.vibe.cli.tts.mistral_tts_client import MistralTTSClient
from ovat.tui.vibe.cli.tts.tts_client_port import TTSClientPort


def make_tts_client(
    provider: AudioProviderView,
    model: TTSModelConfigView,
    metadata_getter: RequestMetadataGetter | None = None,
) -> TTSClientPort:
    return MistralTTSClient(
        provider=provider, model=model, metadata_getter=metadata_getter
    )
