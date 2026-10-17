from __future__ import annotations

from ovat.tui.vibe.app_server.config import AudioProviderView, TTSModelConfigView
from ovat.tui.vibe.cli.audio_request_metadata import RequestMetadataGetter
from ovat.tui.vibe.cli.tts.tts_client_port import TTSClientPort


def make_tts_client(
    provider: AudioProviderView,
    model: TTSModelConfigView,
    metadata_getter: RequestMetadataGetter | None = None,
) -> TTSClientPort:
    # OVAT: mistral_tts_client not ported (it needs the mistralai package; OVAT
    # supplies its own speech backend, or none). Imported here so this module
    # still imports.
    from ovat.tui.vibe.cli.tts.mistral_tts_client import MistralTTSClient

    return MistralTTSClient(
        provider=provider, model=model, metadata_getter=metadata_getter
    )
