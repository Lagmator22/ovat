from __future__ import annotations

from ovat.tui.vibe.app_server.config import AudioProviderView, TranscribeModelConfigView
from ovat.tui.vibe.cli.audio_request_metadata import RequestMetadataGetter
from ovat.tui.vibe.cli.transcribe.transcribe_client_port import TranscribeClientPort


def make_transcribe_client(
    provider: AudioProviderView,
    model: TranscribeModelConfigView,
    metadata_getter: RequestMetadataGetter | None = None,
) -> TranscribeClientPort:
    # OVAT: mistral_transcribe_client not ported (it needs the mistralai package;
    # OVAT supplies its own transcription backend, or none). Imported here so
    # this module still imports.
    from ovat.tui.vibe.cli.transcribe.mistral_transcribe_client import (
        MistralTranscribeClient,
    )

    return MistralTranscribeClient(
        provider=provider, model=model, metadata_getter=metadata_getter
    )
