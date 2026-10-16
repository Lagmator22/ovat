from __future__ import annotations

# OVAT edit: Vibe's VoiceManager (Mistral's speech to text) is not copied;
# OVAT implements the port with local Whisper (ovat/agentic/voice.py).
from ovat.tui.vibe.cli.voice_manager.voice_manager_port import (
    RecordingStartError,
    VoiceManagerPort,
)

__all__ = ["RecordingStartError", "VoiceManagerPort"]
