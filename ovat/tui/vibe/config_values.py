from __future__ import annotations

from typing import Literal

AUTO_THEME = "auto"
FALLBACK_THEME = "ansi-dark"
DEFAULT_THEME = AUTO_THEME
DEFAULT_LOG_LEVEL = "WARNING"

AudioClient = Literal["mistral"]
ThinkingLevel = Literal["off", "low", "medium", "high", "max"]
THINKING_LEVELS: tuple[ThinkingLevel, ...] = ("off", "low", "medium", "high", "max")

TranscriptionEncoding = Literal["pcm_s16le"]
SpeechOutputFormat = Literal["pcm", "wav", "mp3", "flac", "opus"]
