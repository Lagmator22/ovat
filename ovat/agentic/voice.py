"""Local microphone dictation through the existing Whisper pipeline.

It implements the copied client's VoiceManagerPort
(ovat/tui/vibe/cli/voice_manager/voice_manager_port.py).
Whisper takes a whole 16 kHz utterance, so a single worker re-decodes the
growing buffer while capture continues. Only the newest partial is shown.
No recording or transcript is sent to a service or written to disk.
"""
from __future__ import annotations

import asyncio
import os
import re
import threading
from collections import deque

import numpy as np

from ovat.tui.vibe.cli.voice_manager.voice_manager_port import (
    RecordingStartError, TranscribeState)
from ovat.tui.vibe.utils.audio import RecordingMode

RATE = 16000
BLOCK = 1600
MAX_SECONDS = 30
SILENCE_SECONDS = 2.0
SPEECH_PEAK = 0.035
SILENCE_PEAK = 0.015

WHISPER_HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "bye", "bye-bye",
    "i'm going to go to the bathroom", "thank you thank you",
}


def clean_transcript(text: str, peak: float) -> str:
    if not text:
        return ""
    exclamations = text.count("!")
    if exclamations > 4 and exclamations > len(text) * 0.15:
        return ""
    cleaned = re.sub(r"!{2,}", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or set(cleaned) <= {"!", ".", " ", "?", ","}:
        return ""
    norm = re.sub(r"[^\w\s]", "", cleaned.lower()).strip()
    words = norm.split()
    if words and set(words) <= {"you"}:
        return ""
    if words and (
        set(words) <= {"thank", "you"}
        or norm in WHISPER_HALLUCINATIONS
        or any(h in norm for h in ("thanks for watching", "thank you for watching"))
    ):
        if peak < 0.08:
            return ""
    return cleaned


def whisper_settings(config, cwd: str) -> tuple[str, str | None]:
    """Use the workflow's transcribe tool, then the existing env fallback."""
    from ovat.tools.transcribe import _configured_model
    tool = next((t for t in getattr(config, "tools", ())
                 if t.name == "transcribe" and t.type == "builtin"), None)
    model = _configured_model(getattr(tool, "model", None))
    if not os.path.isabs(model):
        model = os.path.join(cwd, model)
    return model, getattr(tool, "device", None)


def open_microphone(callback):
    """InputStream's callback only copies a small block into bounded memory.

    API: python-sounddevice.readthedocs.io/en/latest/api/streams.html.
    PortAudio must be installed separately on Linux.
    """
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        raise RecordingStartError(
            'Voice needs pip install "ovat[voice]"; on Linux also install '
            'the system PortAudio library (libportaudio2).') from None
    stream = None
    try:
        stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                blocksize=BLOCK, callback=callback)
        stream.start()
        return stream
    except Exception:
        if stream is not None:
            stream.close()
        raise RecordingStartError(
            "Cannot open the microphone; check the input device and your "
            "terminal's microphone permission.") from None


class VoiceManager:
    """Vibe VoiceManagerPort, with local rolling transcription.

    A listener with on_transcribe_partial receives replacement text. Other
    listeners receive one final on_transcribe_text, as the port expects.
    Settings are a callable because /workflow may replace the configuration.
    """

    def __init__(self, settings, *, recorder=open_microphone, pipeline=None,
                 interval=1.0):
        self._settings = settings
        self._recorder = recorder
        self._load = pipeline
        self._interval = interval
        self._listeners = []
        self._enabled = True
        self._state = TranscribeState.IDLE
        self._stream = None
        self._task = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._blocks = deque(maxlen=RATE * MAX_SECONDS // BLOCK)
        self._samples = 0
        self._silent = 0
        self._has_signal = False
        self._peak = 0.0

    @property
    def is_enabled(self):
        return self._enabled

    @property
    def transcribe_state(self):
        return self._state

    @property
    def peak(self):
        return self._peak

    def add_listener(self, listener):
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener):
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self, method, *args):
        for listener in tuple(self._listeners):
            fn = getattr(listener, method, None)
            if fn is not None:
                try:
                    fn(*args)
                except Exception:
                    continue

    def _set_state(self, state):
        self._state = state
        self._notify("on_transcribe_state_change", state)

    def apply_enabled(self, enabled):
        self._enabled = enabled
        if not enabled:
            self.cancel_recording()
        self._notify("on_voice_mode_change", enabled)

    def _audio(self, data, frames, timestamp, status):
        if self._cancel.is_set():
            return
        block = np.asarray(data, dtype=np.float32).reshape(-1).copy()
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(block))) if block.size else 0.0
        with self._lock:
            # Stop collecting at the cap even if inference is still busy.
            if self._samples >= RATE * MAX_SECONDS:
                return
            block = block[:RATE * MAX_SECONDS - self._samples]
            self._blocks.append(block)
            self._samples += block.size
            self._peak = peak
            if peak >= SPEECH_PEAK:
                self._has_signal = True
                self._silent = 0
            elif self._has_signal:
                self._silent += block.size

    def start_recording(self, mode=RecordingMode.STREAM):
        if self._state != TranscribeState.IDLE or not self._enabled:
            return
        if self._task is not None and not self._task.done():
            raise RecordingStartError("Voice is finishing the cancelled utterance; try again shortly.")
        model, device = self._settings()
        if self._load is None and not os.path.isdir(model):
            raise RecordingStartError(
                "Voice needs a local Whisper model; set the transcribe "
                "tool's model or OVAT_WHISPER_MODEL to its folder.")
        self._cancel = threading.Event()
        self._blocks.clear()
        self._samples = self._silent = 0
        self._has_signal = False
        self._peak = 0.0
        self._stream = self._recorder(self._audio)
        self._set_state(TranscribeState.RECORDING)
        self._task = asyncio.create_task(self._run(model, device))

    def _snapshot(self):
        with self._lock:
            audio = np.concatenate(list(self._blocks)) if self._blocks else None
            done = (self._samples >= RATE * MAX_SECONDS or
                    (self._has_signal and self._silent >= RATE * SILENCE_SECONDS))
            return audio, self._has_signal, done

    def _stop_stream(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()

    def _decode(self, pipe, audio, cancel):
        if audio is None or audio.size < 400:
            return ""
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < SILENCE_PEAK:
            return ""
        from ovat.tools.transcribe import WHISPER_LOCK
        with WHISPER_LOCK:
            if cancel.is_set():
                return ""
            # The upstream streamer supports short-form audio below 30 s.
            # The penalty and the cap stop a "!!!!" loop on near silence;
            # 224 tokens is more than 30 s of speech (about 90 words), so the
            # cap never cuts real dictation short. Whisper's own limit is 448.
            result = pipe.generate(audio[:RATE * MAX_SECONDS - 1],
                                   language="<|en|>", return_timestamps=False,
                                   repetition_penalty=1.2, max_new_tokens=224,
                                   streamer=lambda _: cancel.is_set())
        texts = getattr(result, "texts", None)
        raw = (texts[0] if texts else str(result)).strip()
        return clean_transcript(raw, peak)

    async def _run(self, model, device):
        try:
            if self._load is None:
                from ovat.tools.transcribe import _load_pipeline
                load = _load_pipeline
            else:
                load = self._load
            pipe = await asyncio.to_thread(load, model, device)
            last_size = -1
            text = ""
            while not self._cancel.is_set():
                audio, signal, done = self._snapshot()
                final = done or self._state == TranscribeState.FLUSHING
                if final:
                    self._stop_stream()
                    self._set_state(TranscribeState.FLUSHING)
                if (signal or final) and audio is not None and audio.size != last_size:
                    # Upstream py_whisper_pipeline.cpp: raw 16 kHz floats;
                    # a true streamer return stops decoding on cancellation.
                    text = await asyncio.to_thread(self._decode, pipe, audio,
                                                   self._cancel)
                    if self._cancel.is_set():
                        return
                    last_size = audio.size
                    if text:
                        self._notify("on_transcribe_partial", text)
                if final:
                    # The decode above already covered all of the audio.
                    if text:
                        self._notify("on_transcribe_text", text)
                    else:
                        self._notify("on_transcribe_notice",
                                     "No speech detected; check the microphone.")
                    return
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._notify("on_transcribe_error",
                         "Local voice transcription failed; check the Whisper "
                         "model and microphone.")
        finally:
            self._stop_stream()
            self._set_state(TranscribeState.IDLE)

    async def stop_recording(self):
        if self._state == TranscribeState.RECORDING:
            self._stop_stream()
            self._set_state(TranscribeState.FLUSHING)
        task = self._task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    def cancel_recording(self):
        self._cancel.set()
        self._stop_stream()
        # A cancelled to_thread future does not stop its native worker.
        # Keep the task until Whisper acknowledges the stop callback, so a
        # second utterance cannot enter the same cached pipeline meanwhile.
        self._set_state(TranscribeState.IDLE)

    async def close(self):
        self.cancel_recording()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        self._listeners.clear()
