"""Local microphone dictation through the existing Whisper pipeline.

It implements the copied client's VoiceManagerPort
(ovat/tui/vibe/cli/voice_manager/voice_manager_port.py).

Recording runs until Ctrl+R is pressed again, or until nothing but quiet has
been heard for SILENCE_SECONDS. Whisper takes at most about 30 s of 16 kHz
audio per call, so the speech is cut into segments: a pause of PAUSE_SECONDS
(or SEGMENT_SECONDS of continuous talking) ends a segment, its text is kept,
and a new one starts. One worker re-decodes the current segment while capture
continues; the shown text is the finished segments plus that live one, so
memory stays bounded however long the recording lasts.
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
SILENCE_SECONDS = 10.0       # quiet for this long ends the recording
PAUSE_SECONDS = 1.5          # quiet this long inside speech ends a segment
SEGMENT_SECONDS = 20         # a segment is cut here, under Whisper's 30 s
HARD_LIMIT_SECONDS = 90      # memory guard if decoding falls far behind
SILENCE_PEAK = 0.015
# End of speech is judged on a block's ENERGY (rms), not its peak: a click or
# a keyboard tap has a high peak and almost no energy, and with the peak the
# "2 seconds of quiet" counter reset on every one of them, so a recording in
# an ordinary room ran to the 30 s cap. A block counts as speech when its rms
# is 3x the room's quiet level (the quietest block of the last 3 s), kept
# between SPEECH_RMS_MIN and SPEECH_RMS_MAX so a loud room cannot hide speech.
SPEECH_RMS_MIN = 0.01
SPEECH_RMS_MAX = 0.04
NOISE_WINDOW = 30            # blocks of 0.1 s

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
        self._blocks = deque()          # the current segment's audio
        self._samples = 0               # samples in the current segment
        self._silent = 0                # quiet samples since its last speech
        self._idle = 0                  # quiet samples since ANY speech
        self._has_signal = False
        self._peak = 0.0                # the level bar, 0 to 1
        self._quiet = deque(maxlen=NOISE_WINDOW)

    @property
    def is_enabled(self):
        return self._enabled

    @property
    def transcribe_state(self):
        return self._state

    @property
    def peak(self):
        """The input level for the recording bar, 0 to 1. Square-rooted so
        that ordinary speech (rms 0.02 to 0.2) visibly moves the bar."""
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
        rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
        with self._lock:
            self._peak = min(1.0, (rms * 6) ** 0.5)
            self._quiet.append(rms)
            if self._samples >= RATE * HARD_LIMIT_SECONDS:
                # Decoding is far behind; keep memory bounded, but still
                # watch the level so the quiet timeout keeps working.
                if rms < min(max(3 * min(self._quiet), SPEECH_RMS_MIN),
                             SPEECH_RMS_MAX):
                    self._idle += block.size
                else:
                    self._idle = 0
                return
            self._blocks.append(block)
            self._samples += block.size
            threshold = min(max(3 * min(self._quiet), SPEECH_RMS_MIN),
                            SPEECH_RMS_MAX)
            if rms >= threshold:
                self._has_signal = True
                self._silent = 0
                self._idle = 0
            else:
                self._idle += block.size
                if self._has_signal:
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
        self._samples = self._silent = self._idle = 0
        self._has_signal = False
        self._peak = 0.0
        self._quiet.clear()
        self._stream = self._recorder(self._audio)
        self._set_state(TranscribeState.RECORDING)
        self._task = asyncio.create_task(self._run(model, device))

    def _snapshot(self):
        """(segment audio, heard speech in it, the segment is over, the whole
        recording is over)."""
        with self._lock:
            audio = np.concatenate(list(self._blocks)) if self._blocks else None
            segment_over = self._has_signal and (
                self._silent >= RATE * PAUSE_SECONDS
                or self._samples >= RATE * SEGMENT_SECONDS)
            recording_over = self._idle >= RATE * SILENCE_SECONDS
            return audio, self._has_signal, segment_over, recording_over

    def _end_segment(self, used):
        """Drop the first `used` samples (the audio that was decoded). What
        arrived while decoding stays and starts the next segment. It is
        scanned again for speech: decoding can take seconds, and a sentence
        said meanwhile must not be left waiting for a later loud block."""
        with self._lock:
            rest = (np.concatenate(list(self._blocks))[used:]
                    if self._blocks else np.zeros(0, dtype=np.float32))
            self._blocks.clear()
            if rest.size:
                self._blocks.append(rest)
            self._samples = rest.size
            self._silent = 0
            self._has_signal = False
            threshold = min(max(3 * min(self._quiet, default=0.0),
                                SPEECH_RMS_MIN), SPEECH_RMS_MAX)
            for start in range(0, rest.size, BLOCK):
                chunk = rest[start:start + BLOCK]
                if float(np.sqrt(np.mean(np.square(chunk)))) >= threshold:
                    self._has_signal, self._silent = True, 0
                elif self._has_signal:
                    self._silent += chunk.size

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
            result = pipe.generate(audio[:RATE * 30 - 1],
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
            done_text = []          # finished segments
            live = ""               # the current segment, decoded so far
            while not self._cancel.is_set():
                audio, signal, segment_over, recording_over = self._snapshot()
                final = recording_over or self._state == TranscribeState.FLUSHING
                if final:
                    self._stop_stream()
                    self._set_state(TranscribeState.FLUSHING)
                if ((signal or final) and audio is not None
                        and audio.size != last_size):
                    # Upstream py_whisper_pipeline.cpp: raw 16 kHz floats;
                    # a true streamer return stops decoding on cancellation.
                    live = await asyncio.to_thread(self._decode, pipe, audio,
                                                   self._cancel)
                    if self._cancel.is_set():
                        return
                    last_size = audio.size
                    if live or done_text:
                        self._notify("on_transcribe_partial",
                                     " ".join([*done_text, live]).strip())
                if final:
                    # The decode above already covered all of the audio.
                    text = " ".join([*done_text, live]).strip()
                    if text:
                        self._notify("on_transcribe_text", text)
                    else:
                        self._notify("on_transcribe_notice",
                                     "No speech detected; check the microphone.")
                    return
                if segment_over and audio is not None:
                    if live:
                        done_text.append(live)
                    self._end_segment(audio.size)
                    live, last_size = "", -1
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
