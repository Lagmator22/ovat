"""Local microphone dictation through the existing Whisper pipeline.

It implements the copied client's VoiceManagerPort
(ovat/tui/vibe/cli/voice_manager/voice_manager_port.py).

Recording runs until Ctrl+R is pressed again, or until nothing but quiet has
been heard for SILENCE_SECONDS. Whisper takes at most about 30 s of 16 kHz
audio per call, so the speech is cut into segments: a pause of PAUSE_SECONDS
(or CUT_AFTER_SECONDS of continuous talking) ends a segment, its text is kept,
and a new one starts. One worker re-decodes the current segment while capture
continues, so memory stays bounded however long the recording lasts.

Shown text is append-only. Whisper revises earlier words when more audio
arrives, and showing every revision made the sentence flicker and erase
itself (worst at full speed, where the buffer is long). So a word is
COMMITTED only when two decodes in a row agree on it (the "LocalAgreement"
rule from the whisper-streaming work) and only committed words are shown,
about a second behind the speech; they are never removed. The tentative tail
after them is added when its phrase ends or the recording does. Speech with no
pause is cut, after CUT_AFTER_SECONDS, where Whisper itself ended a phrase
(its segment timestamps), so a word is not split in two. Whisper-base is
unstable on long buffers (measured: words dropped at 16 s, garbage at 24 s).
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
PAUSE_SECONDS = 1.0          # quiet this long inside speech ends a segment
CUT_AFTER_SECONDS = 10       # speech with no pause is cut where Whisper
                             # itself ended a phrase
HARD_CUT_SECONDS = 20        # no phrase end found: cut at the quietest point
CUT_SEARCH_SECONDS = 4
CUT_KEEP_SECONDS = 1.5       # the newest audio is still being spoken
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


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def _same(a: list[str], b: list[str]) -> bool:
    return [_norm(w) for w in a] == [_norm(w) for w in b]


def _aligned_end(committed: list[str], words: list[str]) -> int:
    """Where the committed words end inside a fresh decode `words`. Usually
    the decode starts with them; when Whisper reworded something inside the
    committed part, find where they still line up."""
    n = len(committed)
    if _same(words[:n], committed):
        return n
    from difflib import SequenceMatcher
    blocks = SequenceMatcher(None, [_norm(w) for w in committed],
                             [_norm(w) for w in words],
                             autojunk=False).get_matching_blocks()
    ends = [b.b + b.size for b in blocks if b.size]
    return max(ends) if ends else min(n, len(words))


def agree(committed: list[str], previous: list[str], words: list[str]
          ) -> tuple[list[str], list[str]]:
    """One step of LocalAgreement-2. Returns (committed, tentative tail).

    The words that this decode and the previous one share at the start become
    committed (never fewer than before). Everything after the committed words
    in this decode is the tentative tail, which may still change."""
    shared = 0
    for a, b in zip(previous, words):
        if _norm(a) != _norm(b):
            break
        shared += 1
    aligned = _aligned_end(committed, words)
    if shared > aligned:
        # Both decodes agree past the committed words (even if Whisper has
        # reworded one of them since): commit the new words too.
        committed = [*committed, *words[aligned:shared]]
    return committed, words[_aligned_end(committed, words):]


def drop_overlap(done: list[str], words: list[str], most: int = 3) -> list[str]:
    """`words` without the start that repeats the end of `done`. A forced cut
    can fall inside a word, and then both segments decode it."""
    for k in range(min(most, len(done), len(words)), 0, -1):
        if _same(done[-k:], words[:k]):
            return words[k:]
    return words


def phrase_cut(phrases, size: int):
    """(sample to cut at, the words before it) from Whisper's segments: the
    end of the last phrase that finished at least CUT_KEEP_SECONDS before the
    end of the audio. (None, None) when there is no such phrase."""
    limit = size / RATE - CUT_KEEP_SECONDS
    done = [p for p in phrases if p[1] <= limit]
    if not done:
        return None, None
    return (int(done[-1][1] * RATE),
            " ".join(p[2].strip() for p in done).split())


def quiet_point(audio: np.ndarray, floor: int = 0) -> int:
    """The sample to cut long speech at: the middle of the quietest 0.1 s
    block in the last CUT_SEARCH_SECONDS (leaving the last 0.3 s alone, which
    is still being spoken). Even fast speech has short dips between words."""
    end = audio.size - 3 * BLOCK // 2
    start = max(floor, end - RATE * CUT_SEARCH_SECONDS)
    best, best_rms = end, None
    for at in range(start, end - BLOCK + 1, BLOCK // 2):
        chunk = audio[at:at + BLOCK]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        if best_rms is None or rms < best_rms:
            best, best_rms = at + BLOCK // 2, rms
    return best


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
        """(segment audio, heard speech in it, a pause ended the segment, the
        segment has run long, the whole recording is over)."""
        with self._lock:
            audio = np.concatenate(list(self._blocks)) if self._blocks else None
            paused = (self._has_signal
                      and self._silent >= RATE * PAUSE_SECONDS)
            long_run = (self._has_signal
                        and self._samples >= RATE * CUT_AFTER_SECONDS)
            recording_over = self._idle >= RATE * SILENCE_SECONDS
            return audio, self._has_signal, paused, long_run, recording_over

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
        """(text, phrases): the cleaned text and Whisper's segments as
        (start s, end s, text), which the caller cuts long speech at."""
        if audio is None or audio.size < 400:
            return "", []
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < SILENCE_PEAK:
            return "", []
        from ovat.tools.transcribe import WHISPER_LOCK
        with WHISPER_LOCK:
            if cancel.is_set():
                return "", []
            # The upstream streamer supports short-form audio below 30 s.
            # The penalty and the cap stop a "!!!!" loop on near silence;
            # 224 tokens is more than 30 s of speech (about 90 words), so the
            # cap never cuts real dictation short. Whisper's own limit is 448.
            result = pipe.generate(audio[:RATE * 30 - 1],
                                   language="<|en|>", return_timestamps=True,
                                   repetition_penalty=1.2, max_new_tokens=224,
                                   streamer=lambda _: cancel.is_set())
        texts = getattr(result, "texts", None)
        raw = (texts[0] if texts else str(result)).strip()
        text = clean_transcript(raw, peak)
        phrases = []
        if text:
            for c in getattr(result, "chunks", None) or []:
                try:
                    phrases.append((float(c.start_ts), float(c.end_ts),
                                    str(c.text)))
                except (AttributeError, TypeError, ValueError):
                    phrases = []
                    break
        return text, phrases

    async def _run(self, model, device):
        try:
            if self._load is None:
                from ovat.tools.transcribe import _load_pipeline
                load = _load_pipeline
            else:
                load = self._load
            pipe = await asyncio.to_thread(load, model, device)
            last_size = -1
            done = []               # words of finished segments
            seg, previous, tail = [], [], []     # the current segment
            last_text = None
            phrases = []            # Whisper's segments of the last decode
            overlap = False         # the last cut was forced, may repeat a word

            async def decode(chunk):
                text, phrases = await asyncio.to_thread(
                    self._decode, pipe, chunk, self._cancel)
                return text.split(), phrases

            def show():
                """Live text is what is settled: finished segments plus the
                words two decodes agreed on. The tentative tail is left out,
                so what is shown only ever grows; the tail is added when its
                phrase ends or the recording does."""
                nonlocal last_text
                body = drop_overlap(done, seg) if overlap else seg
                text = " ".join([*done, *body])
                if text and text != last_text:
                    last_text = text
                    self._notify("on_transcribe_partial", text)

            while not self._cancel.is_set():
                audio, signal, paused, long_run, over = self._snapshot()
                final = over or self._state == TranscribeState.FLUSHING
                if final:
                    self._stop_stream()
                    self._set_state(TranscribeState.FLUSHING)
                if ((signal or final) and audio is not None
                        and audio.size != last_size):
                    # Upstream py_whisper_pipeline.cpp: raw 16 kHz floats;
                    # a true streamer return stops decoding on cancellation.
                    words, phrases = await decode(audio)
                    if self._cancel.is_set():
                        return
                    last_size = audio.size
                    seg, tail = agree(seg, previous, words)
                    previous = words
                    show()
                if final:
                    # The decode above already covered all of the audio.
                    body = [*seg, *tail]
                    text = " ".join([*done, *(drop_overlap(done, body)
                                              if overlap else body)])
                    if text:
                        self._notify("on_transcribe_text", text)
                    else:
                        self._notify("on_transcribe_notice",
                                     "No speech detected; check the microphone.")
                    return
                cut, forced, cut_words = None, False, None
                if audio is not None and paused:
                    cut = audio.size
                elif audio is not None and long_run:
                    forced = True
                    cut, cut_words = (phrase_cut(phrases, audio.size)
                                      if last_size == audio.size
                                      else (None, None))
                    if cut is None and audio.size >= RATE * HARD_CUT_SECONDS:
                        # No phrase end to use: the quietest point, then
                        # decode exactly the audio before it.
                        cut = quiet_point(audio)
                        cut_words, _ = await decode(audio[:cut])
                        if self._cancel.is_set():
                            return
                if cut is not None:
                    if cut_words is not None:
                        seg = [*seg, *cut_words[_aligned_end(seg, cut_words):]]
                        tail = []
                    done += (drop_overlap(done, [*seg, *tail]) if overlap
                             else [*seg, *tail])
                    seg, previous, tail, last_size = [], [], [], -1
                    # Not after a forced cut: the words spoken after the cut
                    # are not decoded yet, and showing now would hide them
                    # for a moment. The next decode shows everything.
                    if not forced:
                        show()
                    overlap = forced
                    self._end_segment(cut)
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
