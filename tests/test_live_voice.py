"""Live dictation with a fake microphone and Whisper, never real devices."""
import asyncio
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from ovat.agentic.voice import VoiceManager, open_microphone, whisper_settings
from ovat.tui.vibe.cli.voice_manager.voice_manager_port import (
    RecordingStartError, TranscribeState)
from ovat.tui.voice import PromptListener


class Recorder:
    def __init__(self):
        self.closed = False

    def __call__(self, callback):
        self.callback = callback
        self.closed = False
        return self

    def send(self, amplitude=0.2, seconds=0.1):
        self.callback(np.full((int(seconds * 16000), 1), amplitude,
                              dtype=np.float32), 0, None, None)

    def close(self):
        self.closed = True


class Listener:
    def __init__(self):
        self.states, self.partials, self.final, self.errors = [], [], [], []

    def on_transcribe_state_change(self, value):
        self.states.append(value)

    def on_transcribe_partial(self, value):
        self.partials.append(value)

    def on_transcribe_text(self, value):
        self.final.append(value)

    def on_transcribe_error(self, value):
        self.errors.append(value)


async def until(predicate):
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("Voice did not reach the expected state")


def test_live_partials_are_revised_before_stop_and_finalised_on_silence():
    async def go():
        recorder, listener = Recorder(), Listener()
        calls = []

        def generate(audio, **kwargs):
            calls.append(audio.copy())
            assert kwargs["language"] == "<|en|>"
            return SimpleNamespace(texts=["hello" if len(calls) == 1
                                          else "hello world"])
        manager = VoiceManager(lambda: ("model", "CPU"), recorder=recorder,
                               pipeline=lambda *a: SimpleNamespace(generate=generate),
                               interval=0.01)
        manager.add_listener(listener)
        manager.start_recording()
        recorder.send()
        await until(lambda: listener.partials == ["hello"])
        assert manager.transcribe_state == TranscribeState.RECORDING
        recorder.send()
        await until(lambda: listener.partials[-1] == "hello world")
        recorder.send(0, 2.5)
        await until(lambda: manager.transcribe_state == TranscribeState.IDLE)
        assert listener.final == ["hello world"]
        assert recorder.closed
        assert calls[0].dtype == np.float32
        assert listener.states[-2:] == [TranscribeState.FLUSHING,
                                       TranscribeState.IDLE]
        await manager.close()
    asyncio.run(go())


def test_stop_flushes_and_cancel_drops_late_inference():
    async def go():
        recorder, listener = Recorder(), Listener()
        manager = VoiceManager(lambda: ("m", None), recorder=recorder,
                               pipeline=lambda *a: SimpleNamespace(
                                   generate=lambda *a, **k: "one"), interval=0.01)
        manager.add_listener(listener)
        manager.start_recording()
        recorder.send()
        await manager.stop_recording()
        assert listener.final == ["one"] and recorder.closed
        manager.start_recording()
        recorder.send()
        manager.cancel_recording()
        await manager.close()
        assert listener.final == ["one"]
        assert manager.transcribe_state == TranscribeState.IDLE
    asyncio.run(go())


def test_capture_is_bounded_to_thirty_seconds():
    async def go():
        recorder, listener = Recorder(), Listener()
        lengths = []
        manager = VoiceManager(lambda: ("m", None), recorder=recorder,
                               pipeline=lambda *a: SimpleNamespace(generate=
                                   lambda audio, **k: lengths.append(len(audio)) or "cap"),
                               interval=0.01)
        manager.add_listener(listener)
        manager.start_recording()
        for _ in range(400):
            recorder.send()
        await until(lambda: manager.transcribe_state == TranscribeState.IDLE)
        assert lengths == [16000 * 30 - 1]
        assert recorder.closed
        await manager.close()
    asyncio.run(go())


def test_missing_model_and_mic_errors_are_plain_sentences(tmp_path):
    async def go():
        manager = VoiceManager(lambda: (str(tmp_path / "absent"), None))
        with pytest.raises(RecordingStartError, match="OVAT_WHISPER_MODEL"):
            manager.start_recording()
        assert manager.transcribe_state == TranscribeState.IDLE
        await manager.close()
    asyncio.run(go())


def test_missing_extra_or_microphone_gives_an_install_or_access_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(RecordingStartError, match="PortAudio"):
        open_microphone(lambda *a: None)
    def broken(**kwargs):
        raise OSError("no device")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=broken))
    with pytest.raises(RecordingStartError, match="microphone permission"):
        open_microphone(lambda *a: None)


def test_cancellation_waits_for_native_decoder_and_never_inserts_late_words():
    async def go():
        recorder, listener = Recorder(), Listener()
        entered, release = threading.Event(), threading.Event()
        def generate(audio, **kwargs):
            entered.set()
            release.wait(5)
            assert kwargs["streamer"]("") is True
            return "late words"
        manager = VoiceManager(lambda: ("m", None), recorder=recorder,
                               pipeline=lambda *a: SimpleNamespace(generate=generate),
                               interval=0.01)
        manager.add_listener(listener)
        manager.start_recording()
        recorder.send()
        await until(entered.is_set)
        manager.cancel_recording()
        assert recorder.closed
        try:
            with pytest.raises(RecordingStartError, match="finishing"):
                manager.start_recording()
        finally:
            release.set()
            await manager.close()
        assert not listener.final and not listener.partials
    asyncio.run(go())


def test_whisper_failure_closes_mic_and_reports_one_plain_message():
    async def go():
        recorder, listener = Recorder(), Listener()
        def broken(*args):
            raise RuntimeError("native internal details")
        manager = VoiceManager(lambda: ("m", None), recorder=recorder,
                               pipeline=broken, interval=0.01)
        manager.add_listener(listener)
        manager.start_recording()
        await until(lambda: manager.transcribe_state == TranscribeState.IDLE)
        assert recorder.closed
        assert len(listener.errors) == 1
        assert "native internal" not in listener.errors[0]
        await manager.close()
    asyncio.run(go())


def test_voice_waits_for_the_file_tools_cached_pipeline():
    from ovat.tools.transcribe import WHISPER_LOCK
    async def go():
        recorder = Recorder()
        entered = threading.Event()
        def generate(audio, **kwargs):
            entered.set()
            return "words"
        manager = VoiceManager(lambda: ("m", None), recorder=recorder,
                               pipeline=lambda *a: SimpleNamespace(generate=generate),
                               interval=0.01)
        WHISPER_LOCK.acquire()
        try:
            manager.start_recording()
            recorder.send()
            await asyncio.sleep(0.1)
            assert not entered.is_set()
        finally:
            WHISPER_LOCK.release()
        await until(entered.is_set)
        await manager.stop_recording()
        await manager.close()
    asyncio.run(go())


def test_model_resolution_prefers_workflow_then_environment(tmp_path, monkeypatch):
    from ovat.config.workflow import WorkflowConfig
    monkeypatch.setenv("OVAT_WHISPER_MODEL", "fallback")
    cfg = WorkflowConfig(model={"name": "m"}, tools=[{
        "name": "transcribe", "model": "whisper", "device": "CPU"}])
    assert whisper_settings(cfg, str(tmp_path)) == (str(tmp_path / "whisper"), "CPU")
    assert whisper_settings(None, str(tmp_path))[0] == str(tmp_path / "fallback")


def test_prompt_replacements_preserve_typed_text_and_do_not_duplicate():
    class Input:
        text = "before after"
        def get_full_text(self):
            return self.text
        def get_cursor_offset(self):
            return 7
    class Prompt(Listener):
        input_widget = Input()
        def replace_input(self, text, cursor_offset):
            self.text, self.cursor = text, cursor_offset
    prompt = Prompt()
    adapter = PromptListener(prompt)
    adapter.on_transcribe_state_change(TranscribeState.RECORDING)
    adapter.on_transcribe_partial("hi")
    adapter.on_transcribe_partial("hello ")
    adapter.on_transcribe_text("hello ")
    assert prompt.text == "before hello after"
    assert prompt.cursor == 13
    assert prompt.final == []


def test_voice_key_shows_live_words_then_finishes_without_duplication(monkeypatch):
    from ovat.tui import voice
    from tests.test_vibe_agent_tui import running, say, until as ui_until
    original = voice.PromptVoiceManager
    recorder = Recorder()
    manager = original(lambda: ("m", None), recorder=recorder,
                       pipeline=lambda *a: SimpleNamespace(
                           generate=lambda audio, **kw: "hello world"), interval=0.01)
    monkeypatch.setattr(voice, "PromptVoiceManager", lambda settings: manager)
    async def go():
        async with running() as (app, pilot, fake, handle):
            await pilot.press("ctrl+r")
            await ui_until(pilot, lambda: manager.transcribe_state == TranscribeState.RECORDING)
            recorder.send()
            await ui_until(pilot, lambda: app.query_one("#input").text == "hello world")
            assert manager.transcribe_state == TranscribeState.RECORDING
            recorder.send(0, 2.5)
            await ui_until(pilot, lambda: manager.transcribe_state == TranscribeState.IDLE)
            assert app.query_one("#input").text == "hello world"
            app.query_one("#input").load_text("")
            await say(pilot, "/voice")
            await ui_until(pilot, lambda: not manager.is_enabled)
    asyncio.run(go())


def test_clean_transcript_filters_hallucinations_and_exclamation_loops():
    from ovat.agentic.voice import clean_transcript
    assert clean_transcript("!!!!!!!!!!", 0.05) == ""
    assert clean_transcript("! . !", 0.05) == ""
    assert clean_transcript(" you", 0.005) == ""
    assert clean_transcript("Thank you.", 0.01) == ""
    assert clean_transcript("Thanks for watching.", 0.02) == ""
    assert clean_transcript("you!!!!!!!!!!!!!!!!Thank you. Thank you.!!!!!!!!!!I'm going to go to the bathroom.", 0.02) == ""
    assert clean_transcript("Hello world", 0.005) == "Hello world"
    assert clean_transcript("Thank you", 0.1) == "Thank you"
    assert clean_transcript("You.", 0.05) == ""
    assert clean_transcript("You you", 0.02) == ""
    assert clean_transcript("you you you", 0.15) == ""


def test_prompt_listener_updates_final_text_when_partial_was_seen():
    class Input:
        text = "before after"
        def get_full_text(self):
            return self.text
        def get_cursor_offset(self):
            return 7
    class Prompt(Listener):
        input_widget = Input()
        def replace_input(self, text, cursor_offset):
            self.text, self.cursor = text, cursor_offset
    prompt = Prompt()
    adapter = PromptListener(prompt)
    adapter.on_transcribe_state_change(TranscribeState.RECORDING)
    adapter.on_transcribe_partial("partial ")
    assert prompt.text == "before partial after"
    adapter.on_transcribe_text("final utterance ")
    assert prompt.text == "before final utterance after"
    assert prompt.cursor == len("before final utterance ")


def test_voice_key_control_modifier_does_not_abort_recording(monkeypatch):
    from ovat.tui import voice
    from tests.test_vibe_agent_tui import running, until as ui_until
    original = voice.PromptVoiceManager
    recorder = Recorder()
    manager = original(lambda: ("m", None), recorder=recorder,
                       pipeline=lambda *a: SimpleNamespace(
                           generate=lambda audio, **kw: "hello"), interval=0.01)
    monkeypatch.setattr(voice, "PromptVoiceManager", lambda settings: manager)
    async def go():
        async with running() as (app, pilot, fake, handle):
            await pilot.press("ctrl+r")
            await ui_until(pilot, lambda: manager.transcribe_state == TranscribeState.RECORDING)
            # Pressing control modifier alone should not abort recording
            await pilot.press("control")
            await asyncio.sleep(0.05)
            assert manager.transcribe_state == TranscribeState.RECORDING
            # Wait past debounce and press ctrl+r to stop recording
            await asyncio.sleep(0.45)
            await pilot.press("ctrl+r")
            await ui_until(pilot, lambda: manager.transcribe_state == TranscribeState.IDLE)
    asyncio.run(go())



def test_keys_pressed_while_recording_do_not_type_or_send(monkeypatch):
    """Only Ctrl+R stops a recording. Other keys used to fall through to the
    prompt: letters were typed and then overwritten by the transcript, and
    Enter sent the prompt in the middle of the recording."""
    from ovat.tui import voice
    from tests.test_vibe_agent_tui import running, until as ui_until
    recorder = Recorder()
    manager = voice.PromptVoiceManager(
        lambda: ("m", None), recorder=recorder,
        pipeline=lambda *a: SimpleNamespace(generate=lambda audio, **kw: ""),
        interval=0.01)
    monkeypatch.setattr(voice, "PromptVoiceManager", lambda settings: manager)

    async def go():
        async with running() as (app, pilot, fake, handle):
            await pilot.press("ctrl+r")
            await ui_until(pilot, lambda: manager.transcribe_state
                           == TranscribeState.RECORDING)
            await pilot.press("x", "enter")
            await pilot.pause(0.2)
            assert app.query_one("#input").text == ""
            assert fake.asked == []
            assert manager.transcribe_state == TranscribeState.RECORDING
            await asyncio.sleep(0.45)
            await pilot.press("ctrl+r")
            await ui_until(pilot, lambda: manager.transcribe_state
                           == TranscribeState.IDLE)
            # Esc is the one key let through: the app cancels with it
            await asyncio.sleep(0.45)
            await pilot.press("ctrl+r")
            await ui_until(pilot, lambda: manager.transcribe_state
                           == TranscribeState.RECORDING)
            await pilot.press("escape")
            await ui_until(pilot, lambda: manager.transcribe_state
                           == TranscribeState.IDLE)
    asyncio.run(go())


def test_stopping_does_not_decode_the_same_audio_twice():
    """Whisper is the slow part. When the last partial was empty, the final
    flush decoded the same, unchanged audio a second time and got the same
    nothing back."""
    async def go():
        recorder, calls = Recorder(), []

        def generate(audio, **kw):
            calls.append(audio.size)
            return ""
        manager = VoiceManager(lambda: ("m", None), recorder=recorder,
                               pipeline=lambda *a: SimpleNamespace(
                                   generate=generate), interval=0.01)
        manager.start_recording()
        recorder.send(seconds=0.5)
        await until(lambda: calls)
        await asyncio.sleep(0.05)       # loop ticks with no new audio
        await manager.stop_recording()
        assert calls == [8000]
    asyncio.run(go())


def _noise_with_clicks(seconds, level=0.004, click=0.06):
    """Quiet room noise (rms about `level`) with a short click every 0.5 s,
    like a keyboard tap or a laptop mic pop. The clicks have a high PEAK
    but almost no energy."""
    n = int(seconds * 16000)
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(n) * level).astype(np.float32)
    data[::8000] = click
    return data.reshape(-1, 1)


def test_room_noise_and_clicks_after_speech_do_not_keep_it_recording():
    async def go():
        recorder, listener = Recorder(), Listener()
        manager = VoiceManager(
            lambda: ("model", "CPU"), recorder=recorder, interval=0.01,
            pipeline=lambda *a: SimpleNamespace(generate=lambda audio, **k:
                                                SimpleNamespace(texts=["hello"])))
        manager.add_listener(listener)
        manager.start_recording()
        recorder.send(0.2, 0.5)                            # speech
        await until(lambda: listener.partials == ["hello"])
        recorder.callback(_noise_with_clicks(3.0), 0, None, None)  # then quiet
        await until(lambda: manager.transcribe_state == TranscribeState.IDLE)
        assert listener.final == ["hello"]
        assert recorder.closed
        await manager.close()
    asyncio.run(go())


def test_steady_background_hum_is_not_taken_for_speech_or_endless_speech():
    async def go():
        recorder, listener = Recorder(), Listener()
        manager = VoiceManager(
            lambda: ("model", "CPU"), recorder=recorder, interval=0.01,
            pipeline=lambda *a: SimpleNamespace(generate=lambda audio, **k:
                                                SimpleNamespace(texts=["hello"])))
        manager.add_listener(listener)
        manager.start_recording()
        recorder.send(0.2, 0.5)                            # speech
        await until(lambda: listener.partials == ["hello"])
        recorder.send(0.012, 3.0)                          # steady hum, rms 0.012
        await until(lambda: manager.transcribe_state == TranscribeState.IDLE)
        assert listener.final == ["hello"]
        await manager.close()
    asyncio.run(go())
