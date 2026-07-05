# tests/test_tools.py
"""Tests for the two MCP tools, search_docs and transcribe.

Note to myself: I test the plain impl functions directly with fakes, so I do
not need a running MCP server, a vector database, or the Whisper model loaded.
This keeps the suite fast and green on my Mac.
"""
import wave

import numpy as np

from ovat.tools.search_docs import search_docs_impl
from ovat.tools.transcribe import transcribe_impl


# search_docs

def test_search_docs_stub_mode_returns_obvious_stub():
    out = search_docs_impl("hello", retriever=None)
    assert len(out) == 1
    assert "[stub]" in out[0]["text"]
    assert "hello" in out[0]["text"]


def test_search_docs_uses_real_retriever_when_given():
    class FakeRetriever:
        def retrieve(self, query, top_k=5):
            return [{"text": f"match for {query}", "distance": 0.1}]

    out = search_docs_impl("python", top_k=3, retriever=FakeRetriever())
    assert out == [{"text": "match for python", "distance": 0.1}]


# transcribe

def test_transcribe_reports_missing_file_clearly():
    out = transcribe_impl("/no/such/file.wav")
    assert out.startswith("Error:")
    assert "could not find" in out


def test_transcribe_reads_audio_and_calls_pipeline(tmp_path):
    # I write a tiny real WAV so _read_wav has something valid to parse.
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # 2 bytes is 16 bit, what my reader expects
        w.setframerate(16000)
        w.writeframes(np.zeros(1600, dtype=np.int16).tobytes())

    class FakePipeline:
        def generate(self, samples, language=None):
            # I assert my reader handed me floats and the language was passed
            # through in Whisper's token form, then return a fake transcript.
            assert samples.dtype == np.float32
            assert language == "<|en|>"
            return "hello world"

    out = transcribe_impl(str(path), pipeline=FakePipeline())
    assert out == "hello world"


def _write_wav(path, channels, rate):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(1600 * channels, dtype=np.int16).tobytes())


def test_transcribe_rejects_non_mono_audio(tmp_path):
    # A stereo file should fail clearly instead of producing a garbled result.
    path = tmp_path / "stereo.wav"
    _write_wav(path, channels=2, rate=16000)
    out = transcribe_impl(str(path), pipeline=object())   # pipeline never reached
    assert out.startswith("Error:") and "mono" in out


def test_transcribe_rejects_wrong_sample_rate(tmp_path):
    # A 44.1 kHz file should fail clearly rather than transcribe sped-up speech.
    path = tmp_path / "hires.wav"
    _write_wav(path, channels=1, rate=44100)
    out = transcribe_impl(str(path), pipeline=object())
    assert out.startswith("Error:") and "16 kHz" in out


# describe_image (the VLM tool): impl-level with a fake provider

def test_describe_image_missing_file_is_a_readable_error():
    from ovat.tools.describe_image import describe_image_impl
    out = describe_image_impl("/no/such/photo.jpg", provider=object())
    assert out.startswith("Error:") and "photo.jpg" in out


def test_describe_image_calls_the_provider_with_path_and_prompt(tmp_path):
    from ovat.tools.describe_image import describe_image_impl

    class FakeVLM:
        def __init__(self):
            self.seen = None

        def generate(self, prompt, images):
            self.seen = (prompt, images)
            return "a dog on a beach"

    image = tmp_path / "dog.jpg"
    image.write_bytes(b"not really a jpg, the fake never reads it")
    fake = FakeVLM()
    out = describe_image_impl(str(image), prompt="what animal?", provider=fake)
    assert out == "a dog on a beach"
    assert fake.seen == ("what animal?", [str(image)])


def test_describe_image_is_a_wired_builtin():
    from ovat.agent.factory import build_tools
    from ovat.config.workflow import WorkflowConfig

    cfg = WorkflowConfig(model={"name": "m"},
                         tools=[{"name": "describe_image", "type": "builtin"}])
    tools = build_tools(cfg)
    assert tools["describe_image"]["schema"]["function"]["name"] == "describe_image"
    # The bound function is the impl itself; a missing file proves it runs
    # without ever touching a real model.
    out = tools["describe_image"]["function"](image_path="/nope.png")
    assert out.startswith("Error:")
