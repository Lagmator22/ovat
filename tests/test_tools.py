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
        def generate(self, samples):
            # I assert my reader handed me floats, then return a fake transcript.
            assert samples.dtype == np.float32
            return "hello world"

    out = transcribe_impl(str(path), pipeline=FakePipeline())
    assert out == "hello world"
