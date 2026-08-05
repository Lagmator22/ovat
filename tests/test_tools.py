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


def test_search_docs_reports_a_retrieval_failure_in_its_declared_shape():
    """A failing retriever must come back as a RESULT, not a different type.

    search_docs declares `-> list[dict]` and returned a bare string on error.
    The native loop hid it, because AgentLoop._execute calls str() on whatever
    a tool returns. Over MCP -- a path the README documents, with
    `command: ["python", "-m", "ovat.tools.search_docs"]` -- FastMCP validates
    the return against that annotation and raised instead:

        Invalid structured content returned by tool search_docs:
        'Error retrieving documents: db gone' is not of type 'array'

    So a locked database became a client-side exception rather than a sentence
    the model could read and recover from.
    """
    class Boom:
        def retrieve(self, query, top_k=5):
            raise RuntimeError("db gone")

    out = search_docs_impl("q", retriever=Boom())
    assert isinstance(out, list), f"declared list[dict], returned {type(out).__name__}"
    assert len(out) == 1
    assert "db gone" in out[0]["text"]      # the cause still reaches the model
    assert out[0]["text"].startswith("Error")


def test_search_docs_error_survives_a_real_mcp_round_trip():
    """The regression above, through the actual MCP server rather than a mock."""
    import asyncio

    from fastmcp import Client
    from ovat.tools import search_docs as sd

    class Boom:
        def retrieve(self, query, top_k=5):
            raise RuntimeError("db gone")

    async def call():
        sd.configure(Boom())
        try:
            async with Client(sd.mcp) as client:
                return await client.call_tool("search_docs", {"query": "q"})
        finally:
            sd.configure(None)              # never leak into other tests

    result = asyncio.run(call())
    assert "db gone" in str(result.content)


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


# Tool devices are a setting, not a literal

def test_transcribe_device_follows_the_router_by_default(monkeypatch):
    """It hardcoded "CPU". DeviceManager.get_whisper_device() existed and was
    ignored, so the routing table was advice nothing followed. Today the
    answer is still CPU, so nothing changes yet; the point is that the NPU/GPU
    stretch goal becomes a setting instead of a code edit."""
    from ovat.tools import transcribe

    monkeypatch.setattr(transcribe, "WHISPER_DEVICE", "")
    assert transcribe._resolve_device() == "CPU"     # what the router says here


def test_an_env_var_overrides_the_router(monkeypatch):
    from ovat.tools import transcribe

    monkeypatch.setattr(transcribe, "WHISPER_DEVICE", "NPU")
    assert transcribe._resolve_device() == "NPU"


def test_a_broken_openvino_falls_back_to_cpu(monkeypatch):
    """A tool that will not load at all is worse than one on a slower device."""
    from ovat.tools import transcribe

    monkeypatch.setattr(transcribe, "WHISPER_DEVICE", "")
    import ovat.core.device_manager as dm

    def explode():
        raise RuntimeError("no openvino here")

    monkeypatch.setattr(dm, "DeviceManager", explode)
    assert transcribe._resolve_device() == "CPU"


def test_describe_image_follows_the_llm_recommendation(monkeypatch):
    """A VLM is heavy, so it follows the LLM route (GPU when present) rather
    than whisper's CPU one."""
    from ovat.tools import describe_image

    monkeypatch.setattr(describe_image, "VLM_DEVICE", "GPU")
    assert describe_image._resolve_device() == "GPU"


def test_the_mcp_server_can_build_its_own_retriever_from_a_config(tmp_path, monkeypatch):
    """An MCP-served search_docs was PERMANENTLY stuck in stub mode.

    factory.build_agent calls search_docs_tool.configure(retriever), but that
    runs in the PARENT process. `type: mcp_stdio` launches a separate Python
    via `python -m ovat.tools.search_docs`, whose __main__ was just mcp.run()
    -- nothing in the child ever called configure(), and the child had no way
    to see the parent's rag: config. So _retriever stayed None for the whole
    life of the server and every call took the stub branch.

    Measured on the AI PC: the MCP path returned the 104-char
    "[stub] search_docs has no retriever wired yet" while the builtin path
    retrieved 1932 real characters from the same index in the same session.
    The comment in factory.py claiming the retriever is "shared by the
    in-process tool and the standalone MCP server" was simply wrong: objects
    do not cross a process boundary.

    The child needs its own config so it can build its own retriever.
    """
    from ovat.tools import search_docs as sd

    assert hasattr(sd, "configure_from_config"), \
        "the MCP child has no way to build a retriever from a workflow file"

    built = {}

    def fake_build_rag(cfg):
        built["name"] = cfg.model.name

        class R:
            def retrieve(self, query, top_k=5):
                return [{"text": "real chunk", "source": "notes.md",
                         "distance": 0.1}]
        return R()

    monkeypatch.setattr("ovat.agent.factory.build_rag", fake_build_rag)
    config = tmp_path / "w.yml"
    config.write_text(
        "model:\n  name: m\n"
        "rag:\n"
        "  embeddings:\n    provider: genai\n    model: x\n    dim: 384\n"
        "  retriever:\n    provider: sqlite-vec\n    db_path: x.db\n",
        encoding="utf-8")

    try:
        sd.configure_from_config(str(config))
        assert built["name"] == "m", "the child never loaded the config"
        out = sd.search_docs_impl("q", retriever=sd._retriever)
        assert out[0]["text"] == "real chunk"     # NOT the stub
    finally:
        sd.configure(None)


def test_the_mcp_server_stays_in_stub_mode_with_no_config():
    """Standalone-with-no-config must keep working: it is the documented way
    to run the tool by itself, and tests depend on that branch."""
    from ovat.tools import search_docs as sd

    sd.configure(None)
    out = sd.search_docs_impl("hello", retriever=sd._retriever)
    assert "[stub]" in out[0]["text"]
