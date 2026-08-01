# tests/test_providers.py
"""
Tests written as we go, so the 80%-coverage success criterion isn't a
panic at the end.

Two kinds of tests:
  1. CONTRACT/UNIT tests (fast, no models or server): class shapes, ABC rules,
     URL building, construction.
  2. INTEGRATION tests (slow, load real models): auto-skipped when the local
     OpenVINO models aren't present, so the suite stays green everywhere.

OVMS plugs are tested for construction only (creating the client doesn't
connect); full OVMS calls get tested once a server is up (Docker / AI PC).
"""
import json
import os

import pytest

from ovat.core.device_manager import DeviceManager
from ovat.core.model_manager import ModelManager
from ovat.core.model_server import ModelServer
from ovat.providers.base import (
    LLMProvider,
    EmbeddingsProvider,
    RetrieverProvider,
    VLMProvider,
)
from ovat.providers import llm_genai
from ovat.providers.llm_genai import GenAILLMProvider
from ovat.providers.embeddings_genai import GenAIEmbeddingsProvider
from ovat.providers.retriever_sqlitevec import SQLiteVecRetrieverProvider
from ovat.providers.vlm_genai import GenAIVLMProvider
from ovat.providers.llm_ovms import OVMSLLMProvider
from ovat.providers.embeddings_ovms import OVMSEmbeddingsProvider

# Where the real OpenVINO test models live. OVAT_TEST_MODELS_DIR makes these
# integration tests runnable on ANY machine (AI PC, a mentor's box); the old
# hardcoded personal path meant they silently skipped everywhere but one Mac.
MODELS_DIR = os.environ.get(
    "OVAT_TEST_MODELS_DIR", "/Users/lagmator22/OpenvinoDemo/OvaSearch/models"
)
LLM_MODEL = os.path.join(MODELS_DIR, "TinyLlama-1.1B-Chat-v1.0-INT4")
EMB_MODEL = os.path.join(MODELS_DIR, "bge-small-en-v1.5")
VLM_MODEL = os.path.join(MODELS_DIR, "Qwen2-VL-2B-Instruct-INT4")
DOG_IMG = os.environ.get(
    "OVAT_TEST_IMAGE", "/Users/lagmator22/OpenvinoDemo/OvaSearch/data/dog.jpg"
)

needs_llm = pytest.mark.skipif(not os.path.isdir(LLM_MODEL), reason="local LLM model not present")
needs_emb = pytest.mark.skipif(not os.path.isdir(EMB_MODEL), reason="local embedding model not present")
needs_vlm = pytest.mark.skipif(
    not (os.path.isdir(VLM_MODEL) and os.path.isfile(DOG_IMG)),
    reason="local VLM model or test image not present",
)


# ───────────────────────── DeviceManager (Layer 9) ─────────────────────────

def test_device_manager_runs():
    dm = DeviceManager()
    s = dm.summary()
    assert "CPU" in s["available"]
    assert s["whisper"] == "CPU"


def test_routing_falls_back_to_cpu():
    dm = DeviceManager()
    if dm.available == ["CPU"]:
        assert dm.get_llm_device() == "CPU"
        assert dm.get_embedding_device() == "CPU"


def test_summary_has_all_keys():
    dm = DeviceManager()
    assert set(dm.summary().keys()) == {"available", "llm", "embeddings", "whisper"}


# ───────────────────── Contract tests for sockets/plugs ────────────────────

def test_abcs_cannot_be_instantiated():
    for abc in (LLMProvider, EmbeddingsProvider, RetrieverProvider, VLMProvider):
        with pytest.raises(TypeError):
            abc()


def test_plugs_are_subclasses_of_their_sockets():
    assert issubclass(GenAILLMProvider, LLMProvider)
    assert issubclass(GenAIEmbeddingsProvider, EmbeddingsProvider)
    assert issubclass(SQLiteVecRetrieverProvider, RetrieverProvider)
    assert issubclass(GenAIVLMProvider, VLMProvider)
    assert issubclass(OVMSLLMProvider, LLMProvider)
    assert issubclass(OVMSEmbeddingsProvider, EmbeddingsProvider)


# ──────────── OVMS plugs + core classes (construct only, no server) ─────────

def test_ovms_providers_construct_without_server():
    # Building the OpenAI client doesn't connect; safe without OVMS running.
    OVMSLLMProvider(base_url="http://localhost:8000/v3", model="x")
    OVMSEmbeddingsProvider(base_url="http://localhost:8000/v3", model="y")


def test_model_server_builds_urls():
    s = ModelServer("my-model", port=9001)
    assert s.base_url == "http://localhost:9001/v3"
    assert s.health_url == "http://localhost:9001/v2/health/ready"
    assert s.process is None


def test_model_manager_stores_binary():
    assert ModelManager("ovms").ovms == "ovms"


# ──────────────── Integration tests (load real models, may skip) ───────────

@needs_llm
def test_genai_llm_returns_contract_shape():
    p = GenAILLMProvider(LLM_MODEL, max_new_tokens=16)
    r = p.chat([{"role": "user", "content": "Say hello."}])
    assert r["finish_reason"] == "stop"
    assert isinstance(r["content"], str) and len(r["content"]) > 0
    assert r["tool_calls"] is None


@needs_emb
def test_genai_embeddings_have_right_dim():
    vectors = GenAIEmbeddingsProvider(EMB_MODEL).embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384


@needs_emb
def test_retriever_finds_relevant_doc():
    retriever = SQLiteVecRetrieverProvider(GenAIEmbeddingsProvider(EMB_MODEL), dim=384)
    retriever.add(["Python is a language used for AI.", "The Eiffel Tower is in Paris."])
    hits = retriever.retrieve("what do people code machine learning in?", top_k=1)
    assert hits[0]["text"] == "Python is a language used for AI."


@needs_vlm
def test_genai_vlm_describes_image():
    out = GenAIVLMProvider(VLM_MODEL, max_new_tokens=32).generate(
        "Describe this image in one sentence.", [DOG_IMG]
    )
    assert "dog" in out.lower()


# The generation cap: a number caps the answer, None means "until EOS".

class _FakePipe:
    """Records the kwargs GenAILLMProvider hands openvino_genai."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(kwargs)
        return "generated"


def _fake_genai_provider(monkeypatch, **kwargs):
    pipe = _FakePipe()
    monkeypatch.setattr(llm_genai.ov_genai, "LLMPipeline",
                        lambda path, device: pipe)
    return GenAILLMProvider("model-dir", **kwargs), pipe


def test_generation_is_capped_by_default(monkeypatch):
    """The default must stay a real cap, not openvino_genai's 2**64-1.

    A base model or a bad chat template can fail to emit EOS, and uncapped
    that means minutes of CPU generation with the UI frozen.
    """
    provider, pipe = _fake_genai_provider(monkeypatch)
    assert provider.max_new_tokens == 256
    provider.chat([{"role": "user", "content": "hi"}])
    assert pipe.calls[-1]["max_new_tokens"] == 256


def test_none_omits_the_cap_entirely(monkeypatch):
    # Passing None through would not be read as a number: the argument has to
    # be absent for openvino_genai to mean "no cap".
    provider, pipe = _fake_genai_provider(monkeypatch, max_new_tokens=None)
    provider.chat([{"role": "user", "content": "hi"}])
    assert "max_new_tokens" not in pipe.calls[-1]


def test_the_cap_is_re_read_per_call_so_it_can_be_retuned_live(monkeypatch):
    # This is what lets the TUI's /tokens work without a ~30s model reload.
    provider, pipe = _fake_genai_provider(monkeypatch)
    provider.chat([{"role": "user", "content": "hi"}])
    provider.max_new_tokens = 1024
    provider.chat([{"role": "user", "content": "hi"}])
    provider.max_new_tokens = None
    provider.chat([{"role": "user", "content": "hi"}])
    assert pipe.calls[0]["max_new_tokens"] == 256
    assert pipe.calls[1]["max_new_tokens"] == 1024
    assert "max_new_tokens" not in pipe.calls[2]


def test_the_cap_applies_on_the_streaming_path_too(monkeypatch):
    provider, pipe = _fake_genai_provider(monkeypatch, max_new_tokens=64)
    provider.chat([{"role": "user", "content": "hi"}], on_token=lambda t: False)
    assert pipe.calls[-1]["max_new_tokens"] == 64
    assert "streamer" in pipe.calls[-1]        # still streaming


# Unified multimodal models load through VLMPipeline, not LLMPipeline.

def _unified_export(tmp_path, name="Qwen3.5-0.8B-int4-ov"):
    """The real Qwen3.5 file layout (verified against the downloaded repo)."""
    folder = tmp_path / name
    folder.mkdir()
    for f in ("openvino_language_model.xml",
              "openvino_text_embeddings_model.xml",
              "openvino_vision_embeddings_model.xml"):
        (folder / f).write_text("")
    (folder / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5",
         "architectures": ["Qwen3_5ForConditionalGeneration"]}))
    return str(folder)


class _FakeVLMPipe(_FakePipe):
    """VLMPipeline has the chat-session calls LLMPipeline does not."""

    def __init__(self):
        super().__init__()
        self.chat_open = False

    def start_chat(self):
        self.chat_open = True

    def finish_chat(self):
        self.chat_open = False


def test_a_unified_model_loads_through_vlm_pipeline(tmp_path, monkeypatch):
    """MEASURED, not guessed: on the real Qwen3.5-0.8B, LLMPipeline builds
    fine and then dies at generate() with

        Port for tensor name input_ids was not found.

    VLMPipeline answers the same prompt in 1.3s. Choosing the wrong one is a
    C++ traceback on the user's first question, so the choice is asserted.
    """
    vlm_pipe = _FakeVLMPipe()

    def _boom(*a, **k):
        raise AssertionError("LLMPipeline cannot generate from this export")

    monkeypatch.setattr(llm_genai.ov_genai, "LLMPipeline", _boom)
    monkeypatch.setattr(llm_genai.ov_genai, "VLMPipeline",
                        lambda path, device: vlm_pipe)

    provider = GenAILLMProvider(_unified_export(tmp_path), max_new_tokens=32)
    out = provider.chat([{"role": "user", "content": "hi"}])

    assert out["content"] == "generated"
    assert vlm_pipe.calls[-1]["max_new_tokens"] == 32
    # VLMPipeline.generate REQUIRES images; omitting it is a TypeError.
    assert vlm_pipe.calls[-1]["images"] == []


def test_a_plain_text_export_still_uses_llm_pipeline(tmp_path, monkeypatch):
    """The fix must not reroute ordinary text models through VLMPipeline."""
    folder = tmp_path / "Llama-3.2-3B-Instruct-INT4"
    folder.mkdir()
    (folder / "openvino_model.xml").write_text("")
    (folder / "config.json").write_text(json.dumps(
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))

    pipe = _FakePipe()
    monkeypatch.setattr(llm_genai.ov_genai, "LLMPipeline",
                        lambda path, device: pipe)
    monkeypatch.setattr(llm_genai.ov_genai, "VLMPipeline",
                        lambda *a, **k: pytest.fail("should not be used"))

    provider = GenAILLMProvider(str(folder), max_new_tokens=16)
    provider.chat([{"role": "user", "content": "hi"}])
    assert "images" not in pipe.calls[-1]        # the text path, unchanged


def test_model_manager_always_passes_the_repository_path(monkeypatch):
    """`ovms --list_models` alone has no repository to list and exits
    non-zero, which is why `ovat models list` never worked."""
    calls = []

    class Result:
        returncode = 0
        stdout = "Qwen3-8B-int4-ov\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Result()

    monkeypatch.setattr("ovat.core.model_manager.subprocess.run", fake_run)
    mgr = ModelManager("ovms")

    assert mgr.list_models("C:/models") == ["Qwen3-8B-int4-ov"]
    assert "--model_repository_path" in calls[-1]
    assert "C:/models" in calls[-1]

    mgr.pull("OpenVINO/x", "C:/models")
    assert "--model_repository_path" in calls[-1]


def test_model_manager_reports_a_refusal_as_a_readable_error(monkeypatch):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "  no such repository  "

    monkeypatch.setattr("ovat.core.model_manager.subprocess.run",
                        lambda cmd, **kw: Result())
    with pytest.raises(RuntimeError, match="no such repository"):
        ModelManager("ovms").list_models()
