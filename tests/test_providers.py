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
from ovat.providers.llm_genai import GenAILLMProvider
from ovat.providers.embeddings_genai import GenAIEmbeddingsProvider
from ovat.providers.retriever_sqlitevec import SQLiteVecRetrieverProvider
from ovat.providers.vlm_genai import GenAIVLMProvider
from ovat.providers.llm_ovms import OVMSLLMProvider
from ovat.providers.embeddings_ovms import OVMSEmbeddingsProvider

# Local model paths (only on this Mac). Integration tests skip if missing.
MODELS_DIR = "/Users/lagmator22/OpenvinoDemo/OvaSearch/models"
LLM_MODEL = os.path.join(MODELS_DIR, "TinyLlama-1.1B-Chat-v1.0-INT4")
EMB_MODEL = os.path.join(MODELS_DIR, "bge-small-en-v1.5")
VLM_MODEL = os.path.join(MODELS_DIR, "Qwen2-VL-2B-Instruct-INT4")
DOG_IMG = "/Users/lagmator22/OpenvinoDemo/OvaSearch/data/dog.jpg"

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
