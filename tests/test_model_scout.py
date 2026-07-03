# tests/test_model_scout.py
"""Tests for model discovery + identification.

Note to myself: the fake folders below copy the REAL file signatures I
verified on disk — a Qwen2-VL export has openvino_language_model.xml and
vision parts (no plain openvino_model.xml); a Llama export has
openvino_model.xml + LlamaForCausalLM in config.json; whisper has
encoder+decoder; bge has model_type bert. No real model loads here.
"""
import json
import os

import pytest
import typer

from ovat.core.model_scout import find_models, identify_model, pick_chat_llm


def _make_model(root, name, files, config=None):
    folder = root / name
    folder.mkdir(parents=True)
    for f in files:
        (folder / f).write_text("")
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config))
    return str(folder)


def _llm(root, name="Llama-3.2-3B-Instruct-INT4"):
    return _make_model(root, name, ["openvino_model.xml", "openvino_model.bin"],
                       {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})


def _vlm(root, name="Qwen2-VL-2B-Instruct-INT4"):
    return _make_model(root, name,
                       ["openvino_language_model.xml",
                        "openvino_vision_embeddings_model.xml"],
                       {"model_type": "qwen2_vl"})


def _embedder(root, name="bge-small-en-v1.5"):
    return _make_model(root, name, ["openvino_model.xml"],
                       {"model_type": "bert"})


def _whisper(root, name="whisper-base"):
    return _make_model(root, name,
                       ["openvino_encoder_model.xml", "openvino_decoder_model.xml"],
                       {"model_type": "whisper"})


# identify_model — one folder, what is it

def test_identifies_all_four_kinds(tmp_path):
    assert identify_model(_llm(tmp_path))[0] == "llm"
    assert identify_model(_vlm(tmp_path))[0] == "vlm"
    assert identify_model(_embedder(tmp_path))[0] == "embeddings"
    assert identify_model(_whisper(tmp_path))[0] == "whisper"


def test_non_model_folders_are_rejected_kindly(tmp_path):
    assert identify_model(str(tmp_path / "missing"))[0] == "not-a-model"
    (tmp_path / "docs").mkdir()
    kind, why = identify_model(str(tmp_path / "docs"))
    assert kind == "not-a-model" and "openvino" in why


# find_models — scanning the roots

def test_find_models_scans_ovat_models_env(tmp_path, monkeypatch):
    _llm(tmp_path); _vlm(tmp_path); _embedder(tmp_path)
    (tmp_path / "random-junk").mkdir()               # ignored: no xml inside
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path))
    monkeypatch.chdir(tmp_path)                      # ./models absent — fine
    models = find_models()
    assert {m["kind"] for m in models} == {"llm", "vlm", "embeddings"}
    assert [m["kind"] for m in find_models("llm")] == ["llm"]


def test_env_root_may_be_a_model_folder_itself(tmp_path, monkeypatch):
    folder = _llm(tmp_path)
    monkeypatch.setenv("OVAT_MODELS", folder)        # points AT the model
    monkeypatch.chdir(tmp_path)
    assert any(m["path"] == folder for m in find_models("llm"))


# pick_chat_llm — the auto-detect choice

def test_pick_prefers_instruct_tuned_models(tmp_path, monkeypatch):
    _make_model(tmp_path, "base-llm", ["openvino_model.xml"],
                {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    _llm(tmp_path, "Zeta-Instruct")                  # later alphabetically
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    choice, llms = pick_chat_llm()
    assert choice["name"] == "Zeta-Instruct"         # instruct beats base
    assert len(llms) == 2


# resolve_chat_model — the chat command's guard rail

def test_resolve_rejects_a_vision_model_with_a_human_sentence(tmp_path, monkeypatch, capsys):
    from ovat.cli.main import resolve_chat_model
    vlm = _vlm(tmp_path)
    _llm(tmp_path)                                   # so it can suggest one
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(typer.Exit):
        resolve_chat_model(vlm)
    out = capsys.readouterr().out
    assert "not a text" in out                       # the human sentence
    assert "Llama-3.2-3B-Instruct-INT4" in out       # and the suggestion


def test_resolve_auto_detects_when_no_path_given(tmp_path, monkeypatch, capsys):
    from ovat.cli.main import resolve_chat_model
    llm = _llm(tmp_path)
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert resolve_chat_model(None) == llm
    assert "auto-detected" in capsys.readouterr().out


def test_resolve_with_nothing_found_gives_the_fix(tmp_path, monkeypatch, capsys):
    from ovat.cli.main import resolve_chat_model
    _vlm(tmp_path)                                   # wrong kind only
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(typer.Exit):
        resolve_chat_model(None)
    out = capsys.readouterr().out
    assert "No local text LLM found" in out
    assert "OVAT_MODELS" in out                      # names the fix
    assert "vlm" in out                              # shows what WAS found