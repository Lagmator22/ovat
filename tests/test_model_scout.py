# tests/test_model_scout.py
"""Tests for model discovery + identification.

Note to myself: the fake folders below copy the REAL file signatures I
verified on disk; a Qwen2-VL export has openvino_language_model.xml and
vision parts (no plain openvino_model.xml); a Llama export has
openvino_model.xml + LlamaForCausalLM in config.json; whisper has
encoder+decoder; bge has model_type bert. No real model loads here.
"""
import json
import os

import pytest
import typer

from ovat.core.model_scout import find_models, identify_model, pick_chat_llm


@pytest.fixture(autouse=True)
def _fake_home(tmp_path, monkeypatch):
    """Point HOME at an empty folder for every test in this file.

    find_models() scans `~/models` as one of its roots, so without this these
    tests describe whichever models happen to sit in the developer's home
    directory rather than the code. On the AI PC, which really does have
    ~/models/OpenVINO/Qwen3-8B-int4-ov, three of them failed: a real 8B model
    joined the counts (`assert 3 == 2`) and the "nothing found" case found
    something and never raised.

    expanduser reads USERPROFILE (then HOMEDRIVE+HOMEPATH) on Windows and
    HOME on POSIX, so every variable it consults is redirected.
    """
    home = tmp_path / "fake-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOMEDRIVE", os.path.splitdrive(str(home))[0])
    monkeypatch.setenv("HOMEPATH", os.path.splitdrive(str(home))[1])
    assert os.path.expanduser("~") == str(home)      # the redirect really took
    return home


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


def _unified(root, name="Qwen3.5-0.8B-int4-ov"):
    """A UNIFIED multimodal export: vision parts, but a real text LLM too.

    This file list and config are copied from the actual
    OpenVINO/Qwen3.5-0.8B-int4-ov repository, downloaded and loaded on
    2026-08-01. It looks exactly like a VLM on disk, which is the whole
    problem: the layout check alone calls it "vlm" and chat refuses it,
    even though openvino_genai answers text-only prompts from it happily.
    """
    return _make_model(root, name,
                       ["openvino_language_model.xml",
                        "openvino_text_embeddings_model.xml",
                        "openvino_vision_embeddings_model.xml",
                        "openvino_vision_embeddings_merger_model.xml",
                        "openvino_vision_embeddings_pos_model.xml"],
                       {"model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "text_config": {"model_type": "qwen3_5_text"}})


# identify_model: one folder, what is it

def test_identifies_all_four_kinds(tmp_path):
    assert identify_model(_llm(tmp_path))[0] == "llm"
    assert identify_model(_vlm(tmp_path))[0] == "vlm"
    assert identify_model(_embedder(tmp_path))[0] == "embeddings"
    assert identify_model(_whisper(tmp_path))[0] == "whisper"


def test_a_unified_multimodal_export_is_not_mistaken_for_a_plain_vlm(tmp_path):
    """Qwen3.5 ships vision parts AND serves text, so "vlm" is the wrong answer.

    Verified on the real model: openvino_genai answers a text-only prompt from
    OpenVINO/Qwen3.5-0.8B-int4-ov in 1.3s. Calling it "vlm" is what makes
    `ovat chat` refuse the model Intel's own guidance recommends.
    """
    kind, why = identify_model(_unified(tmp_path))
    assert kind == "unified", f"got {kind!r} ({why})"
    # A pure vision model must NOT be swept up by the same rule.
    assert identify_model(_vlm(tmp_path))[0] == "vlm"


def test_a_unified_model_is_offered_as_both_a_text_llm_and_a_vision_model(
        tmp_path, monkeypatch):
    """One model, two roles. Filtering by either kind has to return it, or the
    audio+multimodal example needs two downloads instead of one."""
    folder = _unified(tmp_path)
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert [m["path"] for m in find_models("llm")] == [folder]
    assert [m["path"] for m in find_models("vlm")] == [folder]


def test_chat_accepts_a_unified_model_instead_of_refusing_it(
        tmp_path, monkeypatch):
    """The regression this whole change exists to prevent: `ovat chat` told a
    user their recommended model "is not a text LLM" and exited 1."""
    from ovat.cli.main import resolve_chat_model

    folder = _unified(tmp_path)
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert resolve_chat_model(folder) == folder      # no typer.Exit


def test_non_model_folders_are_rejected_kindly(tmp_path):
    assert identify_model(str(tmp_path / "missing"))[0] == "not-a-model"
    (tmp_path / "docs").mkdir()
    kind, why = identify_model(str(tmp_path / "docs"))
    assert kind == "not-a-model" and "openvino" in why


# find_models: scanning the roots

def test_find_models_scans_ovat_models_env(tmp_path, monkeypatch):
    _llm(tmp_path); _vlm(tmp_path); _embedder(tmp_path)
    (tmp_path / "random-junk").mkdir()               # ignored: no xml inside
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path))
    monkeypatch.chdir(tmp_path)                      # ./models absent; fine
    models = find_models()
    assert {m["kind"] for m in models} == {"llm", "vlm", "embeddings"}
    assert [m["kind"] for m in find_models("llm")] == ["llm"]


def test_env_root_may_be_a_model_folder_itself(tmp_path, monkeypatch):
    folder = _llm(tmp_path)
    monkeypatch.setenv("OVAT_MODELS", folder)        # points AT the model
    monkeypatch.chdir(tmp_path)
    assert any(m["path"] == folder for m in find_models("llm"))


# pick_chat_llm: the auto-detect choice

def test_pick_prefers_instruct_tuned_models(tmp_path, monkeypatch):
    _make_model(tmp_path, "base-llm", ["openvino_model.xml"],
                {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    _llm(tmp_path, "Zeta-Instruct")                  # later alphabetically
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    choice, llms = pick_chat_llm()
    assert choice["name"] == "Zeta-Instruct"         # instruct beats base
    assert len(llms) == 2


# resolve_chat_model: the chat command's guard rail

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

# Models nested one folder deeper: the layout `ovms --pull` actually creates

@pytest.fixture
def only_these_models(tmp_path, monkeypatch):
    """Isolate the scan from this machine.

    _roots() ALWAYS includes cwd/models and ~/models, so without this the
    repo's own models/bge-small-en-v1.5 turns up in every assertion and the
    tests describe the developer's disk rather than the code.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "nohome"))
    return tmp_path



def _export(folder, model_type="qwen3", arch="Qwen3ForCausalLM"):
    """Write the minimum that identify_model() calls a text LLM."""
    import json
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (folder / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [arch]}),
        encoding="utf-8")
    return folder


def test_a_model_nested_under_an_org_folder_is_found(only_these_models, tmp_path, monkeypatch):
    """The bug that made chat unusable on the AI PC.

    `ovms --pull` lays models out by ORG: models/OpenVINO/Qwen3-8B-int4-ov,
    which `ovat models list` prints as "OpenVINO\\Qwen3-8B-int4-ov". A
    one-level scan looked at models/OpenVINO, found no openvino*.xml in it,
    wrote it off and never went deeper, so a model sitting right there was
    reported as "No local text LLM found".
    """
    from ovat.core.model_scout import find_models, pick_chat_llm

    _export(tmp_path / "models" / "OpenVINO" / "Qwen3-8B-int4-ov")
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path / "models"))

    names = [m["name"] for m in find_models()]
    assert "Qwen3-8B-int4-ov" in names, (
        f"the org-nested model was not found; saw {names}")
    choice, _ = pick_chat_llm()
    assert choice is not None and choice["name"] == "Qwen3-8B-int4-ov"


def test_a_model_directly_in_the_root_still_works(only_these_models, tmp_path, monkeypatch):
    """The flat layout must not regress while fixing the nested one."""
    from ovat.core.model_scout import find_models

    _export(tmp_path / "models" / "Llama-3B-instruct")
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path / "models"))
    assert [m["name"] for m in find_models()] == ["Llama-3B-instruct"]


def test_the_root_itself_being_a_model_still_works(only_these_models, tmp_path, monkeypatch):
    """OVAT_MODELS pointed straight at one model folder."""
    from ovat.core.model_scout import find_models

    _export(tmp_path / "Llama-3B")
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path / "Llama-3B"))
    assert [m["name"] for m in find_models()] == ["Llama-3B"]


def test_scanning_does_not_descend_into_a_model_folder(only_these_models, tmp_path, monkeypatch):
    """An export can contain subfolders. Listing them as siblings of the model
    itself would fill the picker with junk the user cannot use."""
    from ovat.core.model_scout import find_models

    model = _export(tmp_path / "models" / "Qwen3-8B")
    (model / "openvino_tokenizer").mkdir()
    (model / "openvino_tokenizer" / "openvino_model.xml").write_text(
        "<net/>", encoding="utf-8")
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path / "models"))
    assert [m["name"] for m in find_models()] == ["Qwen3-8B"]


def test_a_third_level_is_not_scanned(only_these_models, tmp_path, monkeypatch):
    """Two levels covers the org layout. Deeper would start crawling
    unrelated trees for no benefit, which on a big disk is slow."""
    from ovat.core.model_scout import find_models

    _export(tmp_path / "models" / "a" / "b" / "TooDeep")
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path / "models"))
    assert [m["name"] for m in find_models()] == []


def test_searched_roots_names_the_real_folders(only_these_models, tmp_path, monkeypatch):
    """The error message used to say "scanned OVAT_MODELS, ./models,
    ~/models", which left the user unable to tell which paths that was."""
    from ovat.core.model_scout import searched_roots

    monkeypatch.setenv("OVAT_MODELS", str(tmp_path / "one"))
    roots = searched_roots()
    assert str(tmp_path / "one") in roots
    assert any(r.endswith("models") for r in roots)


def test_an_unreadable_folder_is_skipped_not_fatal(only_these_models, tmp_path, monkeypatch):
    """A permission error mid-scan must cost that folder, not the command."""
    from ovat.core import model_scout

    _export(tmp_path / "models" / "OpenVINO" / "Qwen3-8B")
    monkeypatch.setenv("OVAT_MODELS", str(tmp_path / "models"))

    real_listdir = model_scout.os.listdir

    def deny(path):
        if path.endswith("OpenVINO"):
            raise PermissionError("denied")
        return real_listdir(path)

    monkeypatch.setattr(model_scout.os, "listdir", deny)
    assert model_scout.find_models() == []      # empty, but no exception
