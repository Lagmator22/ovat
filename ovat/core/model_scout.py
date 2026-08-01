# ovat/core/model_scout.py
"""Find and identify OpenVINO models on this machine.

Why this exists: a new user pointed `chat` at a Qwen2-VL folder (a VISION
model) and got a C++ traceback about a missing `input_ids` tensor port.
Nothing told them "that's not a text LLM". Every exported OpenVINO model
folder carries the evidence of what it is; this module reads it so commands
can refuse politely, suggest the right folder, or pick one automatically.

Identification signals (checked in this order: file layout first because it
never lies, then config.json):
  - a vision layout whose family is ALSO a text LLM      -> a UNIFIED export
  - openvino_vision_*.xml / openvino_language_model.xml  -> a VLM export
  - openvino_encoder_model.xml + openvino_decoder_model.xml -> whisper-style
  - config.json model_type in the known VLM/whisper/embedding families
  - architectures ending in ForCausalLM (or a known text model_type) -> llm

The "unified" kind exists because newer families (Qwen3.5) ship ONE export
that is both a text LLM and a vision model: vision parts on disk, but text
generation works perfectly. Verified 2026-08-01 on the real
OpenVINO/Qwen3.5-0.8B-int4-ov: a text-only prompt answers in 1.3s. The
layout check alone called it "vlm", so `ovat chat` refused the very model
Intel's guidance recommends. A unified model answers to BOTH kind filters.

Discovery roots for find_models(): the OVAT_MODELS env var (os.pathsep-
separated folders) first, then ./models and ~/models. One level deep: model
folders live inside a models/ dir, not nested labyrinths.
"""
import json
import os

# model_type values seen in exported config.json files, by family.
_VLM_TYPES = {"qwen2_vl", "qwen2-vl", "qwen2_5_vl", "llava", "llava_next",
              "internvl_chat", "minicpmv", "phi3_v", "gemma3"}
_EMB_TYPES = {"bert", "roberta", "xlm-roberta", "distilbert", "mpnet",
              "nomic-bert", "new"}
_LLM_TYPES = {"llama", "qwen2", "qwen3", "mistral", "phi", "phi3", "gemma",
              "gemma2", "gpt2", "opt", "falcon", "stablelm", "tinyllama"}

# Families that ship ONE export serving as both a text LLM and a vision model.
# Kept as data, not scattered `if`s, so adding the next family is one line.
_UNIFIED_TYPES = {"qwen3_5", "qwen3_5_text", "qwen3_6"}

# Asking for "llm" must also surface a unified export, and so must "vlm":
# it genuinely is both, and the alternative is making users download two
# models to do what one already does.
_KIND_ALIASES = {"llm": {"llm", "unified"}, "vlm": {"vlm", "unified"}}


def _is_unified_multimodal(model_type: str, architectures: list[str]) -> bool:
    """Is this vision-layout export ALSO usable as a plain text LLM?

    Called only for folders that already look like a vision export, so the
    question is narrowly "is this one of the both-at-once families, or a
    vision-only model like Qwen2-VL?".

    Getting this wrong is expensive in both directions:
      - too eager  -> `ovat chat` accepts a vision-only model and the user
                      gets the C++ "Port for tensor name input_ids was not
                      found" traceback this module exists to prevent.
      - too strict -> the recommended model is refused as "not a text LLM".

    Evidence available, from the real Qwen3.5 config.json:
        model_type    = "qwen3_5"
        architectures = ["Qwen3_5ForConditionalGeneration"]
      versus a vision-only Qwen2-VL:
        model_type    = "qwen2_vl"
        architectures = ["Qwen2VLForConditionalGeneration"]
    Note that BOTH end in ForConditionalGeneration, so the architecture
    suffix alone cannot separate them.

    An EXACT membership test, deliberately, not a prefix or substring rule.
    "qwen3_5".startswith() style matching would also accept a future
    qwen3_5_vl, and a vision-only model wrongly accepted here is precisely
    the C++ traceback this module was written to stop. Exact matching fails
    the other way instead: an unlisted family is refused with a readable
    sentence, and adding it is one line in _UNIFIED_TYPES.

    architectures is accepted but not consulted: both families end in
    ForConditionalGeneration, so it cannot separate them. It stays in the
    signature because the next family may need it and callers already pass it.
    """
    return model_type in _UNIFIED_TYPES


def identify_model(path: str) -> tuple[str, str]:
    """Classify a model folder. Returns (kind, why).

    kind is one of: llm, vlm, whisper, embeddings, unknown, not-a-model.
    `why` is a short human phrase for error messages and listings.
    """
    if not os.path.isdir(path):
        return "not-a-model", f"no folder at {path}"
    try:
        files = set(os.listdir(path))
    except OSError as exc:
        return "not-a-model", f"unreadable: {exc}"

    if not any(f.endswith(".xml") and f.startswith("openvino") for f in files):
        return "not-a-model", "no openvino*.xml inside, not an exported model"

    # config.json is read BEFORE the layout verdict now: a vision layout is no
    # longer conclusive on its own, because a unified export wears the same one.
    model_type, architectures = "", []
    config_path = os.path.join(path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            model_type = str(config.get("model_type", "")).lower()
            architectures = [str(a) for a in config.get("architectures") or []]
        except (OSError, ValueError):
            pass

    # File layout: a VLM export has vision/language parts instead of a single
    # openvino_model.xml, and whisper exports encoder+decoder.
    if any(f.startswith("openvino_vision") for f in files) or \
            "openvino_language_model.xml" in files:
        if _is_unified_multimodal(model_type, architectures):
            return "unified", (f"unified multimodal export "
                               f"(model_type={model_type or 'unknown'}): "
                               f"text LLM and vision model in one")
        return "vlm", "vision-language export (vision/language model parts)"
    if "openvino_encoder_model.xml" in files and \
            "openvino_decoder_model.xml" in files:
        return "whisper", "encoder+decoder export (speech-to-text)"

    if model_type in _VLM_TYPES:
        return "vlm", f"config model_type={model_type} (vision-language)"
    if model_type == "whisper":
        return "whisper", "config model_type=whisper (speech-to-text)"
    if model_type in _EMB_TYPES:
        return "embeddings", f"config model_type={model_type} (embedder)"
    if any(a.endswith("ForCausalLM") for a in architectures) \
            or model_type in _LLM_TYPES:
        return "llm", f"config model_type={model_type or 'causal-lm'}"
    return "unknown", (f"could not classify (model_type="
                       f"{model_type or 'missing'})")


def _roots() -> list[str]:
    env = os.environ.get("OVAT_MODELS", "")
    roots = [p for p in env.split(os.pathsep) if p.strip()]
    roots += [os.path.join(os.getcwd(), "models"),
              os.path.expanduser("~/models")]
    return [os.path.expanduser(r) for r in roots]


# How far below a discovery root a model folder may sit.
#
# Two, not one, and this is the whole reason chat could not find a model on
# the AI PC. `ovms --pull` lays models out by ORG: the repository ends up as
# models/OpenVINO/Qwen3-8B-int4-ov, which `ovat models list` prints as
# "OpenVINO\Qwen3-8B-int4-ov". A one-level scan looked at models/OpenVINO,
# found no openvino*.xml in it, wrote it off as not-a-model and never went
# deeper, so the model sitting right there was reported as "no local text LLM
# found". Three levels would start crawling unrelated trees for no benefit.
_MAX_DEPTH = 2


def _walk_candidates(root: str, depth: int = _MAX_DEPTH):
    """Yield folders at or below `root`, at most `depth` levels down.

    A folder that IS a model is never descended into: an exported model can
    contain subfolders and there is nothing below it worth finding.
    """
    yield root
    if depth <= 0:
        return
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return                      # unreadable root: not fatal, just empty
    for entry in entries:
        child = os.path.join(root, entry)
        if not os.path.isdir(child):
            continue
        yield child
        if identify_model(child)[0] == "not-a-model":
            # Only an ORG-style container is worth opening. Recursing under a
            # real model would list its subfolders as siblings of itself.
            yield from _walk_candidates(child, depth - 1)


def find_models(kind: str | None = None) -> list[dict]:
    """Scan the discovery roots for model folders. Optionally filter by kind.

    Returns [{"name", "path", "kind", "why"}, ...], deduplicated, sorted by
    name so the output is stable between runs.
    """
    found: dict = {}
    for root in _roots():
        if not os.path.isdir(root):
            continue
        # The root itself may BE a model folder (OVAT_MODELS=.../Llama-3B),
        # which _walk_candidates yields first.
        for candidate in _walk_candidates(root):
            real = os.path.realpath(candidate)
            if real in found or not os.path.isdir(candidate):
                continue
            model_kind, why = identify_model(candidate)
            if model_kind == "not-a-model":
                continue
            found[real] = {"name": os.path.basename(candidate.rstrip(os.sep)),
                           "path": candidate, "kind": model_kind, "why": why}
    models = sorted(found.values(), key=lambda m: m["name"].lower())
    if kind:
        # A unified export satisfies both "llm" and "vlm"; see _KIND_ALIASES.
        wanted = _KIND_ALIASES.get(kind, {kind})
        models = [m for m in models if m["kind"] in wanted]
    return models


def searched_roots() -> list[str]:
    """The folders find_models() actually looks in, for error messages.

    Naming the real paths beats naming the env var: "scanned OVAT_MODELS,
    ./models, ~/models" left the user with no idea WHICH folders that came
    out to, and on Windows the answer is rarely what they assumed.
    """
    return list(_roots())


def pick_chat_llm() -> tuple[dict | None, list[dict]]:
    """Choose the best local text LLM for chat. Returns (choice, all_llms).

    Preference: an instruct/chat-tuned model over a base one (that is what a
    chat UI wants), otherwise simply the first found. The full list rides
    along so callers can show the alternatives.
    """
    llms = find_models("llm")
    if not llms:
        return None, []
    for model in llms:
        if "instruct" in model["name"].lower() or "chat" in model["name"].lower():
            return model, llms
    return llms[0], llms
