# ovat/core/model_scout.py
"""Find and identify OpenVINO models on this machine.

Why this exists: a new user pointed `chat` at a Qwen2-VL folder (a VISION
model) and got a C++ traceback about a missing `input_ids` tensor port.
Nothing told them "that's not a text LLM". Every exported OpenVINO model
folder carries the evidence of what it is; this module reads it so commands
can refuse politely, suggest the right folder, or pick one automatically.

Identification signals (checked in this order: file layout first because it
never lies, then config.json):
  - openvino_vision_*.xml / openvino_language_model.xml  -> a VLM export
  - openvino_encoder_model.xml + openvino_decoder_model.xml -> whisper-style
  - config.json model_type in the known VLM/whisper/embedding families
  - architectures ending in ForCausalLM (or a known text model_type) -> llm

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

    # File layout first: a VLM export has vision/language parts instead of a
    # single openvino_model.xml, and whisper exports encoder+decoder.
    if any(f.startswith("openvino_vision") for f in files) or \
            "openvino_language_model.xml" in files:
        return "vlm", "vision-language export (vision/language model parts)"
    if "openvino_encoder_model.xml" in files and \
            "openvino_decoder_model.xml" in files:
        return "whisper", "encoder+decoder export (speech-to-text)"

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


def find_models(kind: str | None = None) -> list[dict]:
    """Scan the discovery roots for model folders. Optionally filter by kind.

    Returns [{"name", "path", "kind", "why"}, ...], deduplicated, sorted by
    name so the output is stable between runs.
    """
    found: dict = {}
    for root in _roots():
        if not os.path.isdir(root):
            continue
        # The root itself may BE a model folder (OVAT_MODELS=.../Llama-3B).
        candidates = [root] + [os.path.join(root, entry)
                               for entry in sorted(os.listdir(root))]
        for candidate in candidates:
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
        models = [m for m in models if m["kind"] == kind]
    return models


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
