# ovat/tools/describe_image.py
"""The describe_image tool: an image path in, a text description out.

This is what finally puts GenAIVLMProvider (Qwen2-VL) on a path a user can
reach: declare the tool in workflow.yml and the agent can look at pictures.
Same design as transcribe.py: the real logic in a plain testable function,
a lazily-cached heavy pipeline, a co-located SCHEMA that is the complete
contract (the LangChain path derives its argument model from it), and a thin
FastMCP wrapper so it can also serve any MCP-aware agent standalone.
"""
import os

from fastmcp import FastMCP

# The VLM is heavy (a ~2 GB pipeline); build it once, on first use, and only
# if this machine actually has the model. The env var mirrors transcribe's.
# Cached BY (model, device), not in one slot -- see transcribe.py for why a
# singleton stops being correct the moment workflow.yml can name the model.
_providers: dict = {}

#: Fallback when nothing is configured.
DEFAULT_MODEL_DIR = "models/Qwen2-VL-2B-Instruct-INT4"


def _configured_model(model: str | None = None) -> str:
    """Where the vision model lives: config, then env var, then the default.

    Read here rather than at import, so a late env var still works and the
    config can win over it.
    """
    return model or os.environ.get("OVAT_VLM_MODEL") or DEFAULT_MODEL_DIR

mcp = FastMCP("describe_image")


def _resolve_device(device: str | None = None) -> str:
    """Config, then env var, then DeviceManager's LLM advice, then CPU.

    A VLM is heavy, so it follows the LLM recommendation (GPU when present)
    rather than whisper's CPU one.
    """
    chosen = device or os.environ.get("OVAT_VLM_DEVICE")
    if chosen:
        return chosen
    try:
        from ovat.core.device_manager import DeviceManager
        return DeviceManager().get_llm_device()
    except Exception:
        return "CPU"


def _load_provider(model: str | None = None, device: str | None = None):
    key = (_configured_model(model), _resolve_device(device))
    if key not in _providers:
        if not os.path.isdir(key[0]):
            # Same guard, and the same reason, as transcribe._load_pipeline:
            # openvino_genai answers a missing folder with a C++ assertion
            # naming openvino_language_model.xml, which buries the one
            # sentence the caller already appends about setting `model:`.
            raise FileNotFoundError("the folder does not exist")
        # Imported here so the module loads on machines without the model.
        from ovat.providers.vlm_genai import GenAIVLMProvider
        _providers[key] = GenAIVLMProvider(*key)
    return _providers[key]


def describe_image_impl(image_path: str,
                        prompt: str = "Describe this image in one paragraph.",
                        provider=None, model: str | None = None,
                        device: str | None = None) -> str:
    """The real logic, separate from the MCP wrapper so tests can fake the VLM.

    Errors come back as readable strings, not exceptions; the agent loop
    hands them to the model, which can correct its call on the next turn.
    """
    from ovat.tools.fuzzy import resolve_path
    image_path = resolve_path(image_path)
    if not os.path.isfile(image_path):
        return f"Error: I could not find an image file at: {image_path}"
    if provider is None:
        try:
            provider = _load_provider(model, device)
        except Exception as exc:
            return (f"Error: could not load the vision model at "
                    f"{_configured_model(model)}: {exc}. Set the tool's "
                    f"`model:` in workflow.yml to a local OpenVINO VLM "
                    f"folder.")
    try:
        return str(provider.generate(prompt, [image_path]))
    except Exception as exc:
        return f"Error analyzing image: {exc}"


# The OpenAI-style schema the agent's menu shows. Co-located and complete
# (defaults included): the single source of truth for both engines.
SCHEMA = {
    "type": "function",
    "function": {
        "name": "describe_image",
        "description": "Look at an image file and describe what it shows. Use "
                       "when the user gives a path to a picture and asks about "
                       "its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string",
                               "description": "path to an image file (jpg/png)"},
                "prompt": {"type": "string",
                           "description": "what to ask about the image",
                           "default": "Describe this image in one paragraph."},
            },
            "required": ["image_path"],
        },
    },
}


@mcp.tool
def describe_image(image_path: str,
                   prompt: str = "Describe this image in one paragraph.") -> str:
    """Look at an image file and describe what it shows.

    Use me when the user gives a path to a picture and wants to know what is
    in it. I take the file path (and optionally a specific question) and
    return the description as text.
    """
    return describe_image_impl(image_path, prompt)


if __name__ == "__main__":
    # Standalone MCP server mode, like search_docs and transcribe.
    mcp.run()
