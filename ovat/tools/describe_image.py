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
_provider = None
VLM_MODEL_DIR = os.environ.get("OVAT_VLM_MODEL", "models/Qwen2-VL-2B-Instruct-INT4")

mcp = FastMCP("describe_image")


def _load_provider():
    global _provider
    if _provider is None:
        # Imported here so the module loads on machines without the model.
        from ovat.providers.vlm_genai import GenAIVLMProvider
        _provider = GenAIVLMProvider(VLM_MODEL_DIR, "CPU")
    return _provider


def describe_image_impl(image_path: str,
                        prompt: str = "Describe this image in one paragraph.",
                        provider=None) -> str:
    """The real logic, separate from the MCP wrapper so tests can fake the VLM.

    Errors come back as readable strings, not exceptions; the agent loop
    hands them to the model, which can correct its call on the next turn.
    """
    if not os.path.isfile(image_path):
        return f"Error: I could not find an image file at: {image_path}"
    if provider is None:
        try:
            provider = _load_provider()
        except Exception as exc:
            return (f"Error: could not load the vision model at "
                    f"{VLM_MODEL_DIR}: {exc}. Set OVAT_VLM_MODEL to a local "
                    f"OpenVINO VLM folder.")
    return str(provider.generate(prompt, [image_path]))


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
    return describe_image_impl(image_path, prompt, _provider)


if __name__ == "__main__":
    # Standalone MCP server mode, like search_docs and transcribe.
    mcp.run()
