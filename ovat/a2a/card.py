# ovat/a2a/card.py
"""The Agent Card: how this agent introduces itself, as plain JSON.

Built as a dict in the spec's own JSON shape (camelCase, section 4.4 and
a2a.proto `AgentCard`), not as an SDK object. That keeps it testable
without the extra, printable by `ovat a2a card`, and independent of the
SDK's class names. sdk/server.py turns the dict into the SDK's type.

The card is PUBLIC: anyone who can reach the port can read it without a
token (the spec requires that). So it never contains the system prompt,
which can hold private instructions, or any path on this machine.
"""
import re

from ovat.a2a.protocol import SPEC_VERSION
from ovat.config.workflow import A2AServerConfig, WorkflowConfig

# Longest skill description taken from a tool's schema. The card is read by
# other agents' models, so it is kept short.
_MAX_SKILL_TEXT = 300


def _tool_skill(name: str, spec: dict) -> dict:
    text = spec["schema"]["function"].get("description") or name
    return {"id": name, "name": name,
            "description": text[:_MAX_SKILL_TEXT],
            "tags": ["ovat", spec.get("kind", "builtin")]}


def _public_model(name: str) -> str:
    """The model's name without any folder: a local model is configured by
    path, and the card is readable without a token."""
    return re.split(r"[\\/]", str(name).rstrip("\\/"))[-1] or "a local model"


def card_dict(config: WorkflowConfig, server: A2AServerConfig, tools: dict,
              url: str, default_name: str, version: str) -> dict:
    """The card for a served workflow. `tools` are the EXPOSED ones only."""
    name = server.name or default_name
    model = _public_model(config.model.name)
    if server.description:
        description = server.description
    else:
        offered = ", ".join(tools) or "no tools"
        description = (f"OVAT agent running {model} on "
                       f"{config.model.device}, on this machine. "
                       f"Tools: {offered}.")
    if server.skills:
        skills = [s.model_dump() for s in server.skills]
    elif tools:
        skills = [_tool_skill(n, spec) for n, spec in tools.items()]
    else:
        skills = [{"id": "chat", "name": "chat",
                   "description": f"Answers questions using {model}.",
                   "tags": ["ovat"]}]
    return {
        "name": name,
        "description": description,
        "version": version,
        "supportedInterfaces": [
            {"protocolBinding": "JSONRPC", "protocolVersion": SPEC_VERSION,
             "url": url},
        ],
        "capabilities": {"streaming": True, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "securitySchemes": {
            "bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "securityRequirements": [{"schemes": {"bearer": {}}}],
        "skills": skills,
    }
