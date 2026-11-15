# ovat/a2a/exposure.py
"""Tool exposure policy for remote A2A callers.

Unlike local workflows where a user selects tools directly, remote callers can
craft prompts to invoke any exposed tool. Served agents are therefore built
with only the exposed subset (via ``build_agent(..., select_tools=...)``) so
hidden tools are absent from both the Agent Card and the model's tool menu.

Default exposure rules (restrictive by default):
- File-path builtins (``transcribe``, ``describe_image``) are hidden so remote
  requests cannot reference arbitrary local files.
- ``mcp_stdio`` tools are hidden because third-party servers execute arbitrary
  local behavior.
- ``a2a`` tools and ``delegate`` (subagents) are hidden to prevent unapproved
  relaying to downstream agents or cloud APIs.
- Coding tools (``read_file``, ``write_file``, ``edit``, ``grep``, ``shell``),
  web tools (``web_fetch``, ``web_search``), and ``ask_user`` are never exposed
  remotely, even when listed in ``expose_tools``.
- Safe local tools (such as ``search_docs``) are exposed unless restricted.

Setting ``expose_tools`` explicitly overrides the default deny rules (except
for permanently blocked local/interactive tools).
"""
from dataclasses import dataclass, field

from ovat.tools.project import NEVER_REMOTE

# Builtins whose arguments name a file on this machine.
PATH_TOOLS = frozenset({"transcribe", "describe_image"})

_HIDDEN_BY_KIND = {
    "mcp_stdio": "an MCP tool; list it in a2a.server.expose_tools to offer it",
    "mcp_http": "an MCP tool; list it in a2a.server.expose_tools to offer it",
    "mcp_sse": "an MCP tool; list it in a2a.server.expose_tools to offer it",
    "a2a": "it would let a caller send messages on to another agent",
    "subagent": "it would let a caller hand tasks on to this workflow's "
                "subagents",
}


@dataclass
class Exposure:
    exposed: list[str] = field(default_factory=list)
    hidden: dict[str, str] = field(default_factory=dict)   # name -> why
    warnings: list[str] = field(default_factory=list)

    def select(self, tools: dict) -> dict:
        """The subset of `tools` a served agent is built with."""
        return {n: spec for n, spec in tools.items() if n in self.exposed}


def decide(tools: dict, expose_tools: list[str] | None) -> Exposure:
    """Split built `tools` ({name: {schema, function, kind}}) in two."""
    result = Exposure()
    if expose_tools is not None:
        coding = [n for n in expose_tools if n in tools
                  and tools[n].get("kind", "builtin") == "builtin"
                  and n in NEVER_REMOTE]
        if coding:
            raise ValueError(
                f"a2a.server.expose_tools names {', '.join(coding)}, which "
                f"act on this machine (files, commands, web requests or its "
                f"user). They are never offered to a remote caller; remove "
                f"them from the list.")
        unknown = [n for n in expose_tools if n not in tools]
        if unknown:
            raise ValueError(
                f"a2a.server.expose_tools names {', '.join(unknown)}, which "
                f"this workflow does not have. Its tools: "
                f"{', '.join(tools) or 'none'}.")
        for name in tools:
            if name in expose_tools:
                result.exposed.append(name)
                if name in PATH_TOOLS:
                    result.warnings.append(
                        f"{name} is exposed and takes a file path: a caller "
                        f"can name any file this user can read.")
            else:
                result.hidden[name] = "not listed in a2a.server.expose_tools"
        return result

    for name, spec in tools.items():
        kind = spec.get("kind", "builtin")
        if kind == "builtin" and name in NEVER_REMOTE:
            result.hidden[name] = (f"{NEVER_REMOTE[name]}, and is never "
                                   f"offered to a remote caller")
        elif kind in _HIDDEN_BY_KIND:
            result.hidden[name] = _HIDDEN_BY_KIND[kind]
        elif name in PATH_TOOLS:
            result.hidden[name] = ("takes a file path; list it in "
                                   "a2a.server.expose_tools to offer it")
        else:
            result.exposed.append(name)
    return result
