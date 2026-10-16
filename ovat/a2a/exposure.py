# ovat/a2a/exposure.py
"""Which of a workflow's tools a REMOTE caller may trigger.

Note to myself: a local user chose their own tools. A remote caller did
not, and can phrase a request to make the model call anything it is shown.
So a served agent is BUILT with the exposed tools only (build_agent's
select_tools); the rest are not just missing from the card, the model never
sees them.

Default rules, safest first:
- builtin tools that take a FILE PATH (transcribe, describe_image) are
  hidden: "transcribe C:\\Users\\me\\board-call.wav" would read any file.
- mcp_stdio tools are hidden: OVAT cannot know what a third-party server's
  tools touch.
- a2a tools and `delegate` (subagents) are hidden: a remote caller could
  otherwise make this machine send work on to a third agent or a cloud API.
- the coding tools (read_file, write_file, edit, grep, shell) are NEVER
  offered, not even through `expose_tools`: they read and change this
  machine's files and run commands, and no owner line should hand that to
  a remote caller.
- everything else (today: search_docs) is exposed.
`expose_tools` overrides all of this with an explicit list, which is a
line the owner wrote on purpose.
"""
from dataclasses import dataclass, field

from ovat.tools.project import CODING_TOOLS

# Builtins whose arguments name a file on this machine.
PATH_TOOLS = frozenset({"transcribe", "describe_image"})

_HIDDEN_BY_KIND = {
    "mcp_stdio": "an MCP tool; list it in a2a.server.expose_tools to offer it",
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
                  and n in CODING_TOOLS]
        if coding:
            raise ValueError(
                f"a2a.server.expose_tools names {', '.join(coding)}, which "
                f"can read or change files and run commands on this machine. "
                f"They are never offered to a remote caller; remove them "
                f"from the list.")
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
        if kind == "builtin" and name in CODING_TOOLS:
            result.hidden[name] = ("it works on this machine's files and "
                                   "shell, and is never offered to a "
                                   "remote caller")
        elif kind in _HIDDEN_BY_KIND:
            result.hidden[name] = _HIDDEN_BY_KIND[kind]
        elif name in PATH_TOOLS:
            result.hidden[name] = ("takes a file path; list it in "
                                   "a2a.server.expose_tools to offer it")
        else:
            result.exposed.append(name)
    return result
