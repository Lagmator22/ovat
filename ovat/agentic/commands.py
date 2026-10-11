# ovat/agentic/commands.py
"""Slash commands: one registry, a parser and a fuzzy matcher.

One table is the single source of truth for what exists: the completion
popup, /help, and the dispatcher all read it, so a command cannot appear in
one and be missing from another. The front end maps a command NAME to a
handler; nothing here knows about the UI.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    usage: str = ""
    aliases: tuple[str, ...] = ()


COMMANDS: tuple[Command, ...] = (
    Command("help", "Show commands and keys", aliases=("?",)),
    Command("new", "Start a new conversation (same workflow)",
            aliases=("clear", "reset")),
    Command("tools", "List this agent's tools and their permissions"),
    Command("agents", "Subagents, A2A peers and the local A2A server"),
    Command("ask", "Send a task straight to one subagent",
            usage="<subagent> <task>"),
    Command("serve", "Offer this workflow to other agents over A2A",
            usage="[stop]"),
    Command("engine", "Switch engine: native, react, llamaindex, "
                      "openai-agents", usage="<engine>"),
    Command("trace", "What the last answer did: turns, tools, tokens"),
    Command("yolo", "Approve every tool for this session (again: off)"),
    Command("workflow", "Load another workflow file", usage="<path.yml>"),
    Command("save", "Save the conversation", usage="[name]"),
    Command("load", "Load a saved conversation", usage="<name>"),
    Command("copy", "Copy the last answer to the clipboard"),
    Command("doctor", "Check this machine, OVMS and the workflow"),
    Command("telemetry", "Live CPU, GPU, NPU and power"),
    Command("quit", "Leave", aliases=("exit", "q")),
)

_BY_NAME = {c.name: c for c in COMMANDS}
for _c in COMMANDS:
    for _a in _c.aliases:
        _BY_NAME[_a] = _c


def parse(text: str) -> tuple[Command | None, str]:
    """"/ask researcher find X" -> (ask, "researcher find X").

    (None, "") for text that is not a slash command, and (None, name) for
    one that does not exist, so the caller can say which.
    """
    if not text.startswith("/"):
        return None, ""
    head, _, rest = text[1:].partition(" ")
    command = _BY_NAME.get(head.lower())
    if command is None:
        return None, head
    return command, rest.strip()


def _subsequence(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


def complete(prefix: str, limit: int = 8) -> list[Command]:
    """Commands matching what was typed after "/": prefix matches first,
    then names containing the letters in order ("tl" finds "tools")."""
    needle = prefix.lstrip("/").split(" ")[0].lower()
    if not needle:
        return list(COMMANDS[:limit])
    starts = [c for c in COMMANDS
              if c.name.startswith(needle)
              or any(a.startswith(needle) for a in c.aliases)]
    fuzzy = [c for c in COMMANDS
             if c not in starts and _subsequence(needle, c.name)]
    return (starts + fuzzy)[:limit]
