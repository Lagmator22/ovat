# ovat/tui/agent_text.py
"""What the agent TUI says, as plain values both of its apps share.

There are two apps over the same AgentSession: the classic one
(ovat/tui/app.py) and the one built on the copied Vibe app
(ovat/tui/agent_app.py). Their commands print the same tables and lines, so
the words live here once and each app only decides where to draw them.

Functions return markdown text or a rich Text. A save or a load returns
`(text, level)` so the caller can show a warning in the right style.
Nothing in this file imports the UI toolkit's widgets.
"""
import time
from pathlib import Path

from rich.text import Text

from ovat.agentic import commands as ovat_commands
from ovat.agentic.sessions import sessions_dir
from ovat.cli import ui

ENGINES = ("native", "react", "llamaindex", "openai-agents")


def short_path(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home) else path


# ---- tables for /tools, /agents, /trace -----------------------------------

def tools_markdown(info: dict) -> str:
    tools = info.get("tools", [])
    if not tools:
        return "This agent has no tools."
    lines = ["| tool | kind | permission |", "| --- | --- | --- |"]
    lines += [f"| `{t['name']}` | {t['kind']} | {t['permission']} |"
              for t in tools]
    return "\n".join(lines)


def agents_markdown(info: dict, serving: bool) -> str:
    lines = ["### Subagents", ""]
    subs = info.get("subagents", [])
    lines += ([f"- `{x['name']}` ({x['where']}): {x['description']}"
               for x in subs] or ["- none (add `subagents:` to the workflow)"])
    peers = [t for t in info.get("tools", []) if t["kind"] == "a2a"]
    lines += ["", "### A2A agents used as tools", ""]
    lines += ([f"- `{p['name']}` ({p['permission']})" for p in peers]
              or ["- none"])
    lines += ["", "### This workflow over A2A", "",
              "- serving (`/serve stop` ends it)" if serving
              else "- not serving (`/serve` starts it)"]
    return "\n".join(lines)


def trace_markdown(trace: dict) -> str:
    if not trace:
        return "Nothing yet: ask something first."
    totals = trace.get("totals", {})
    flags = [k for k in ("failed", "truncated", "undecoded_tool_call",
                         "empty_answer") if totals.get(k)]
    lines = [f"**{trace.get('engine')}**: {totals.get('turns')} turns, "
             f"{totals.get('tool_calls')} tool calls, "
             f"{totals.get('latency_s')} s"
             + (f", **{', '.join(flags)}**" if flags else ""), ""]
    for i, turn in enumerate(trace.get("turns", []) or [], 1):
        lines.append(f"- turn {i}: {turn.get('latency_s')} s, "
                     f"{turn.get('prompt_tokens')} → "
                     f"{turn.get('completion_tokens')} tokens, "
                     f"{turn.get('finish_reason')}")
        for call in turn.get("tool_calls", []):
            lines.append(f"  - `{call.get('name')}` "
                         f"{call.get('duration_s')} s")
    return "\n".join(lines)


def commands_markdown() -> list[str]:
    """The `/help` command list (classic app)."""
    lines = []
    for c in ovat_commands.COMMANDS:
        usage = f" {c.usage}" if c.usage else ""
        lines.append(f"- `/{c.name}{usage}`: {c.help}")
    return lines


# ---- the bottom bar --------------------------------------------------------

def facts_text(info: dict, totals: dict, serving: bool,
               reading: dict) -> Text:
    """The agent's facts for the bottom bar: device, model, engine, tokens
    used, whether A2A is serving, and the live hardware reading."""
    facts = Text(no_wrap=True, overflow="ellipsis")
    if info.get("model"):
        facts.append(info["device"], style=f"bold {ui.CYAN}")
        facts.append(f" {info['model']} · {info.get('engine')}")
    used = totals.get("prompt_tokens", 0) + totals.get("completion_tokens", 0)
    if used:
        facts.append(f" · {used:,} tokens")
    if serving:
        facts.append(" · ")
        facts.append("a2a serving", style=ui.GREEN)
    if reading:
        from ovat.agentic.hardware import format_reading
        facts.append(" · " + format_reading(reading).replace("  ·  ", " · "))
    return facts


# ---- /yolo, /save, /load ---------------------------------------------------

def yolo_text(on: bool) -> tuple[str, str]:
    if on:
        return ("Every tool is approved for this session. `/yolo` again "
                "turns it off.", "warn")
    return "Tools ask again.", "info"


def save_conversation(agent, cwd: str, name: str) -> tuple[str, str]:
    memory = getattr(agent, "session", None)
    if memory is None:
        return "Saving works on the native engine.", "warn"
    name = name.strip() or time.strftime("agent-%Y%m%d-%H%M%S")
    path = sessions_dir(cwd) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    memory.save(str(path))
    return f"Saved to `{short_path(str(path))}`", "info"


def load_conversation(agent_session, cwd: str, name: str) -> tuple[str, str]:
    """Put a saved conversation into the agent's memory (native engine)."""
    from ovat.agent.session import Session
    name = name.strip()
    path = sessions_dir(cwd) / f"{name}.json"
    if not name or not path.is_file():
        saved = sorted(p.stem for p in sessions_dir(cwd).glob("*.json"))
        return ("Usage: `/load <name>`. Saved: "
                + (", ".join(saved[-8:]) or "none"), "warn")
    if getattr(agent_session.agent, "session", None) is None:
        return "Loading works on the native engine.", "warn"
    agent_session.agent.session = Session.load(str(path))
    return (f"Loaded `{name}`. The model remembers it; the transcript above "
            f"is not redrawn.", "info")
