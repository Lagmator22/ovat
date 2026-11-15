# ovat/tui/agent_commands.py
"""Slash-command registry for the agent TUI.

Combines OVAT's built-in commands with the terminal-client commands supported by
OVAT's JSON-RPC app server. Every client command is mapped to its required
app-server routes (or unsupported capability markers) so unreachable actions are
filtered cleanly instead of failing at runtime.

Commands overlapping OVAT's table (`help`, `new`/`clear`, `copy`, `quit`/`exit`)
are unified so names and aliases dispatch to a single handler.
"""
from ovat.agentic import commands as ovat_commands
from ovat.tui.vibe.cli.commands import Command, CommandRegistry

# Required server routes per client command ("capability:..." marks features
# intentionally omitted from OVAT's local runtime).
CLIENT_COMMAND_NEEDS: dict[str, tuple[str, ...]] = {
    "help": (),
    "config": ("config/fields/read", "config/write"),
    "model": ("config/fields/read", "config/model/write"),
    "skills": ("skills/installed",),
    "thinking": ("config/model/write",),
    "reload": ("config/reload",),
    "clear": ("session/history/clear",),
    "copy": (),
    "paste-image": (),
    "log": ("session/log/read",),
    # Requires a client-side log_level field unused by OVAT.
    "log-level": ("capability:a log level setting",),
    "debug": ("diagnostics/logs/read",),
    "compact": ("session/compact",),
    "exit": ("session/stop",),
    "status": ("stats/read",),
    "proxy-setup": ("config/proxy/read", "config/proxy/write"),
    "resume": ("session/list", "session/read", "session/resume",
               "session/delete"),
    "rename": ("session/rename",),
    "mcp": ("mcp_catalog/read", "mcp_catalog/toggle", "mcp_catalog/refresh",
            "connector_catalog/read"),
    # Reads the plan from the todo tool's TODO cards, client side.
    "todo": (),
    "voice": (),
    "rewind": ("session/rewind/read", "session/rewind"),
    "branch": ("session/fork",),
    "retry": ("capability:continuing a reply",),
    "loop": ("loops/list",),
    "theme": (),
}

CAPABILITY = "capability:"

# OVAT command -> the client command that is the same thing.
SHARED_COMMANDS = {"help": "help", "new": "clear", "copy": "copy", "quit": "exit"}

# Upstream client descriptions rewritten for OVAT behavior, plus unused aliases.
OWN_WORDS = {
    "voice": "Turn local voice dictation on or off. Ctrl+R records; the "
             "audio stays on this machine",
    "reload": "Read the workflow file again and rebuild the agent if it "
              "changed",
    "mcp": "The workflow's MCP servers and their tools. Pass a name to list "
           "its tools; switch one on or off for this session",
    "log": "Show the folder where conversations are saved",
}
DROPPED_ALIASES = {"mcp": {"/connectors"}}

# Read-only OVAT commands that may run while the agent works.
SIDE_CHANNEL = frozenset({"tools", "agents", "trace", "yolo", "serve",
                          "doctor", "telemetry"})

KEYS = """### Keys

- `Enter` send, `Ctrl+J` new line, `Up/Down` history
- `!cmd` runs a shell command here
- `Esc` stops the answer or the command
- `Esc Esc` on an empty prompt goes back to an earlier message (rewind)
- `Ctrl+O` (or `Ctrl+T`) opens or folds the tool calls
- `Ctrl+R` records local voice until you press it again or 10 seconds pass in quiet, `Esc` cancels
- `Shift+Tab` switches between asking and auto-approve
- `Ctrl+C` copies a selection, stops the answer, and quits when pressed twice
"""


def served_methods() -> frozenset[str]:
    """Every JSON-RPC method OVAT's app server answers."""
    from ovat.tui.server.methods import all_routes
    return frozenset(all_routes().names())


def client_command_works(name: str, served: frozenset[str]) -> bool:
    """True when every method the client command needs is served. A command
    this table does not know is NOT kept."""
    needs = CLIENT_COMMAND_NEEDS.get(name)
    return needs is not None and all(m in served for m in needs)


class OvatCommandRegistry(CommandRegistry):
    """The copied registry, narrowed to what works here and widened with
    OVAT's commands."""

    def _build_commands(self) -> dict[str, Command]:
        served = served_methods()
        commands = {name: command
                    for name, command in super()._build_commands().items()
                    if client_command_works(name, served)}
        # OVAT always enables the todo tool directly in workflow sessions,
        # without requiring an experimental harness toggle.
        if "todo" in commands:
            commands["todo"].is_available = None
        for name, words in OWN_WORDS.items():
            if name in commands:
                commands[name].description = words
        for name, aliases in DROPPED_ALIASES.items():
            if name in commands:
                commands[name].aliases = commands[name].aliases - aliases
        for c in ovat_commands.COMMANDS:
            aliases = frozenset({f"/{c.name}", *(f"/{a}" for a in c.aliases
                                                 if a.isalnum())})
            shared_name = SHARED_COMMANDS.get(c.name)
            if shared_name is not None:
                if shared_name in commands:
                    old = commands[shared_name]
                    old.aliases = old.aliases | aliases
                continue
            commands[c.name] = Command(
                aliases=aliases,
                description=c.help + (f"  ({c.usage})" if c.usage else ""),
                handler=f"_cmd_{c.name}",
                side_channel=c.name in SIDE_CHANNEL)
        return commands

    def get_help_text(self) -> str:
        lines = [KEYS, "### Commands", ""]
        for name, cmd in sorted(self.commands.items()):
            canonical = f"/{name}"
            aliases = ", ".join(
                f"`{alias}`" for alias in sorted(
                    (a for a in cmd.aliases if a.startswith("/")),
                    key=lambda a: (a != canonical, a)))
            lines.append(f"- {aliases}: {_escape_tags(cmd.description)}")
        return "\n".join(lines)


def _escape_tags(text: str) -> str:
    """A usage like <engine> would be read as an HTML tag and vanish, so it
    is escaped; but not inside a `code span`, where Markdown shows the
    escape itself ("--resume &lt;id&gt;")."""
    parts = text.split("`")
    for i in range(0, len(parts), 2):      # the even parts are outside code
        parts[i] = parts[i].replace("<", "&lt;").replace(">", "&gt;")
    return "`".join(parts)
