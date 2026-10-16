# ovat/tui/agent_commands.py
"""The slash commands of the new agent app: OVAT's, plus the Vibe ones that
work on OVAT's app server.

The copied registry (ovat/tui/vibe/cli/commands.py) lists 34 commands. Most
of them need something OVAT's server does not have (a model picker, MCP,
an account...), and a command that cannot work must not
be reachable: it would only end in a "not supported" error. So the table
below says, for EVERY Vibe command, which server methods its handler calls
(or which capability it assumes), and the registry keeps a command only when
all of those are routes the server answers. Nothing is a hand-kept "removed"
list: when the server learns `session/rename`, `/rename` is kept at the next
start, and a Vibe update that adds a command must be classified here or a
test fails.

OVAT's own commands come from ovat/agentic/commands.py, the one table that
the classic app and the completion popup use too. Four of them are the same
thing as a Vibe command and keep Vibe's handler: help, new (Vibe's /clear),
copy and quit (Vibe's /exit). Their names and aliases are merged.
"""
from ovat.agentic import commands as ovat_commands
from ovat.tui.vibe.cli.commands import Command, CommandRegistry

# What each Vibe command's handler asks of the server. A name with a slash is
# a JSON-RPC method; "capability:..." is something OVAT does not have at all.
VIBE_NEEDS: dict[str, tuple[str, ...]] = {
    "help": (),
    "config": ("config/fields/read", "config/write"),
    "model": ("config/fields/read", "config/model/write"),
    "skills": ("skills/installed",),
    "thinking": ("config/model/write",),
    "reload": ("config/reload",),
    "clear": ("session/history/clear",),
    "copy": (),
    "paste-image": ("capability:image input",),
    "log": ("session/log/read",),
    # It writes Vibe's log_level, which OVAT does not use: config/write
    # answers "not used by OVAT", so the command would only end in that.
    "log-level": ("capability:a log level setting",),
    "debug": ("diagnostics/logs/read",),
    "compact": ("session/compact",),
    "exit": ("session/stop",),
    "status": ("stats/read",),
    "whoami": ("capability:an account",),
    "teleport": ("vibeCode/teleport/start",),
    "remote-project": ("vibeCode/projects/open",),
    "proxy-setup": ("config/proxy/read", "config/proxy/write"),
    "resume": ("session/list", "session/read", "session/resume",
               "session/delete"),
    "rename": ("session/rename",),
    "mcp": ("mcp_catalog/read", "mcp_catalog/toggle", "mcp_catalog/refresh",
            "connector_catalog/read"),
    "plugins": ("plugin_catalog/read",),
    "reload-plugins": ("plugin/reload",),
    "todo": ("capability:a todo tool",),
    # The voice panel writes voice_mode_enabled, also refused (voice comes
    # with its own step).
    "voice": ("capability:voice input",),
    "leanstall": ("agents/install",),
    "unleanstall": ("agents/uninstall",),
    "rewind": ("session/rewind/read", "session/rewind"),
    "branch": ("session/fork",),
    "retry": ("capability:continuing a reply",),
    "loop": ("loops/list",),
    "data-retention": ("capability:a cloud account",),
    "theme": (),
}

CAPABILITY = "capability:"

# OVAT command -> the Vibe command that is the same thing.
SAME_AS_VIBE = {"help": "help", "new": "clear", "copy": "copy", "quit": "exit"}

# Read-only OVAT commands that may run while the agent works.
SIDE_CHANNEL = frozenset({"tools", "agents", "trace", "yolo", "serve",
                          "doctor", "telemetry"})

KEYS = """### Keys

- `Enter` send, `Ctrl+J` new line, `Up/Down` history
- `!cmd` runs a shell command here
- `Esc` stops the answer or the command
- `Esc Esc` on an empty prompt goes back to an earlier message (rewind)
- `Ctrl+O` (or `Ctrl+T`) opens or folds the tool calls
- `Shift+Tab` switches between asking and auto-approve
- `Ctrl+C` copies a selection, stops the answer, and quits when pressed twice
"""


def served_methods() -> frozenset[str]:
    """Every JSON-RPC method OVAT's app server answers."""
    from ovat.tui.server.methods import all_routes
    return frozenset(all_routes().names())


def vibe_command_works(name: str, served: frozenset[str]) -> bool:
    """True when every method the Vibe command needs is served. A command
    this table does not know is NOT kept."""
    needs = VIBE_NEEDS.get(name)
    return needs is not None and all(m in served for m in needs)


class OvatCommandRegistry(CommandRegistry):
    """The copied registry, narrowed to what works here and widened with
    OVAT's commands."""

    def _build_commands(self) -> dict[str, Command]:
        served = served_methods()
        commands = {name: command
                    for name, command in super()._build_commands().items()
                    if vibe_command_works(name, served)}
        for c in ovat_commands.COMMANDS:
            aliases = frozenset({f"/{c.name}", *(f"/{a}" for a in c.aliases
                                                 if a.isalnum())})
            vibe_name = SAME_AS_VIBE.get(c.name)
            if vibe_name is not None:
                if vibe_name in commands:
                    old = commands[vibe_name]
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
