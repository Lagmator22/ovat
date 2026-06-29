# ovat/cli/commands.py
"""The shared command layer behind the slash commands in the TUI.

Why this file exists: the TUI must be wired to the real toolkit, not a demo
mock. So every slash command resolves to a handler here that runs the actual
logic (the same diagnostics, the same config loader, the same indexer the CLI
uses) and returns something to display. Keeping this layer free of any Textual
import means I can unit test each command with no terminal at all, which is how
I prove the TUI is genuinely wired.

Adding a command later is a one-liner: write a handler and register it in
COMMANDS. That is the "simple now, expandable later" shape.
"""
import shlex
from dataclasses import dataclass
from typing import Callable

from rich.table import Table
from rich.text import Text

from ovat.cli import ui

# The starter workflow written by /init (and by `ovat init`). It lives here so
# the CLI command and the TUI command share one template.
STARTER_YAML = """\
# OVAT workflow. Edit this, then run:  ovat run workflow.yml --input "..."
model:
  name: Qwen3-8B-int4-ov
  device: GPU
  ovms_url: http://localhost:8000/v3
  tool_parser: hermes3
  # Only used by `ovat serve` to start OVMS and locate the model:
  source_model: OpenVINO/Qwen3-8B-int4-ov
  model_repository_path: models

tools:
  - name: search_docs
    type: builtin
  - name: transcribe
    type: builtin

agent:
  type: native
  max_iterations: 10
  system_prompt: "You are a helpful assistant that uses tools when needed."

# RAG for the search_docs tool. Run `ovat index <folder> workflow.yml` first,
# then ask questions. Swap a provider string to change a backend.
rag:
  embeddings:
    provider: genai
    model: models/bge-small-en-v1.5
    device: CPU
    dim: 384
  retriever:
    provider: sqlite-vec
    db_path: ovat_index.db
  chunk:
    size: 512
    overlap: 64
"""


@dataclass
class CommandResult:
    """What a handler hands back: something to show, and an optional UI action."""

    renderable: object              # anything a rich console / RichLog can render
    action: str | None = None       # None, "clear", or "exit"


@dataclass
class SlashCommand:
    """One slash command: its name, how to call it, and the code it runs."""

    name: str
    usage: str
    description: str
    handler: Callable[[list], CommandResult]


def _ok(msg: str) -> CommandResult:
    return CommandResult(Text(msg, style=ui.GREEN))


def _info(msg: str) -> CommandResult:
    return CommandResult(Text(msg, style=ui.CYAN))


def _error(msg: str) -> CommandResult:
    return CommandResult(Text(msg, style=ui.RED))


def render_checks_table(checks) -> Table:
    """Render doctor checks as a coloured table, using hex so it works anywhere.

    I avoid theme-named styles here on purpose: the TUI's log widget does not
    carry my rich Theme, so a named style like 'ovat.ok' would fail to resolve.
    Plain hex renders the same in the themed CLI and in the TUI.
    """
    table = Table(header_style=f"bold {ui.BLUE}", border_style=ui.DIM, expand=False)
    table.add_column("Check", style=ui.CYAN, no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail", style=ui.DIM)
    for c in checks:
        glyph, color = ui.STATUS_HEX.get(c.status, ("?", ui.DIM))
        table.add_row(c.name, Text(f"{glyph} {c.status}", style=color), c.detail)
    return table


# Handlers. Each takes the parsed args (a list of strings) and returns a result.

def cmd_help(args: list) -> CommandResult:
    table = Table(title="OVAT commands", header_style=f"bold {ui.BLUE}",
                  border_style=ui.DIM, title_style=f"bold {ui.CYAN}")
    table.add_column("Command", style=ui.PURPLE, no_wrap=True)
    table.add_column("What it does", style=ui.DIM)
    for cmd in COMMANDS.values():
        table.add_row(cmd.usage, cmd.description)
    return CommandResult(table)


def cmd_doctor(args: list) -> CommandResult:
    from ovat.cli import diagnostics
    config = args[0] if args else None
    return CommandResult(render_checks_table(diagnostics.run_checks(config)))


def cmd_init(args: list) -> CommandResult:
    import os
    path = args[0] if args else "workflow.yml"
    if os.path.exists(path):
        return _error(f"Refusing to overwrite an existing file: {path}")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(STARTER_YAML)
    except OSError as exc:
        return _error(f"Could not write {path}: {exc}")
    return _ok(f"Wrote a starter workflow to {path}")


def cmd_validate(args: list) -> CommandResult:
    from ovat.config.workflow import load_workflow
    if not args:
        return _error("Usage: /validate <config.yml>")
    try:
        cfg = load_workflow(args[0])
    except FileNotFoundError:
        return _error(f"No such file: {args[0]}")
    except Exception as exc:
        return _error(f"Invalid workflow: {exc}")
    return _ok(f"Valid. model={cfg.model.name}  agent={cfg.agent.type}  "
               f"tools={[t.name for t in cfg.tools]}  "
               f"rag={'on' if cfg.rag else 'off'}")


def cmd_index(args: list) -> CommandResult:
    from ovat.agent.factory import build_rag
    from ovat.config.workflow import load_workflow
    from ovat.rag.indexer import index_folder
    if len(args) < 2:
        return _error("Usage: /index <folder> <config.yml>")
    folder, config = args[0], args[1]
    try:
        cfg = load_workflow(config)
    except Exception as exc:
        return _error(f"Could not load {config}: {exc}")
    if cfg.rag is None:
        return _error("That workflow has no rag: section to index into.")
    try:
        retriever = build_rag(cfg)
        summary = index_folder(folder, retriever,
                               size=cfg.rag.chunk.size, overlap=cfg.rag.chunk.overlap)
    except FileNotFoundError as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Indexing failed: {exc}")
    return _ok(f"Indexed {summary['chunks']} chunks from {summary['files']} files "
               f"into {cfg.rag.retriever.db_path}")


def cmd_models(args: list) -> CommandResult:
    from ovat.core.model_manager import ModelManager
    try:
        names = ModelManager().list_models()
    except FileNotFoundError:
        return CommandResult(Text("ovms not on PATH; model listing needs the AI PC.",
                                  style=ui.YELLOW))
    except Exception as exc:
        return _error(f"Could not list models: {exc}")
    if not names:
        return _info("No models reported by OVMS.")
    return CommandResult(Text("\n".join(names), style=ui.CYAN))


def cmd_clear(args: list) -> CommandResult:
    return CommandResult(Text(""), action="clear")


def cmd_exit(args: list) -> CommandResult:
    return CommandResult(Text("Bye."), action="exit")


# The registry. Order here is the order shown in /help and the slash menu.
COMMANDS: dict[str, SlashCommand] = {
    "help":     SlashCommand("help", "/help", "show this list of commands", cmd_help),
    "doctor":   SlashCommand("doctor", "/doctor [config]",
                             "check Python, deps, devices, OVMS, a config", cmd_doctor),
    "init":     SlashCommand("init", "/init [path]",
                             "write a starter workflow.yml you can edit", cmd_init),
    "validate": SlashCommand("validate", "/validate <config>",
                             "load and validate a workflow file", cmd_validate),
    "index":    SlashCommand("index", "/index <folder> <config>",
                             "index a folder of docs for search_docs", cmd_index),
    "models":   SlashCommand("models", "/models",
                             "list models OVMS can serve (AI PC)", cmd_models),
    "clear":    SlashCommand("clear", "/clear", "clear the output area", cmd_clear),
    "exit":     SlashCommand("exit", "/exit", "leave the TUI", cmd_exit),
}


def match_commands(prefix: str) -> list:
    """Return the commands whose name starts with `prefix` (no leading slash).

    The TUI calls this as the user types to build the live slash menu.
    """
    prefix = prefix.lstrip("/").lower()
    return [c for name, c in COMMANDS.items() if name.startswith(prefix)]


def dispatch(line: str) -> CommandResult | None:
    """Parse a typed line and run the matching command. None means 'do nothing'.

    The line is expected to start with '/'. I use shlex so a quoted path with
    spaces stays one argument, the same way a shell would split it.
    """
    line = line.strip()
    if not line:
        return None
    if not line.startswith("/"):
        return CommandResult(Text("Type a /command. Try /help.", style=ui.YELLOW))
    try:
        parts = shlex.split(line[1:])
    except ValueError:
        return _error("Could not parse that line (check your quotes).")
    if not parts:
        return cmd_help([])
    name, args = parts[0].lower(), parts[1:]
    cmd = COMMANDS.get(name)
    if cmd is None:
        return _error(f"Unknown command: /{name}. Try /help.")
    return cmd.handler(args)
