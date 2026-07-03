# ovat/cli/shell.py
"""Run real terminal commands from the TUI, inside the venv, with colour.

The TUI is a styled front-end over a real shell: whatever the user types (that
is not a UI action) is executed here exactly as it would be in their terminal.
Keeping this layer free of any Textual import means I can unit test it with a
trivial command like `echo` and never open a screen.

Two design choices make it behave like an activated venv:
  * I prepend the venv's bin directory to PATH, so `ovat`, `python`, and
    `pytest` resolve to the venv copies even when the venv is not "activated".
  * I set FORCE_COLOR so tools like ovat and pytest still emit ANSI colour into
    the pipe; the TUI turns those codes back into styled text.
"""
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class SlashTemplate:
    """A slash shortcut: the text it inserts, or None if it is a UI action."""

    name: str               # e.g. "/doctor"
    insert: str | None      # the command text it fills in, or None for actions
    description: str


# The slash menu. Each non-action entry inserts a real `ovat ...` command the
# user can finish and run. /clear and /exit are handled by the TUI itself.
TEMPLATES = [
    SlashTemplate("/doctor", "ovat doctor ", "check env + an optional config"),
    SlashTemplate("/run", "ovat run  --input \"\"", "run the agent on a config"),
    SlashTemplate("/index", "ovat index  ", "index a docs folder for search_docs"),
    SlashTemplate("/init", "ovat init ", "write a starter workflow.yml"),
    SlashTemplate("/serve", "ovat serve ", "start OVMS for a config (AI PC)"),
    SlashTemplate("/models", "ovat models list", "list models OVMS can serve (AI PC)"),
    SlashTemplate("/help", "ovat --help", "show every OVAT subcommand"),
    SlashTemplate("/clear", None, "clear the output area"),
    SlashTemplate("/exit", None, "leave the TUI"),
]

TEMPLATES_BY_NAME = {t.name: t for t in TEMPLATES}


def match_templates(prefix: str) -> list:
    """Templates whose name starts with `prefix` (used to build the dropdown)."""
    prefix = prefix.lower()
    return [t for t in TEMPLATES if t.name.startswith(prefix)]


def venv_env(columns: int | None = None) -> dict:
    """An environment that resolves venv tools and keeps colour on."""
    env = dict(os.environ)
    bin_dir = os.path.dirname(sys.executable)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env.setdefault("FORCE_COLOR", "1")        # ovat/pytest keep their colours
    env["PYTHONUNBUFFERED"] = "1"             # stream output line by line
    # Mark every child as "inside the TUI" (like tmux's $TMUX): a bare `ovat`
    # typed in here must NOT open a second TUI in a piped subprocess.
    env["OVAT_TUI"] = "1"
    if columns:
        # So rich tools wrap to the TUI width instead of the default 80.
        env["COLUMNS"] = str(columns)
    return env


def spawn(cmd: str, cwd: str, env: dict | None = None) -> subprocess.Popen:
    """Start `cmd` in a shell. The caller streams stdout and can terminate it.

    I route through the user's shell so pipes, globs, and redirects work like a
    normal terminal. stdin is closed so an interactive program gets EOF and
    exits instead of hanging the UI.
    """
    env = env or venv_env()
    shell = env.get("SHELL", "/bin/sh")
    return subprocess.Popen(
        cmd, shell=True, executable=shell, cwd=cwd, env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )


def run_command(cmd: str, cwd: str, on_line, env: dict | None = None) -> int:
    """Run `cmd` to completion, calling on_line(text) for each output line.

    Returns the process exit code. This is the synchronous helper my tests use;
    the TUI uses spawn() directly so it can also cancel a long-running command.
    """
    proc = spawn(cmd, cwd, env)
    try:
        for line in proc.stdout:
            on_line(line.rstrip("\n"))
    finally:
        if proc.stdout:
            proc.stdout.close()
    return proc.wait()
