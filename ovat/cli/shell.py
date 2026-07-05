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
import time
from dataclasses import dataclass


@dataclass
class SlashTemplate:
    """A slash shortcut: the text it inserts, or None if it is a UI action."""

    name: str               # e.g. "/doctor"
    insert: str | None      # the command text it fills in, or None for actions
    description: str
    # Where the cursor should land inside `insert` (None = end of text).
    # `/run` inserts `ovat run  --input ""`; the user's next keystrokes are
    # the config path, so the cursor belongs in that gap, not at the end.
    cursor: int | None = None


# The slash menu. Each non-action entry inserts a real `ovat ...` command the
# user can finish and run. /clear and /exit are handled by the TUI itself.
TEMPLATES = [
    SlashTemplate("/chat", None, "chat with your indexed docs (local model)"),
    SlashTemplate("/doctor", "ovat doctor ", "check env + an optional config"),
    SlashTemplate("/run", "ovat run  --input \"\"", "run the agent on a config",
                  cursor=len("ovat run ")),
    SlashTemplate("/index", "ovat index  ", "index a docs folder for search_docs",
                  cursor=len("ovat index ")),
    SlashTemplate("/init", "ovat init ", "write a starter workflow.yml"),
    SlashTemplate("/validate", "ovat doctor ", "validate a workflow file (via doctor)"),
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


def iter_display_lines(stream, progress_interval: float = 0.5,
                       _clock=time.monotonic):
    """Yield printable lines from a process stream, taming \\r progress bars.

    The problem: pip/tqdm redraw ONE line many times a second with bare
    carriage returns and send no newline until the end. Iterating the stream
    by lines therefore shows nothing for minutes, then dumps one giant line.

    Two tricks here:
    - os.read() on the raw fd returns whatever bytes are AVAILABLE (a plain
      text-mode read(4096) would block until it had all 4096 chars; no
      streaming at all).
    - a frame that ends in \\r is a transient redraw. The log is append-only,
      so instead of appending every redraw I let at most one through per
      progress_interval; the final state arrives with the tool's closing
      newline anyway.
    """
    fd = stream.fileno()
    buf = ""
    last_progress = 0.0
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:            # stream closed under us (cancel path)
            break
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        # Normalise Windows line ends, then treat bare \r as its own boundary.
        buf = buf.replace("\r\n", "\n")
        while True:
            i_nl, i_cr = buf.find("\n"), buf.find("\r")
            if i_nl == -1 and i_cr == -1:
                break
            if i_cr == -1 or (i_nl != -1 and i_nl < i_cr):
                line, buf = buf[:i_nl], buf[i_nl + 1:]
                yield line                       # real line: always shown
            else:
                line, buf = buf[:i_cr], buf[i_cr + 1:]
                now = _clock()
                if now - last_progress >= progress_interval:
                    last_progress = now
                    yield line                   # sampled progress frame
    if buf:
        yield buf                                # whatever EOF left behind


def run_command(cmd: str, cwd: str, on_line, env: dict | None = None) -> int:
    """Run `cmd` to completion, calling on_line(text) for each output line.

    Returns the process exit code. This is the synchronous helper my tests use;
    the TUI uses spawn() directly so it can also cancel a long-running command.
    """
    proc = spawn(cmd, cwd, env)
    try:
        for line in iter_display_lines(proc.stdout):
            on_line(line)
    finally:
        if proc.stdout:
            proc.stdout.close()
    return proc.wait()
