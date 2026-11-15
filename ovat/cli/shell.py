# ovat/cli/shell.py
"""Subprocess shell execution helper for interactive UIs without Textual coupling.

Prepends the active Python environment's ``bin``/``Scripts`` directory to
``PATH`` and sets ``FORCE_COLOR=1`` so child CLI commands resolve consistently
and emit ANSI formatting cleanly without requiring Textual in unit tests.
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
    # Include usage arguments in the description so users can pass another
    # config or model path explicitly when .ovat/chat_prefs.json points to an
    # older configuration.
    SlashTemplate("/chat", None,
                  "chat with your docs  ·  /chat [config] [model-path]"),
    # insert=None makes this a UI ACTION: the TUI opens its own doctor screen,
    # which can re-run the checks in place. `/validate` below still inserts the
    # subprocess form for anyone who wants `ovat doctor <config>` in the log.
    SlashTemplate("/doctor", None, "environment + config checks, in-app"),
    SlashTemplate("/telemetry", None, "live numbers and graphs, in-app"),
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
    """Launch ``cmd`` in the platform shell with closed stdin and piped output.

    Uses the user's shell so pipes, globs, and redirects work normally, and sets
    ``stdin=DEVNULL`` so interactive prompts receive EOF instead of hanging UI
    callers. ``executable`` is set only on POSIX because Windows ``shell=True``
    uses ``executable`` to replace ``COMSPEC`` (``cmd.exe``) rather than wrap it.
    """
    env = env or venv_env()
    kwargs = {}
    if os.name != "nt":
        kwargs["executable"] = env.get("SHELL", "/bin/sh")
    return subprocess.Popen(
        cmd, shell=True, cwd=cwd, env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, **kwargs,
    )


def terminate(proc: subprocess.Popen) -> None:
    """Ask a spawn()ed command to stop. On Windows, end its whole tree.

    On POSIX this is plain SIGTERM, a request the program can act on.
    Windows has no such request for a console child, and with shell=True the
    Popen is cmd.exe, not the program: proc.terminate() ended cmd.exe and
    left the real program running (a venv python is two processes deep, a
    launcher and the base interpreter). The orphan kept the output pipe open,
    so the TUI never saw EOF and stayed busy until the program finished by
    itself. Raises ProcessLookupError like Popen.terminate on POSIX.
    """
    if os.name != "nt":
        proc.terminate()
        return
    _kill_tree(proc)


def kill(proc: subprocess.Popen) -> None:
    """Force a spawn()ed command to stop: SIGKILL, or its whole tree on Windows."""
    if os.name != "nt":
        proc.kill()
        return
    _kill_tree(proc)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Windows: kill cmd.exe and every process below it.

    The descendants are listed BEFORE anything dies, because Windows does not
    reparent orphans: once cmd.exe is gone, nothing links its children back to
    the pid we hold. cmd.exe dies first so it cannot start the next command of
    an `a & b` line while its current one is being killed.
    """
    import psutil

    try:
        descendants = psutil.Process(proc.pid).children(recursive=True)
    except psutil.Error:
        descendants = []                  # already gone, or not ours to list
    proc.kill()
    for child in descendants:
        try:
            child.kill()
        except psutil.Error:
            pass                          # exited on its own meanwhile


def iter_display_lines(stream, progress_interval: float = 0.5,
                       _clock=time.monotonic):
    """Yield printable lines from a subprocess pipe, throttling ``\\r`` progress redraws.

    Uses unbuffered ``os.read`` on the raw file descriptor for incremental
    streaming and limits repeated ``\\r`` redraw frames to ``progress_interval``
    so append-only logs do not flood with transient progress updates.
    """
    fd = stream.fileno()
    buf = ""
    last_progress = 0.0
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:            # Stream closed on process exit or cancellation.
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
    """Run ``cmd`` synchronously and pass each display line to ``on_line(text)``."""
    proc = spawn(cmd, cwd, env)
    try:
        for line in iter_display_lines(proc.stdout):
            on_line(line)
    finally:
        if proc.stdout:
            proc.stdout.close()
    return proc.wait()
