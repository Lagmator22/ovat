# ovat/tools/shell.py
"""The shell tool: run one command in the project folder.

Mirrors Mistral Vibe's bash.py (vibe/core/tools/builtins) in what the model
sees: stdout and stderr kept apart, an exit code, a timeout, capped output.
Left out on purpose: its 1,200-line command policy, managed shells and
experimental variants. Whether a command may run is the permission gate's
job (default: ask), not a parser's.

Two things that cost real sessions elsewhere:
- Kill the whole tree. A timed-out `sh -c "sleep 999"` leaves the sleep
  running if only the shell dies. POSIX: the command gets its own process
  group and the group is killed. Windows: ovat.cli.shell.kill (psutil walk),
  the same code Esc uses in the TUI.
- Drain both pipes on threads. Waiting for the exit first deadlocks as soon
  as a command writes more than a pipe buffer holds.
"""
import locale
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from ovat.agent.events import CANCELLED_TEXT
from ovat.cli import shell as cli_shell
from ovat.tools.project import Shown, fail

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_OUTPUT_CHARS = 30_000        # per stream; the head is kept


def _decode(data: bytes) -> str:
    """UTF-8 if it is, else the console's own code page (Windows)."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False),
                           errors="replace")


class _Capture(threading.Thread):
    """Reads one pipe to the end, keeping at most `cap` bytes of it."""

    def __init__(self, stream, cap: int):
        super().__init__(daemon=True, name="ovat-shell-pipe")
        self.stream, self.cap = stream, cap
        self.chunks: list[bytes] = []
        self.kept = 0
        self.truncated = False

    def run(self) -> None:
        for chunk in iter(lambda: self.stream.read1(65536), b""):
            room = self.cap - self.kept
            if room > 0:
                self.chunks.append(chunk[:room])
                self.kept += min(len(chunk), room)
            if len(chunk) > room:
                self.truncated = True

    def text(self) -> str:
        return _decode(b"".join(self.chunks))


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        cli_shell.kill(proc)
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def shell_impl(command: str, root: Path, timeout: int | None = None,
               stop: threading.Event | None = None) -> str:
    if not (command or "").strip():
        return fail("shell", "the command is empty.")
    try:
        seconds = int(timeout) if timeout else DEFAULT_TIMEOUT
    except (TypeError, ValueError):
        seconds = DEFAULT_TIMEOUT
    seconds = max(1, min(seconds, MAX_TIMEOUT))
    if stop is not None and stop.is_set():
        return CANCELLED_TEXT
    # shell=True: cmd.exe on Windows, /bin/sh elsewhere, so pipes and && work
    # as the person expects. stdin is closed so a command that waits for
    # input gets EOF instead of hanging until the timeout.
    extra = {"start_new_session": True} if os.name != "nt" else {}
    try:
        proc = subprocess.Popen(
            command, shell=True, cwd=root, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, **extra)
    except OSError as exc:
        return fail("shell", f"could not start the command: {exc}")
    out = _Capture(proc.stdout, MAX_OUTPUT_CHARS)
    err = _Capture(proc.stderr, MAX_OUTPUT_CHARS)
    out.start()
    err.start()

    deadline = time.monotonic() + seconds
    ended = None                     # "timeout" | "cancelled" | None
    while proc.poll() is None:
        if stop is not None and stop.is_set():
            ended = "cancelled"
        elif time.monotonic() >= deadline:
            ended = "timeout"
        if ended:
            _kill_tree(proc)
            break
        time.sleep(0.05)
    proc.wait()
    out.join(5)
    err.join(5)
    for pipe in (proc.stdout, proc.stderr):
        pipe.close()

    stdout, stderr = out.text(), err.text()
    truncated = out.truncated or err.truncated
    code = proc.returncode
    shown = {"kind": "shell", "command": command, "stdout": stdout,
             "stderr": stderr, "exit_code": code, "truncated": truncated}

    parts = []
    if stdout:
        parts.append(stdout.rstrip("\n"))
    if stderr:
        parts.append("[stderr]\n" + stderr.rstrip("\n"))
    if truncated:
        parts.append(f"[output cut at {MAX_OUTPUT_CHARS} characters]")
    body = "\n".join(parts)

    if ended == "cancelled":
        return Shown(CANCELLED_TEXT, {**shown, "failed": True,
                                      "error": CANCELLED_TEXT})
    if ended == "timeout":
        message = (f"Error: the command timed out after {seconds} s and was "
                   f"stopped.")
        return Shown(f"{message}\n{body}".rstrip(), {
            **shown, "failed": True, "error": message})
    if code != 0:
        message = f"Error: the command exited with code {code}."
        return Shown(f"{message}\n{body}".rstrip(), {
            **shown, "failed": True, "error": message})
    return Shown(body or "(the command succeeded and printed nothing)", shown)


SCHEMA = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a shell command in the user's project folder and "
                       "return its output and exit code. Use it to run tests, "
                       "git, or build tools. The command must not wait for "
                       "keyboard input.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string",
                            "description": "the command line to run"},
                "timeout": {"type": "integer",
                            "description": "seconds before the command is "
                                           "stopped (at most 600)",
                            "default": DEFAULT_TIMEOUT},
            },
            "required": ["command"],
        },
    },
}
