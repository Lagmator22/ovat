# tests/test_shell.py
"""Tests for the shell-execution layer behind the TUI.

Note to myself: this proves the TUI actually runs real commands. I use trivial
commands (echo, exit, pwd) so the tests are fast and deterministic, and I never
need a terminal or Textual here.
"""
import os
import sys

from ovat.cli import shell
from tests.conftest import py_command


def test_venv_env_puts_venv_bin_first_and_keeps_colour():
    env = shell.venv_env()
    bin_dir = os.path.dirname(sys.executable)
    assert env["PATH"].split(os.pathsep)[0] == bin_dir   # venv tools win
    assert "FORCE_COLOR" in env                          # colour stays on


def test_venv_env_marks_children_as_inside_the_tui():
    # The recursion guard's first half: every child must see OVAT_TUI=1 so a
    # bare `ovat` typed inside the TUI refuses to open a second TUI.
    assert shell.venv_env()["OVAT_TUI"] == "1"


def test_match_templates_filters_by_prefix():
    assert {t.name for t in shell.match_templates("/d")} == {"/doctor"}
    assert len(shell.match_templates("/")) == len(shell.TEMPLATES)


def test_run_command_streams_output_and_returns_zero(tmp_path):
    lines = []
    code = shell.run_command("echo OVAT_HELLO", str(tmp_path), lines.append)
    assert code == 0
    assert any("OVAT_HELLO" in ln for ln in lines)


def test_run_command_reports_nonzero_exit():
    lines = []
    code = shell.run_command(py_command("import sys;sys.exit(3)"),
                             os.getcwd(), lines.append)
    assert code == 3


def test_run_command_runs_in_the_given_directory(tmp_path):
    lines = []
    shell.run_command(py_command("import os;print(os.getcwd())"),
                      str(tmp_path), lines.append)
    # macOS may resolve /tmp to /private/tmp, so I match the leaf directory name.
    assert any(tmp_path.name in ln for ln in lines)


def test_spawn_never_hands_windows_a_posix_shell(monkeypatch):
    """`executable` means different things per platform, so it is POSIX-only.

    With shell=True, POSIX uses executable to choose WHICH shell runs the
    command; Windows uses it to REPLACE cmd.exe. SHELL is normally unset on
    Windows, so passing "/bin/sh" there made every TUI command die with
    FileNotFoundError before it ran.
    """
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.clear()
        captured.update(kwargs)
        captured["cmd"] = cmd
        return object()

    monkeypatch.setattr(shell.subprocess, "Popen", fake_popen)
    env = {"SHELL": "/bin/zsh"}

    monkeypatch.setattr(shell.os, "name", "posix")
    shell.spawn("echo hi", os.getcwd(), env)
    assert captured["executable"] == "/bin/zsh"      # the user's own shell

    monkeypatch.setattr(shell.os, "name", "nt")
    shell.spawn("echo hi", os.getcwd(), env)
    assert "executable" not in captured             # cmd.exe, via COMSPEC
    assert captured["shell"] is True                # still a shell command


# iter_display_lines: \r progress bars must neither stall nor flood

class _FakeStream:
    """A fileno()-backed stream fed from a real OS pipe, so os.read works."""

    def __init__(self, payload: bytes):
        r, w = os.pipe()
        os.write(w, payload)
        os.close(w)                  # EOF once the payload is drained
        self._fd = r

    def fileno(self):
        return self._fd


def test_progress_frames_are_rate_limited_but_real_lines_always_pass():
    # Clock stands still -> after the first frame, every further \r frame is
    # inside the interval and must be suppressed; \n lines always pass.
    stream = _FakeStream(b"10%\r20%\r30%\rdone 100%\nnext step\n")
    out = list(shell.iter_display_lines(stream, progress_interval=0.5,
                                        _clock=lambda: 1000.0))
    assert out == ["10%", "done 100%", "next step"]


def test_progress_frames_all_pass_when_interval_is_zero():
    stream = _FakeStream(b"a\rb\rc\rend\n")
    out = list(shell.iter_display_lines(stream, progress_interval=0.0,
                                        _clock=lambda: 1000.0))
    assert out == ["a", "b", "c", "end"]


def test_windows_line_endings_and_eof_tail_are_handled():
    stream = _FakeStream(b"one\r\ntwo\r\nno-newline-at-eof")
    out = list(shell.iter_display_lines(stream, progress_interval=0.0))
    assert out == ["one", "two", "no-newline-at-eof"]


def test_run_command_streams_a_real_progress_bar_without_flooding():
    lines = []
    # chr(13)/chr(10): a literal \r survives neither cmd.exe nor two levels
    # of quoting, and this is the same byte stream either way.
    code = shell.run_command(
        py_command("import sys;sys.stdout.write('x'+chr(13)+'y'+chr(13)"
                   "+'z'+chr(13)+'final'+chr(10))"),
        os.getcwd(), lines.append)
    assert code == 0
    assert lines[-1] == "final"      # the real line always arrives
    assert len(lines) <= 4           # never more frames than were sent
