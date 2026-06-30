# tests/test_shell.py
"""Tests for the shell-execution layer behind the TUI.

Note to myself: this proves the TUI actually runs real commands. I use trivial
commands (echo, exit, pwd) so the tests are fast and deterministic, and I never
need a terminal or Textual here.
"""
import os
import sys

from ovat.cli import shell


def test_venv_env_puts_venv_bin_first_and_keeps_colour():
    env = shell.venv_env()
    bin_dir = os.path.dirname(sys.executable)
    assert env["PATH"].split(os.pathsep)[0] == bin_dir   # venv tools win
    assert "FORCE_COLOR" in env                          # colour stays on


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
    code = shell.run_command("exit 3", os.getcwd(), lines.append)
    assert code == 3


def test_run_command_runs_in_the_given_directory(tmp_path):
    lines = []
    shell.run_command("pwd", str(tmp_path), lines.append)
    # macOS may resolve /tmp to /private/tmp, so I match the leaf directory name.
    assert any(tmp_path.name in ln for ln in lines)
