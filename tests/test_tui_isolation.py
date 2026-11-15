# tests/test_tui_isolation.py
"""The isolation contract: the CLI must fully work WITHOUT the TUI installed.

The TUI is an optional front-end (`pip install 'ovat[tui]'`). These tests
simulate an install without the extra by poisoning sys.modules with None so any
`import ovat.tui.agent_app` raises ImportError.

This file must never import textual directly or indirectly.
"""
import sys

from typer.testing import CliRunner

from ovat.cli.main import app

runner = CliRunner()


def _no_tui(monkeypatch):
    """Make importing the Textual TUI fail, as it would without `[tui]`."""
    monkeypatch.setitem(sys.modules, "ovat.tui.agent_app", None)


def test_bare_ovat_without_tui_prints_the_install_hint(monkeypatch):
    _no_tui(monkeypatch)
    result = runner.invoke(app, [])
    assert result.exit_code == 1                  # a hint, not a traceback
    assert isinstance(result.exception, SystemExit)
    assert "ovat[tui]" in result.output           # tells the user the extra


def test_every_subcommand_still_works_without_the_tui(monkeypatch, tmp_path):
    _no_tui(monkeypatch)
    # init writes a real file; doctor runs real checks; --help exercises the
    # rest of the command table. None of them may touch the TUI import.
    target = tmp_path / "workflow.yml"
    assert runner.invoke(app, ["init", str(target)]).exit_code == 0
    assert target.exists()
    assert runner.invoke(app, ["doctor", str(target)]).exit_code == 0
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "chat", "index", "init", "models", "serve", "doctor"):
        assert command in result.output


def test_cli_modules_never_import_textual_at_module_level():
    # The contract's root: importing the CLI must not drag textual in. If it
    # did, `ovat --help` would crash on a TUI-less install before any guard
    # could help. main.py may only import the TUI lazily inside functions;
    # shell.py must be textual-free entirely (that is what keeps it testable
    # and reusable). A source-level scan pins both.
    import ovat.cli.main
    import ovat.cli.shell

    main_src = open(ovat.cli.main.__file__, encoding="utf-8").read()
    shell_src = open(ovat.cli.shell.__file__, encoding="utf-8").read()
    for line in main_src.splitlines():
        if not line.startswith((" ", "\t")):          # module level only
            assert "textual" not in line, f"top-level textual import: {line!r}"
    assert "textual" not in shell_src


def test_the_cli_never_pulls_in_a_tui_only_module(monkeypatch, tmp_path):
    """The CLI must never import TUI-only Textual modules on normal commands."""
    _no_tui(monkeypatch)
    for name in ("ovat.tui.agent_app", "ovat.tui.telemetry",
                 "ovat.tui.doctor", "ovat.cli.theme"):
        monkeypatch.setitem(sys.modules, name, None)

    target = tmp_path / "workflow.yml"
    assert runner.invoke(app, ["init", str(target)]).exit_code == 0
    assert runner.invoke(app, ["doctor", str(target)]).exit_code == 0
    assert runner.invoke(app, ["--help"]).exit_code == 0


def test_the_agent_tui_is_optional_too(monkeypatch, tmp_path):
    """`ovat agent` is the TUI; without textual it prints the hint, and
    nothing else on the CLI path imports it."""
    _no_tui(monkeypatch)
    monkeypatch.setitem(sys.modules, "ovat.tui", None)
    result = runner.invoke(app, ["agent"])
    assert result.exit_code == 1
    assert "ovat[tui]" in result.output
    target = tmp_path / "workflow.yml"
    assert runner.invoke(app, ["init", str(target)]).exit_code == 0
    assert runner.invoke(app, ["doctor", str(target)]).exit_code == 0
    assert runner.invoke(app, ["--help"]).exit_code == 0


def test_without_textual_agent_tui_prints_the_install_hint(monkeypatch):
    """`ovat.tui` itself imports no toolkit, so the hint comes from the app module failing to import."""
    monkeypatch.setitem(sys.modules, "ovat.tui.agent_app", None)
    for argv in (["agent"], ["tui"], []):
        result = runner.invoke(app, argv)
        assert result.exit_code == 1, (argv, result.output)
        assert "ovat[tui]" in result.output


def test_importing_ovat_tui_pulls_in_no_toolkit():
    import subprocess
    code = ("import sys, ovat.tui; "
            "assert 'textual' not in sys.modules, 'textual was imported'")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True)
    assert done.returncode == 0, done.stderr
