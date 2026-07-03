# tests/test_tui_isolation.py
"""The isolation contract: the CLI must fully work WITHOUT the TUI installed.

The TUI is an optional front-end (`pip install 'ovat[tui]'`) that can be taken
back out at any time. These tests simulate an install without the extra by
poisoning the module cache — putting None in sys.modules makes any
`import ovat.cli.tui` raise ImportError, exactly what a missing textual does.

Note to myself: this file must never import textual (directly or indirectly),
or it would defeat its own purpose on a machine without the extra.
"""
import sys

from typer.testing import CliRunner

from ovat.cli.main import app

runner = CliRunner()


def _no_tui(monkeypatch):
    """Make `import ovat.cli.tui` fail, as it would without the [tui] extra."""
    monkeypatch.setitem(sys.modules, "ovat.cli.tui", None)


def test_bare_ovat_without_tui_prints_the_install_hint(monkeypatch):
    _no_tui(monkeypatch)
    result = runner.invoke(app, [])
    assert result.exit_code == 0                  # a hint, not a crash
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
