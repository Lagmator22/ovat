# tests/test_a2a_isolation.py
"""The A2A SDK is optional, and only ovat/a2a/sdk/ may touch it.

Two contracts, each a test that fails the moment someone breaks it:
- the CLI works on a base install: no `a2a` and no protobuf get imported
  just by running `ovat`, and `ovat a2a ...` without the extra prints the
  install command instead of a traceback;
- the layering in ovat/a2a/__init__.py: an SDK upgrade is a change to
  sdk/ alone, which only stays true while nothing else imports `a2a`.
"""
import re
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from ovat.cli.main import app

runner = CliRunner()
OVAT = Path(__file__).parents[1] / "ovat"
SDK_DIR = OVAT / "a2a" / "sdk"
_IMPORTS_SDK = re.compile(r"^\s*(import a2a\b|from a2a\b)", re.M)


def test_only_the_sdk_adapter_imports_the_sdk():
    offenders = [str(p.relative_to(OVAT.parent))
                 for p in OVAT.rglob("*.py")
                 if SDK_DIR not in p.parents
                 and _IMPORTS_SDK.search(p.read_text(encoding="utf-8"))]
    assert offenders == [], (
        "only ovat/a2a/sdk/ may import the A2A SDK; route these through "
        f"ovat/a2a/protocol.py instead: {offenders}")


def test_importing_the_cli_loads_no_sdk_and_no_protobuf():
    # A subprocess: this process has already imported the SDK elsewhere.
    # Not uvicorn: mcp, a base dependency, already brings it in.
    code = ("import ovat.cli.main, sys;"
            "bad=[m for m in ('a2a','google.protobuf') "
            "if m in sys.modules];print(bad);sys.exit(1 if bad else 0)")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _no_sdk(monkeypatch):
    for name in ("a2a", "ovat.a2a.sdk.server", "ovat.a2a.sdk.client"):
        monkeypatch.setitem(sys.modules, name, None)


def test_the_cli_works_without_the_extra(monkeypatch, tmp_path):
    _no_sdk(monkeypatch)
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["a2a", "--help"]).exit_code == 0
    target = tmp_path / "workflow.yml"
    assert runner.invoke(app, ["init", str(target)]).exit_code == 0
    assert runner.invoke(app, ["doctor", str(target)]).exit_code == 0


def test_a2a_commands_without_the_extra_print_the_install_command(
        monkeypatch, tmp_path):
    _no_sdk(monkeypatch)
    target = tmp_path / "workflow.yml"
    runner.invoke(app, ["init", str(target)])
    for args in (["a2a", "serve", str(target)], ["a2a", "card", str(target)],
                 ["a2a", "call", "http://127.0.0.1:1", "hi"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1, args
        assert 'pip install "ovat[a2a]"' in result.output, args
