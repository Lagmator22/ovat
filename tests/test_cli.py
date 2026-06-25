# tests/test_cli.py
"""Tests for the ovat command line.

Note to myself: typer ships a CliRunner that invokes my commands in-process and
captures their output and exit code, so I can test the CLI without a real shell
or a running OVMS server. I test the wiring (init writes a usable config,
dry-run builds the agent) and the error paths (bad config exits non-zero).
"""
from typer.testing import CliRunner

from ovat.cli.main import app
from ovat.config.workflow import load_workflow

runner = CliRunner()


def test_init_writes_a_loadable_config(tmp_path):
    target = tmp_path / "workflow.yml"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0
    assert target.exists()
    # the file it wrote must itself be a valid workflow
    cfg = load_workflow(str(target))
    assert cfg.model.name == "Qwen3-8B-int4-ov"


def test_init_refuses_to_overwrite(tmp_path):
    target = tmp_path / "workflow.yml"
    target.write_text("model:\n  name: x\n")
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 1


def test_run_dry_run_builds_agent_without_a_server():
    result = runner.invoke(
        app, ["run", "examples/workflow.yml", "--input", "hello", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "search_docs" in result.output
    assert "dry-run" in result.output


def test_run_with_missing_config_fails():
    result = runner.invoke(app, ["run", "/no/such/file.yml", "-i", "hi"])
    assert result.exit_code != 0
