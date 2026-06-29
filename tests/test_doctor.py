# tests/test_doctor.py
"""Tests for `ovat doctor` and the diagnostics behind it.

Note to myself: these checks are about MY environment logic, not about a perfect
machine. I assert the things that are true wherever the test suite runs (Python
is new enough, the core deps import because they are installed for the tests),
and I assert the config logic with temp files I control.
"""
from typer.testing import CliRunner

from ovat.cli import diagnostics
from ovat.cli.diagnostics import (
    FAIL, OK, WARN, check_config, check_core_deps, check_python, run_checks,
)
from ovat.cli.main import app

runner = CliRunner()


def _write(tmp_path, text):
    p = tmp_path / "workflow.yml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_python_check_passes_on_supported_interpreter():
    # The suite itself runs on 3.10+, so this must be ok here.
    assert check_python().status == OK


def test_core_deps_check_passes_when_installed():
    # The deps are installed to run the tests, so none should be missing.
    assert check_core_deps().status == OK


def test_run_checks_includes_the_base_checks():
    names = {c.name for c in run_checks()}
    assert {"Python", "Core dependencies", "OpenVINO devices", "OVMS binary"} <= names


def test_config_check_reports_a_valid_workflow(tmp_path):
    path = _write(tmp_path, "model:\n  name: Qwen3-8B-int4-ov\n"
                            "agent:\n  type: native\n")
    checks = check_config(path)
    config_check = checks[0]
    assert config_check.status == OK
    assert "Qwen3-8B-int4-ov" in config_check.detail
    assert "native" in config_check.detail


def test_config_check_fails_on_missing_file():
    checks = check_config("/no/such/workflow.yml")
    assert checks[0].status == FAIL


def test_config_check_warns_when_embeddings_model_absent(tmp_path):
    path = _write(tmp_path,
                  "model:\n  name: m\n"
                  "rag:\n  embeddings:\n    provider: genai\n"
                  "    model: /definitely/not/here/bge\n")
    checks = check_config(path)
    emb = [c for c in checks if c.name == "Embeddings model"]
    assert emb and emb[0].status == WARN


def test_doctor_command_runs_and_reports(tmp_path):
    # No config: with the test deps installed and no failures, doctor exits 0.
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.output


def test_doctor_command_fails_on_bad_config(tmp_path):
    path = _write(tmp_path, "this: is not a valid workflow\n")
    result = runner.invoke(app, ["doctor", path])
    assert result.exit_code == 1
