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
    assert {"Python", "Core dependencies", "Local GenAI", "OpenVINO devices",
            "Device routing", "OVMS serving"} <= names


def test_doctor_reports_on_every_engine_the_factory_offers():
    """doctor checked LangChain and nothing else, which was complete while
    react was the only framework engine. Since W7-W8 there are four, so on a
    box with LlamaIndex and the OpenAI Agents SDK missing, doctor still said
    "All clear" and `ovat bench` then failed on two of its four rows. The
    point of doctor is to answer "what is ready here" before something else
    has to fail to tell you.

    Driven off factory.AGENT_TYPES so a fifth engine cannot be added without
    this failing: doctor going quietly stale is the bug being fixed.
    """
    from ovat.agent.factory import AGENT_TYPES

    details = " ".join(f"{c.name} {c.detail}" for c in run_checks())
    for engine in AGENT_TYPES:
        if engine == "native":
            continue          # no extra to install; it is always available
        assert f"agent.type: {engine}" in details, \
            f"doctor never mentions the {engine} engine"


def test_local_genai_check_confirms_the_no_server_path():
    # This row is the "how do I use OVAT on a Mac" answer, in the tool itself.
    from ovat.cli.diagnostics import check_local_genai
    check = check_local_genai()
    assert check.status == OK                     # installed for the tests
    assert "no server needed" in check.detail


def test_ovms_serving_is_platform_aware(monkeypatch):
    from ovat.cli import diagnostics

    # On macOS: never "not on PATH"; the honest answer is "does not run here".
    monkeypatch.setattr(diagnostics.sys, "platform", "darwin")
    check = diagnostics.check_ovms_serving()
    assert check.status == WARN
    assert "does not run on macOS" in check.detail

    # Elsewhere: the locator's verdict, with the fix spelled out.
    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    monkeypatch.delenv("OVAT_OVMS", raising=False)
    import ovat.core.ovms_locator as locator
    monkeypatch.setattr(locator.shutil, "which", lambda name: None)
    monkeypatch.setattr(locator, "_KNOWN_DIRS", [])
    check = diagnostics.check_ovms_serving()
    assert check.status == WARN
    assert "ovms_binary" in check.detail          # names the exact fix


def test_ovms_serving_reports_a_config_supplied_binary(monkeypatch, tmp_path):
    from ovat.cli import diagnostics
    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    import ovat.core.ovms_locator as locator
    exe = tmp_path / locator._EXE
    exe.write_text("#!/bin/sh\n")
    check = diagnostics.check_ovms_serving(str(tmp_path))
    assert check.status == OK
    assert "config" in check.detail and str(exe) in check.detail


def test_device_routing_check_shows_the_layer9_table():
    from ovat.cli.diagnostics import check_device_routing
    check = check_device_routing()
    assert check.status == OK
    # On any machine the row names all three model types; on this Mac all
    # three route to CPU, on an AI PC LLM→GPU and embeddings→NPU.
    # "emb" not "embeddings": the doctor's Detail column is 76 wide and the
    # row now carries the config's own devices too, so the labels were
    # shortened to keep both halves visible rather than truncated.
    for label in ("LLM→", "emb→", "whisper→"):
        assert label in check.detail


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


def test_config_check_warns_on_search_docs_without_rag(tmp_path):
    # The stub trap: search_docs declared, no rag: block. Legal, but the tool
    # would return fake "[stub]" text at runtime; doctor must say so.
    path = _write(tmp_path,
                  "model:\n  name: m\n"
                  "tools:\n  - name: search_docs\n    type: builtin\n")
    checks = check_config(path)
    stub = [c for c in checks if c.name == "search_docs mode"]
    assert stub and stub[0].status == WARN
    assert "stub" in stub[0].detail


def test_config_check_no_stub_warning_when_rag_is_configured(tmp_path):
    path = _write(tmp_path,
                  "model:\n  name: m\n"
                  "tools:\n  - name: search_docs\n    type: builtin\n"
                  "rag:\n  embeddings:\n    provider: genai\n"
                  "    model: /definitely/not/here/bge\n")
    checks = check_config(path)
    assert not [c for c in checks if c.name == "search_docs mode"]


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


# Device routing is ADVICE, and used to read as a report of what is running

def test_routing_is_labelled_as_advice_not_as_fact():
    """Unlabelled, a user whose config said `device: CPU` saw
    "embeddings→NPU" and reasonably concluded either the config was being
    ignored or the doctor was wrong. It is neither: DeviceManager describes
    what this hardware would suit, not what the workflow asked for."""
    from ovat.cli.diagnostics import check_device_routing

    detail = check_device_routing().detail
    assert "best here" in detail
    assert "yours" not in detail       # nothing to compare against


def test_routing_shows_the_config_alongside_the_recommendation(tmp_path):
    """With a config loaded the two are shown together, so they read as two
    different things rather than one contradicting the other."""
    from ovat.cli.diagnostics import check_device_routing
    from ovat.config.workflow import WorkflowConfig

    config = WorkflowConfig(
        model={"name": "m", "device": "GPU"},
        rag={"embeddings": {"provider": "genai", "model": "x",
                            "device": "NPU", "dim": 384},
             "retriever": {"provider": "sqlite-vec", "db_path": ":memory:"}})
    detail = check_device_routing(config).detail
    assert "yours: LLM→GPU" in detail
    assert "emb→NPU" in detail


def test_the_routing_line_fits_the_doctor_table():
    """The Detail column is 76 columns. A suffix that gets cut off is worse
    than none, because the reader cannot tell there was more."""
    from ovat.cli.diagnostics import check_device_routing
    from ovat.config.workflow import WorkflowConfig

    config = WorkflowConfig(model={"name": "m", "device": "GPU"})
    assert len(check_device_routing(config).detail) <= 76


def test_doctor_warns_when_a_declared_tool_has_no_model_on_disk(tmp_path, monkeypatch):
    """describe_image and transcribe need a model; say so BEFORE the run.

    Both degrade cleanly at call time -- no traceback, and the agent explains
    itself in plain language -- but the AI PC sweep found neither had ever been
    observed WORKING, because the models were absent and nothing said so until
    the tool was already mid-run. doctor is where a missing prerequisite
    belongs, next to the existing "search_docs mode" and "Embeddings model"
    rows it mirrors.
    """
    from ovat.cli.diagnostics import run_checks

    monkeypatch.delenv("OVAT_VLM_MODEL", raising=False)
    monkeypatch.delenv("OVAT_WHISPER_MODEL", raising=False)
    config = tmp_path / "w.yml"
    config.write_text(
        "model:\n  name: m\n"
        "tools:\n"
        "  - name: describe_image\n    type: builtin\n"
        "  - name: transcribe\n    type: builtin\n", encoding="utf-8")

    labels = {c.name: c for c in run_checks(str(config))}
    assert "describe_image model" in labels
    assert "transcribe model" in labels
    assert labels["describe_image model"].status == "warn"
    assert "OVAT_VLM_MODEL" in labels["describe_image model"].detail
    assert "OVAT_WHISPER_MODEL" in labels["transcribe model"].detail


def test_doctor_says_nothing_about_tools_that_are_not_declared(tmp_path):
    """No vision tool in the workflow means no vision row. The table is a
    report on THIS config, not a catalogue of everything OVAT could do."""
    from ovat.cli.diagnostics import run_checks

    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\ntools:\n  - name: search_docs\n"
                      "    type: builtin\n", encoding="utf-8")
    labels = {c.name for c in run_checks(str(config))}
    assert "describe_image model" not in labels
    assert "transcribe model" not in labels
