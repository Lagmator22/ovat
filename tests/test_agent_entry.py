"""Bare `ovat` and `ovat tui` open the agent TUI, the same one `ovat agent` opens.

The first TUI, the launcher, is only reached with OVAT_LAUNCHER=classic.
"""
import pytest
from typer.testing import CliRunner

from ovat.cli.main import app

runner = CliRunner()


@pytest.fixture
def opened(monkeypatch):
    """Replace the agent TUI with a recorder, so no terminal is needed."""
    calls = []

    def fake_runner():
        def run(config, **kwargs):
            calls.append((config, kwargs))
        return run

    monkeypatch.setattr("ovat.tui.runner", fake_runner)
    monkeypatch.delenv("OVAT_TUI", raising=False)
    monkeypatch.delenv("OVAT_LAUNCHER", raising=False)
    monkeypatch.delenv("OVAT_AGENT_UI", raising=False)
    return calls


def test_bare_ovat_opens_the_agent_tui(opened):
    result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    assert opened == [(None, {})]


def test_ovat_tui_opens_the_agent_tui_and_takes_a_workflow(opened, tmp_path):
    wf = tmp_path / "workflow.yml"
    wf.write_text("x: 1\n")
    result = runner.invoke(app, ["tui", str(wf)])
    assert result.exit_code == 0, result.output
    assert opened == [(str(wf), {})]


def test_ovat_tui_can_continue_the_last_conversation(opened):
    result = runner.invoke(app, ["tui", "--continue"])
    assert result.exit_code == 0, result.output
    assert opened == [(None, {"continue_session": True,
                              "resume_session_id": None})]


def test_the_launcher_is_still_there_behind_an_env_var(opened, monkeypatch):
    pytest.importorskip("textual")
    shown = []
    monkeypatch.setattr("ovat.cli.tui.run_tui", lambda: shown.append(1))
    monkeypatch.setenv("OVAT_LAUNCHER", "classic")
    for argv in ([], ["tui"]):
        assert runner.invoke(app, argv).exit_code == 0
    assert shown == [1, 1]
    assert opened == []
