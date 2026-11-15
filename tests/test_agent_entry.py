"""Bare `ovat` and `ovat tui` open the agent TUI, the same one `ovat agent` opens."""
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
