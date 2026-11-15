"""a workflow.yml found in an untrusted folder is not built on its own.

building a workflow starts its mcp_stdio commands, so opening `ovat` in a
freshly cloned repository must not run what that repository's workflow.yml
says until the folder is trusted. a workflow named on the command line is
the user's own choice and is always used.
"""
import pytest

pytest.importorskip("textual")

from ovat.agentic import trust  # noqa: E402
from ovat.tui import agent_app  # noqa: E402


@pytest.fixture
def opened(monkeypatch, tmp_path):
    seen = []

    async def fake_run_agent(workflow, **kwargs):
        seen.append(workflow)
    monkeypatch.setattr(agent_app, "run_agent", fake_run_agent)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workflow.yml").write_text("model: {name: x}\n")
    return seen, tmp_path


def test_untrusted_folder_does_not_build_its_workflow(opened, capsys):
    seen, folder = opened
    agent_app.run()
    assert seen == [None]
    assert "not trusted" in capsys.readouterr().err


def test_trusted_folder_builds_its_workflow(opened):
    seen, folder = opened
    trust.decide(folder, "trust_cwd")
    agent_app.run()
    assert seen == [str(folder / "workflow.yml")]


def test_a_workflow_named_by_the_user_is_always_used(opened):
    seen, folder = opened
    agent_app.run(str(folder / "workflow.yml"))
    assert seen == [str(folder / "workflow.yml")]
