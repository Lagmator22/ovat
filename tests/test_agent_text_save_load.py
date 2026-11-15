"""/save and /load in the agent TUI stay inside .ovat/sessions."""
from types import SimpleNamespace

from ovat.agent.session import Session
from ovat.agentic.sessions import sessions_dir
from ovat.tui import agent_text


def test_save_cannot_write_outside_the_sessions_folder(tmp_path):
    agent = SimpleNamespace(session=Session(system_prompt="sys"))
    agent_text.save_conversation(agent, str(tmp_path), "../../escape")
    assert not (tmp_path / "escape.json").exists()
    assert not (tmp_path.parent / "escape.json").exists()
    assert (sessions_dir(str(tmp_path)) / "escape.json").is_file()


def test_load_goes_through_restore_memory(tmp_path):
    folder = sessions_dir(str(tmp_path))
    folder.mkdir(parents=True)
    saved = Session(system_prompt="planted prompt")
    saved.messages.append({"role": "user", "content": "hi"})
    saved.save(str(folder / "old.json"))
    seen = []
    session = SimpleNamespace(agent=SimpleNamespace(session=Session()),
                              restore_memory=seen.append)
    agent_text.load_conversation(session, str(tmp_path), "old")
    assert seen and seen[0][-1] == {"role": "user", "content": "hi"}
