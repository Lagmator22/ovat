# tests/test_agentic_sessions.py
"""Saved conversations: the on-disk store, Session rewind, and the memory
API AgentSession offers the app server (restore, truncate, copy)."""
import json
import types

import pytest

from ovat.agent.loop import AgentLoop
from ovat.agent.session import Session
from ovat.agentic import sessions
from ovat.agentic import session as s
from tests.conftest import FakeLLMProvider


def record(sid, updated_at=1, **extra):
    return {"id": sid, "updated_at": updated_at, **extra}


# ---- SessionStore ---------------------------------------------------------

def test_save_then_load_round_trips(tmp_path):
    store = sessions.SessionStore(tmp_path / "s")
    rec = record("abc", 5, messages=[{"role": "user", "content": "hi"}])
    store.save(rec)
    loaded = store.load("abc")
    assert loaded["messages"] == rec["messages"]
    assert loaded["updated_at"] == 5


def test_saved_file_carries_the_format(tmp_path):
    store = sessions.SessionStore(tmp_path / "s")
    path = store.save(record("abc"))
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == sessions.FORMAT


def test_save_leaves_no_temp_file(tmp_path):
    store = sessions.SessionStore(tmp_path / "s")
    store.save(record("abc"))
    store.save(record("abc", 2))
    assert sorted(p.name for p in (tmp_path / "s").iterdir()) == ["abc.json"]


def test_list_is_newest_first(tmp_path):
    store = sessions.SessionStore(tmp_path)
    store.save(record("old", 1))
    store.save(record("new", 9))
    store.save(record("mid", 5))
    assert [r["id"] for r in store.list()] == ["new", "mid", "old"]


def test_list_skips_bad_files_and_keeps_the_good(tmp_path):
    store = sessions.SessionStore(tmp_path)
    store.save(record("good", 3))
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "array.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "oldfmt.json").write_text(
        json.dumps({"id": "oldfmt", "format": sessions.FORMAT + 1}),
        encoding="utf-8")
    (tmp_path / "nofmt.json").write_text(
        json.dumps({"id": "nofmt"}), encoding="utf-8")
    # Renamed by hand: the id inside does not match the file name.
    store.save(record("other", 4))
    (tmp_path / "other.json").rename(tmp_path / "renamed.json")
    assert [r["id"] for r in store.list()] == ["good"]


def test_list_of_a_missing_folder_is_empty(tmp_path):
    assert sessions.SessionStore(tmp_path / "nope").list() == []


def test_load_of_a_missing_id_is_none(tmp_path):
    store = sessions.SessionStore(tmp_path)
    assert store.load("ghost") is None
    # An id that cleans to nothing is also just "not there".
    assert store.load("../..") is None


def test_delete_returns_true_then_false(tmp_path):
    store = sessions.SessionStore(tmp_path)
    store.save(record("abc"))
    assert store.delete("abc") is True
    assert store.load("abc") is None
    assert store.delete("abc") is False
    assert store.delete("...") is False


def test_path_refuses_an_id_that_cleans_to_nothing(tmp_path):
    store = sessions.SessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.path("../")
    with pytest.raises(ValueError):
        store.path("")


def test_path_stays_inside_the_folder(tmp_path):
    store = sessions.SessionStore(tmp_path / "s")
    path = store.path("../evil")
    assert path.parent == tmp_path / "s"
    assert path.name == "evil.json"
    saved = store.save(record("../evil"))
    assert saved.parent == tmp_path / "s"
    assert not (tmp_path / "evil.json").exists()


def test_safe_name_keeps_only_name_characters():
    assert sessions.safe_name("a-b_c1") == "a-b_c1"
    assert sessions.safe_name("../x.y/z") == "xyz"
    assert sessions.safe_name("../") == ""


def test_save_raises_oserror_when_the_folder_cannot_be_made(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    store = sessions.SessionStore(blocker / "sub")
    with pytest.raises(OSError):
        store.save(record("abc"))


def test_sessions_dir_and_store_for_use_the_agent_subfolder(tmp_path, monkeypatch):
    monkeypatch.undo()  # drop conftest's temporary store_for
    assert sessions.store_for(None) is None
    store = sessions.store_for(str(tmp_path))
    assert store.folder == tmp_path / ".ovat" / "sessions" / "agent"
    assert sessions.sessions_dir(str(tmp_path)) == tmp_path / ".ovat" / "sessions"


# ---- Session.snapshot and Session.truncate --------------------------------

def make_conversation():
    session = Session(system_prompt="sys")
    session.add_user("q1")
    session.add_assistant(tool_calls=[{"id": "c1"}])
    session.add_tool_result("c1", "result")
    session.add_assistant("a1")
    session.add_user("q2")
    session.add_assistant("a2")
    return session


def test_snapshot_is_a_copy():
    session = make_conversation()
    snap = session.snapshot()
    snap.append({"role": "user", "content": "extra"})
    snap.clear()
    assert len(session.messages) == 7


def test_truncate_cuts_before_a_user_message():
    session = make_conversation()
    session.truncate(5)
    assert [m["content"] for m in session.messages if m["role"] == "user"] == ["q1"]
    assert len(session.messages) == 5


def test_truncate_at_the_end_is_a_no_op():
    session = make_conversation()
    session.truncate(len(session.messages))
    assert len(session.messages) == 7


def test_truncate_out_of_range_raises():
    session = make_conversation()
    with pytest.raises(ValueError):
        session.truncate(99)
    with pytest.raises(ValueError):
        session.truncate(-1)
    assert len(session.messages) == 7


def test_truncate_refuses_to_split_a_tool_call_pair():
    session = make_conversation()
    # 2 is the tool result, 3 the assistant answer, 0 the system message.
    for bad in (0, 2, 3, 6):
        with pytest.raises(ValueError):
            session.truncate(bad)
    assert len(session.messages) == 7


# ---- AgentSession memory API ----------------------------------------------

def native_session(tmp_path, prompt="new prompt"):
    agent_session = s.AgentSession("w.yml", sink=[].append, cwd=str(tmp_path))
    agent_session.agent = AgentLoop(FakeLLMProvider([]), tools={},
                                    system_prompt=prompt)
    agent_session.config = types.SimpleNamespace(
        agent=types.SimpleNamespace(system_prompt=prompt))
    return agent_session


def test_restore_memory_swaps_in_the_current_system_prompt(tmp_path):
    agent_session = native_session(tmp_path)
    saved = [{"role": "system", "content": "old prompt"},
             {"role": "user", "content": "q1"},
             {"role": "assistant", "content": "a1"}]
    assert agent_session.restore_memory(saved) is True
    assert agent_session.memory() == [
        {"role": "system", "content": "new prompt"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"}]


def test_restore_memory_adds_the_prompt_when_none_was_saved(tmp_path):
    agent_session = native_session(tmp_path)
    saved = [{"role": "user", "content": "q1"},
             {"role": "assistant", "content": "a1"}]
    agent_session.restore_memory(saved)
    assert agent_session.memory()[0] == {"role": "system", "content": "new prompt"}
    assert [m["role"] for m in agent_session.memory()] == [
        "system", "user", "assistant"]


def test_restore_memory_does_not_alias_the_input(tmp_path):
    agent_session = native_session(tmp_path)
    saved = [{"role": "user", "content": "q1"}]
    agent_session.restore_memory(saved)
    saved[0]["content"] = "changed"
    saved.append({"role": "user", "content": "more"})
    assert [m["content"] for m in agent_session.memory()[1:]] == ["q1"]


def test_memory_returns_a_copy(tmp_path):
    agent_session = native_session(tmp_path)
    agent_session.restore_memory([{"role": "user", "content": "q1"}])
    agent_session.memory().append({"role": "user", "content": "sneaky"})
    assert len(agent_session.memory()) == 2


def test_truncate_memory_cuts_at_a_user_boundary(tmp_path):
    agent_session = native_session(tmp_path)
    agent_session.restore_memory([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"}])
    assert agent_session.truncate_memory(3) is True
    assert [m["content"] for m in agent_session.memory()] == [
        "new prompt", "q1", "a1"]


def test_truncate_memory_raises_off_a_boundary(tmp_path):
    agent_session = native_session(tmp_path)
    agent_session.restore_memory([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"}])
    with pytest.raises(ValueError):
        agent_session.truncate_memory(2)
    assert len(agent_session.memory()) == 3


def test_framework_engine_has_no_reachable_memory(tmp_path):
    agent_session = s.AgentSession("w.yml", sink=[].append, cwd=str(tmp_path))
    agent_session.agent = types.SimpleNamespace()
    assert agent_session.memory() is None
    assert agent_session.restore_memory([{"role": "user", "content": "q"}]) is False
    assert agent_session.truncate_memory(0) is False


# ---- /compact -----------------------------------------------------------------

def _talked(agent_session):
    agent_session.agent.session.add_user("q1")
    agent_session.agent.session.add_assistant(
        content=None, tool_calls=[{"id": "c1", "type": "function", "function": {
            "name": "search_docs", "arguments": '{"query": "leave"}'}}])
    agent_session.agent.session.add_tool_result("c1", "20 days")
    agent_session.agent.session.add_assistant(content="20 days")


def test_compact_asks_for_a_summary_without_tools_and_keeps_only_that(tmp_path):
    from tests.conftest import reply
    agent_session = native_session(tmp_path)
    agent_session.agent.llm = FakeLLMProvider([reply(
        "stop", "<think>hm</think><summary>They asked about leave.</summary>")])
    _talked(agent_session)
    assert agent_session.compact("keep the numbers") == "They asked about leave."
    (call,) = agent_session.agent.llm.calls
    assert call["tools"] is None
    sent = call["messages"][-1]["content"]
    assert "keep the numbers" in sent
    # the tool call and its result travel as text, never cut apart
    assert "[calls search_docs" in sent and "20 days" in sent
    memory = agent_session.memory()
    assert [m["role"] for m in memory] == ["system", "user", "assistant"]
    assert memory[0]["content"] == "new prompt"
    assert "They asked about leave." in memory[1]["content"]
    assert "<previous_user_message>\nq1\n" in memory[1]["content"]
    assert agent_session.busy is False


def test_a_second_compaction_does_not_nest_the_first_envelope(tmp_path):
    from tests.conftest import reply
    agent_session = native_session(tmp_path)
    agent_session.agent.llm = FakeLLMProvider([
        reply("stop", "<summary>first</summary>"),
        reply("stop", "<summary>second</summary>")])
    _talked(agent_session)
    agent_session.compact()
    agent_session.agent.session.add_user("q2")
    agent_session.agent.session.add_assistant(content="a2")
    agent_session.compact()
    envelope = agent_session.memory()[1]["content"]
    assert envelope.count("You are continuing a trajectory") == 1
    assert "<previous_user_message>\nq2\n" in envelope
    assert "second" in envelope


def test_a_summary_without_tags_is_still_taken(tmp_path):
    from tests.conftest import reply
    agent_session = native_session(tmp_path)
    agent_session.agent.llm = FakeLLMProvider([reply("stop", "just words")])
    _talked(agent_session)
    assert agent_session.compact() == "just words"


@pytest.mark.parametrize("bad, said", [
    (lambda r: r("stop", "  "), "empty summary"),
    (lambda r: r("length", "<summary>cut"), "max_tokens"),
    (lambda r: r("tool_calls", "x", [object()]), "tool call"),
])
def test_a_bad_summary_leaves_the_memory_alone(tmp_path, bad, said):
    from ovat.agent.compaction import CompactionFailed
    from tests.conftest import reply
    agent_session = native_session(tmp_path)
    agent_session.agent.llm = FakeLLMProvider([bad(reply)])
    _talked(agent_session)
    before = agent_session.memory()
    with pytest.raises(CompactionFailed, match=said):
        agent_session.compact()
    assert agent_session.memory() == before
    assert agent_session.busy is False


def test_a_model_error_leaves_the_memory_alone(tmp_path):
    from ovat.agent.compaction import CompactionFailed

    class Down(FakeLLMProvider):
        def chat(self, messages, tools=None):
            raise ConnectionError("OVMS is not running")

    agent_session = native_session(tmp_path)
    agent_session.agent.llm = Down([])
    _talked(agent_session)
    before = agent_session.memory()
    with pytest.raises(CompactionFailed, match="OVMS is not running"):
        agent_session.compact()
    assert agent_session.memory() == before and agent_session.busy is False


def test_compact_is_refused_on_a_framework_engine_and_during_a_turn(tmp_path):
    from ovat.agent.compaction import CompactionFailed
    agent_session = s.AgentSession("w.yml", sink=[].append, cwd=str(tmp_path))
    agent_session.agent = types.SimpleNamespace()
    with pytest.raises(CompactionFailed, match="native engine"):
        agent_session.compact()
    busy = native_session(tmp_path)
    busy.busy = True
    with pytest.raises(CompactionFailed, match="turn to end"):
        busy.compact()
    assert busy.busy is True            # not ours to clear
