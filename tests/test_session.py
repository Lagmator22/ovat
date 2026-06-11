# tests/test_session.py
"""Tests for Session, the conversation memory.

Note to myself: the headline test is the save then load round trip. If that
passes, I know a chat can survive a restart with its history intact.
"""
from ovat.agent.session import Session


def test_system_prompt_goes_first():
    s = Session(system_prompt="You are helpful.")
    assert s.messages[0] == {"role": "system", "content": "You are helpful."}


def test_empty_session_has_no_messages():
    assert Session().messages == []


def test_messages_keep_their_order():
    s = Session(system_prompt="sys")
    s.add_user("hello")
    s.add_assistant(content="hi there")
    roles = [m["role"] for m in s.messages]
    assert roles == ["system", "user", "assistant"]


def test_tool_result_pairs_with_its_call_id():
    s = Session()
    s.add_assistant(tool_calls=[{"id": "tc_1", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}])
    s.add_tool_result("tc_1", "the answer")
    assert s.messages[-1] == {"role": "tool", "tool_call_id": "tc_1",
                              "content": "the answer"}


def test_save_then_load_is_identical(tmp_path):
    s = Session(system_prompt="sys")
    s.add_user("what is up")
    s.add_assistant(tool_calls=[{"id": "tc_1", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}])
    s.add_tool_result("tc_1", "result text")
    s.add_assistant(content="final answer")

    path = tmp_path / "session.json"
    s.save(str(path))
    loaded = Session.load(str(path))
    assert loaded.messages == s.messages
