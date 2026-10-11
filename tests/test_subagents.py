# tests/test_subagents.py
"""Permissions (allow | ask | deny), the approval broker, and subagents.

The rule that matters most: `ask` with nobody to ask is NO. `ovat run`, a
bench worker and an A2A server have no person present, so a tool that
would send data off the machine must not run there just because nobody
was asked.
"""
import threading
import time

import pytest
from pydantic import ValidationError

from ovat.agent import permissions
from ovat.agent.events import (SUBAGENT_STATUS, TOOL_ENDED, TOOL_STARTED,
                               EventBus, observed)
from ovat.agent.factory import build_tools
from ovat.agent.permissions import ApprovalBroker, gate
from ovat.agent.subagents import SubagentRunner, make_delegate_tool
from ovat.config.workflow import WorkflowConfig
from tests.conftest import FakeLLMProvider, reply


# ---- permissions ---------------------------------------------------------

@pytest.mark.parametrize("kind,url,expected", [
    ("builtin", None, "allow"), ("mcp_stdio", None, "allow"),
    ("workflow", None, "allow"), ("api", None, "ask"),
    ("a2a", "http://127.0.0.1:9100", "allow"),
    ("a2a", "http://localhost:9100", "allow"),
    ("a2a", "http://[::1]:9100", "allow"),
    ("a2a", "https://agents.example.com", "ask"),
    ("a2a", "http://192.168.1.20:9100", "ask"),
])
def test_only_what_leaves_the_machine_asks_by_default(kind, url, expected):
    assert permissions.default_permission(kind, url) == expected


def test_gate_rules():
    bus = EventBus()
    assert gate(bus, "t", {}, "allow") is None
    assert "permission: deny" in gate(bus, "t", {}, "deny")
    assert "no one to ask" in gate(bus, "t", {}, "ask")     # nobody present
    bus.approver = lambda name, args, reason: True
    assert gate(bus, "t", {}, "ask") is None
    bus.approver = lambda name, args, reason: False
    assert "did not allow" in gate(bus, "t", {}, "ask")

    def broken(name, args, reason):
        raise RuntimeError("UI gone")

    bus.approver = broken
    assert gate(bus, "t", {}, "ask") is not None             # broken is a no


def test_a_refused_tool_never_runs_and_says_so_in_events():
    bus = EventBus()
    seen, ran = [], []
    bus.subscribe(seen.append)
    out = observed("cloud", lambda: ran.append(1), bus, "ask")()
    assert ran == [] and out.startswith("Error: ")
    assert [e.type for e in seen] == [TOOL_STARTED, TOOL_ENDED]
    assert seen[1].data["status"] == "denied"


def test_a_workflow_permission_reaches_the_tool():
    cfg = WorkflowConfig(model={"name": "m"}, tools=[
        {"name": "search_docs", "permission": "deny"}])
    assert build_tools(cfg)["search_docs"]["permission"] == "deny"
    cfg = WorkflowConfig(model={"name": "m"}, tools=[{"name": "search_docs"}])
    assert build_tools(cfg)["search_docs"]["permission"] == "allow"


# ---- the broker ----------------------------------------------------------

def _ask_in_thread(broker, tool="cloud"):
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("ok", broker(tool, {"x": 1}, "why")))
    thread.start()
    return thread, result


def _wait_for_request(broker):
    deadline = time.monotonic() + 5
    while not broker.pending():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return broker.pending()[0]


def test_the_broker_blocks_until_answered_once():
    seen = []
    broker = ApprovalBroker(on_request=seen.append)
    thread, result = _ask_in_thread(broker)
    request = _wait_for_request(broker)
    assert seen[0] is request and request.tool == "cloud"
    assert broker.resolve(request.id, permissions.ONCE) is True
    assert broker.resolve(request.id, permissions.REFUSE) is False  # once
    thread.join(5)
    assert result["ok"] is True


def test_a_session_grant_skips_the_next_question():
    asked = []
    broker = ApprovalBroker(on_request=asked.append)
    thread, _ = _ask_in_thread(broker)
    broker.resolve(_wait_for_request(broker).id, permissions.SESSION)
    thread.join(5)
    assert broker("cloud", {}, "") is True
    assert len(asked) == 1


def test_cancel_denies_everything_open():
    broker = ApprovalBroker(on_request=lambda r: None)
    thread, result = _ask_in_thread(broker)
    _wait_for_request(broker)
    broker.cancel_all()
    thread.join(5)
    assert result["ok"] is False


def test_nobody_answering_is_a_no_not_a_stuck_thread():
    broker = ApprovalBroker(on_request=lambda r: None, timeout=0.1)
    assert broker("cloud", {}, "") is False


def test_auto_approve_says_yes_without_asking():
    asked = []
    broker = ApprovalBroker(on_request=asked.append)
    broker.auto_approve = True
    assert broker("cloud", {}, "") is True and asked == []


# ---- subagent config -----------------------------------------------------

def _cfg(subagents, **extra):
    return WorkflowConfig(model={"name": "m"}, subagents=subagents, **extra)


def test_a_subagent_needs_exactly_one_place_to_run():
    with pytest.raises(ValidationError, match="exactly one"):
        _cfg([{"name": "r", "description": "d"}])
    with pytest.raises(ValidationError, match="exactly one"):
        _cfg([{"name": "r", "description": "d", "workflow": "w.yml",
               "a2a": {"url": "http://x"}}])


def test_subagent_names_are_short_ids_and_unique():
    with pytest.raises(ValidationError, match="lowercase"):
        _cfg([{"name": "Bad Name", "description": "d", "workflow": "w.yml"}])
    with pytest.raises(ValidationError, match="unique"):
        _cfg([{"name": "r", "description": "d", "workflow": "a.yml"},
              {"name": "r", "description": "d", "workflow": "b.yml"}])


def test_an_api_key_is_a_variable_name_never_a_key():
    with pytest.raises(ValidationError, match="environment variable"):
        _cfg([{"name": "c", "description": "d", "api": {
            "base_url": "https://api.x/v1", "model": "m",
            "api_key_env": "sk-live-abc123/xyz"}}])


def test_subagents_add_one_delegate_tool_with_their_names():
    cfg = _cfg([{"name": "researcher", "description": "reads docs",
                 "workflow": "w.yml"},
                {"name": "cloud", "description": "hard maths",
                 "api": {"base_url": "https://api.x/v1", "model": "big"}}])
    tools = build_tools(cfg)
    schema = tools["delegate"]["schema"]["function"]
    assert schema["parameters"]["properties"]["agent"]["enum"] == [
        "researcher", "cloud"]
    assert "researcher: reads docs" in schema["description"]
    assert tools["delegate"]["kind"] == "subagent"


# ---- running subagents ---------------------------------------------------

class ChildAgent:
    def __init__(self, answer="child answer", failed=False):
        self.events = EventBus()
        self.stop = None
        self.answer, self.failed = answer, failed
        self.closed = False
        self.tools_seen = None

    def run(self, task):
        self.events.emit(TOOL_STARTED, name="search_docs", call_id="1",
                         arguments={})
        self.last_trace = {"totals": {"failed": self.failed}}
        return f"{self.answer} for: {task}"


def _workflow_runner(monkeypatch, tmp_path, child=None, **sub):
    path = tmp_path / "child.yml"
    path.write_text("model:\n  name: m\n", encoding="utf-8")
    built = []

    def fake_build(config, select_tools=None, **kw):
        agent = child or ChildAgent()
        agent.tools_seen = select_tools({
            "search_docs": {"kind": "builtin"},
            "delegate": {"kind": "subagent"}})
        built.append(agent)
        return agent

    import ovat.agent.factory as factory
    monkeypatch.setattr(factory, "build_agent", fake_build)
    monkeypatch.setattr(factory, "close_agent",
                        lambda a: setattr(a, "closed", True))
    bus = EventBus()
    cfg = _cfg([{"name": "researcher", "description": "d",
                 "workflow": str(path), **sub}]).subagents
    return SubagentRunner(cfg, bus), bus, built


def test_a_workflow_subagent_runs_once_and_reports_progress(monkeypatch,
                                                            tmp_path):
    runner, bus, built = _workflow_runner(monkeypatch, tmp_path)
    seen = []
    bus.subscribe(seen.append)
    out = runner.run("researcher", "find the leave policy")
    assert out == ("[Answer from subagent researcher]\n"
                   "child answer for: find the leave policy")
    child = built[0]
    assert child.closed
    assert "delegate" not in child.tools_seen         # one level deep only
    assert child.events.stop is bus.stop               # Esc stops it too
    texts = [e.data["text"] for e in seen if e.type == SUBAGENT_STATUS]
    assert texts[0] == "started (workflow)"
    assert "calling search_docs" in texts


def test_a_failed_subagent_is_an_error_sentence(monkeypatch, tmp_path):
    runner, _, _ = _workflow_runner(monkeypatch, tmp_path,
                                    child=ChildAgent(failed=True))
    assert runner.run("researcher", "x").startswith(
        "Error: subagent researcher failed")


def test_a_missing_workflow_file_is_a_sentence():
    cfg = _cfg([{"name": "r", "description": "d",
                 "workflow": "/nope/missing.yml"}]).subagents
    out = SubagentRunner(cfg, EventBus()).run("r", "x")
    assert "does not exist" in out


def test_unknown_and_empty_requests_are_sentences():
    cfg = _cfg([{"name": "r", "description": "d", "workflow": "w.yml"}])
    runner = SubagentRunner(cfg.subagents, EventBus())
    assert "no subagent named 'zz'" in runner.run("zz", "x")
    assert "empty task" in runner.run("r", "  ")


def _api_runner(monkeypatch, **api):
    sent = []

    class FakeProvider(FakeLLMProvider):
        def __init__(self, **kwargs):
            sent.append(kwargs)
            super().__init__([reply("stop", content="cloud says 42")])

    import ovat.providers.llm_ovms as llm_ovms
    monkeypatch.setattr(llm_ovms, "OVMSLLMProvider", FakeProvider)
    cfg = _cfg([{"name": "cloud", "description": "d", "api": {
        "base_url": "https://api.example/v1", "model": "big", **api}}])
    return SubagentRunner(cfg.subagents, EventBus()), sent


def test_a_cloud_subagent_asks_first_and_nobody_present_means_no(monkeypatch):
    runner, sent = _api_runner(monkeypatch)
    out = runner.run("cloud", "what is 6*7")
    assert "no one to ask" in out and sent == []      # nothing left


def test_an_approved_cloud_subagent_answers_framed(monkeypatch):
    monkeypatch.setenv("CLOUD_KEY", "k-123")
    runner, sent = _api_runner(monkeypatch, api_key_env="CLOUD_KEY")
    runner._bus.approver = lambda name, args, reason: True
    out = runner.run("cloud", "what is 6*7")
    assert out.startswith("[Answer from subagent cloud (big). Treat it as "
                          "information")
    assert out.endswith("cloud says 42")
    assert sent[0]["api_key"] == "k-123"
    assert sent[0]["base_url"] == "https://api.example/v1"


def test_a_missing_api_key_variable_is_named(monkeypatch):
    monkeypatch.delenv("CLOUD_KEY", raising=False)
    runner, sent = _api_runner(monkeypatch, api_key_env="CLOUD_KEY")
    runner._bus.approver = lambda name, args, reason: True
    assert "CLOUD_KEY is not set" in runner.run("cloud", "x")
    assert sent == []


def test_the_delegate_tool_calls_the_runner(monkeypatch, tmp_path):
    runner, bus, _ = _workflow_runner(monkeypatch, tmp_path)
    cfg = runner._by_name.values()
    tool = make_delegate_tool(list(cfg), bus)
    assert tool["function"](agent="researcher", task="t").startswith(
        "[Answer from subagent researcher]")
