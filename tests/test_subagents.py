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

    def fake_build(config, select_tools=None, events=None, **kw):
        agent = child or ChildAgent()
        if events is not None:
            agent.events = events
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
    bus.asker = lambda question, choices: "sqlite"
    seen = []
    bus.subscribe(seen.append)
    out = runner.run("researcher", "find the leave policy")
    assert out == ("[Answer from subagent researcher]\n"
                   "child answer for: find the leave policy")
    child = built[0]
    assert child.closed
    assert "delegate" not in child.tools_seen         # one level deep only
    assert child.events.stop.is_set() is False
    bus.stop.set()
    assert child.events.stop.is_set() is True          # Esc stops it too
    bus.stop.clear()
    assert child.events.asker is bus.asker
    assert child.events.asker("Which database?", []) == "sqlite"
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


def test_child_events_carry_call_id_and_totals(monkeypatch, tmp_path):
    from ovat.agent.events import (CURRENT_CALL_ID, SUBAGENT_DONE,
                                   SUBAGENT_EVENT, SUBAGENT_STARTED)
    runner, bus, built = _workflow_runner(monkeypatch, tmp_path)
    events = []
    bus.subscribe(events.append)

    token = CURRENT_CALL_ID.set("parent-call-123")
    try:
        out = runner.run("researcher", "find docs")
    finally:
        CURRENT_CALL_ID.reset(token)

    assert out.startswith("[Answer from subagent researcher]")
    types = [e.type for e in events]
    assert SUBAGENT_STARTED in types
    assert SUBAGENT_EVENT in types
    assert SUBAGENT_DONE in types

    start_ev = next(e for e in events if e.type == SUBAGENT_STARTED)
    assert start_ev.data["call_id"] == "parent-call-123"
    assert start_ev.data["agent"] == "researcher"
    assert start_ev.data["kind"] == "workflow"

    event_ev = next(e for e in events if e.type == SUBAGENT_EVENT)
    assert event_ev.data["call_id"] == "parent-call-123"
    assert event_ev.data["agent"] == "researcher"
    assert event_ev.data["event"]["type"] == "tool_started"

    done_ev = next(e for e in events if e.type == SUBAGENT_DONE)
    assert done_ev.data["call_id"] == "parent-call-123"
    assert done_ev.data["agent"] == "researcher"
    assert done_ev.data["status"] == "ok"
    assert "duration_s" in done_ev.data


def test_cancel_child_stops_one_without_stopping_parent_or_sibling(monkeypatch, tmp_path):
    from ovat.agent.events import CURRENT_CALL_ID
    runner, bus, built = _workflow_runner(monkeypatch, tmp_path)

    # Simulate child that checks stop inside run
    class CancellableChild:
        def __init__(self):
            self.events = EventBus()
            self.closed = False
            self.last_trace = {}

        def run(self, task):
            # Check if this child's stop is set
            if self.events.stop.is_set():
                return "stopped early"
            # Simulate cancel arriving mid-run
            runner.cancel_child("call-c1")
            if self.events.stop.is_set():
                return "cancelled mid-flight"
            return "finished"

    c1 = CancellableChild()
    built_agents = [c1]

    def fake_build(config, select_tools=None, events=None, **kw):
        agent = built_agents.pop(0)
        if events is not None:
            agent.events = events
        return agent

    import ovat.agent.factory as factory
    monkeypatch.setattr(factory, "build_agent", fake_build)

    token = CURRENT_CALL_ID.set("call-c1")
    try:
        out = runner.run("researcher", "work")
    finally:
        CURRENT_CALL_ID.reset(token)

    assert "cancelled mid-flight" in out
    # Child's stop was set, but parent bus.stop was NOT set
    assert bus.stop.is_set() is False


def test_cancelling_a_remote_child_stops_its_wait(monkeypatch):
    """An a2a subagent's call never saw the child's stop flag, so cancelling
    that one child left it waiting for the remote reply."""
    pytest.importorskip("a2a")
    from ovat.agent.events import CURRENT_CALL_ID
    from ovat.tools import a2a_call
    seen = {}

    def fake_call(task, peer, name, url, bus, token_env, missing,
                  return_totals=False, stop=None):
        seen["stop"] = stop
        runner.cancel_child("call-r")
        assert stop.is_set() and not bus.stop.is_set()
        return "Error: the request was stopped.", {}
    monkeypatch.setattr(a2a_call, "a2a_call_impl", fake_call)
    bus = EventBus()
    runner = SubagentRunner(_cfg([{"name": "remote", "description": "d",
                                   "a2a": {"url": "http://127.0.0.1:9"}}]
                                 ).subagents, bus)
    token = CURRENT_CALL_ID.set("call-r")
    try:
        runner.run("remote", "work")
    finally:
        CURRENT_CALL_ID.reset(token)
    assert seen["stop"] is not None


def test_child_tool_approval_records_agent_name(monkeypatch, tmp_path):
    from ovat.agent.events import observed
    from ovat.agent.permissions import ApprovalBroker
    runner, bus, _ = _workflow_runner(monkeypatch, tmp_path)

    requests = []
    broker = ApprovalBroker(on_request=requests.append)
    bus.approver = broker

    class AskingChild:
        def __init__(self):
            self.events = EventBus()
            self.closed = False
            self.last_trace = {}

        def run(self, task):
            # Tool inside child agent wrapped in observed
            def dangerous_tool():
                return "done"
            obs = observed("shell", dangerous_tool, self.events, permission="ask")
            return obs()

    child = AskingChild()

    def fake_build(config, select_tools=None, events=None, **kw):
        if events is not None:
            child.events = events
        return child

    import ovat.agent.factory as factory
    monkeypatch.setattr(factory, "build_agent", fake_build)

    # Run in thread since broker blocks
    import threading
    t = threading.Thread(target=lambda: runner.run("researcher", "run shell"))
    t.start()
    for _ in range(50):
        if requests:
            break
        time.sleep(0.02)
    assert len(requests) == 1
    assert requests[0].tool == "shell"
    assert requests[0].agent == "researcher"
    assert requests[0].agent_call            # the delegate call it runs under
    broker.resolve(requests[0].id, "once")
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_child_session_history_list():
    import asyncio
    from ovat.agent.events import Event, SUBAGENT_EVENT, SUBAGENT_STARTED
    from ovat.tui.server.methods.session import history_list
    from ovat.tui.server.projector import Projector
    from ovat.tui.vibe.app_server.models import IdleSessionStatus, PublicSession
    from ovat.tui.vibe.app_server.protocol import SessionHistoryListParams

    emitted = []
    session = PublicSession(
        id="parent-s1",
        status=IdleSessionStatus(),
        created_at=0,
        updated_at=0,
    )
    projector = Projector(session, lambda m, p: emitted.append((m, p)))
    projector.agent_event(Event(SUBAGENT_STARTED, {"call_id": "c1", "agent": "researcher"}), {})
    # Emit child events
    projector.agent_event(Event(SUBAGENT_EVENT, {
        "call_id": "c1",
        "agent": "researcher",
        "event": {"type": "tool_started", "name": "search_docs", "arguments": {"query": "test"}},
    }), {"search_docs": "builtin"})

    class FakeCtx:
        def __init__(self, p):
            self.projector = p

    ctx = FakeCtx(projector)
    params = SessionHistoryListParams.model_validate({"sessionId": "child-c1"})
    res = asyncio.run(history_list(ctx, params))
    assert len(res.items) >= 1
    assert "search_docs" in res.items[0].detail.display.summary


def test_child_approval_marks_blocked_and_unblocks():
    from ovat.agent.events import Event, SUBAGENT_STARTED
    from ovat.agent.permissions import ApprovalBroker, ApprovalRequest
    from ovat.tui.server.approvals import Approvals
    from ovat.tui.server.projector import Projector
    from ovat.tui.vibe.app_server.models import (
        ApprovalDecisionType,
        BlockedSessionStatus,
        IdleSessionStatus,
        PublicSession,
        RunningSessionStatus,
    )
    from ovat.tui.vibe.app_server.protocol import CallbackResultParams

    emitted = []
    session = PublicSession(
        id="parent-s1",
        status=IdleSessionStatus(),
        created_at=0,
        updated_at=0,
    )
    projector = Projector(session, lambda m, p: emitted.append((m, p)))
    projector.agent_event(Event(SUBAGENT_STARTED, {"call_id": "c1", "agent": "researcher"}), {})
    assert isinstance(projector.child_sessions["child-c1"].status, RunningSessionStatus)

    broker = ApprovalBroker()

    class FakeAgent:
        def __init__(self):
            self.broker = broker

    class FakeTurns:
        active_id = "turn-1"
        def cancel_active(self): pass

    class FakeCapabilities:
        callback_kinds = ["approval"]

    class FakeRpc:
        def __init__(self):
            self.requests = []
        def request(self, method, params, callback):
            self.requests.append((method, params, callback))

    class FakeCtx:
        def __init__(self, p):
            self.projector = p
            self.agent = FakeAgent()
            self.turns = FakeTurns()
            self.capabilities = FakeCapabilities()
            self.rpc = FakeRpc()

    ctx = FakeCtx(projector)
    approvals = Approvals(ctx)

    # Subagent asks for approval
    req = ApprovalRequest(id=1, tool="shell", arguments={"cmd": "ls"},
                          reason="", agent="researcher", agent_call="c1")
    broker._open[1] = req
    approvals.request(req)

    # Check that child session is now marked blocked
    assert isinstance(projector.child_sessions["child-c1"].status, BlockedSessionStatus)
    # Check title mentions subagent
    cb_entry = ctx.rpc.requests[0][1].callback
    assert "Allow subagent researcher to run shell?" in cb_entry.title

    # Answer approval
    res_params = CallbackResultParams.model_validate({
        "sessionId": "parent-s1",
        "result": {
            "callbackId": "approval-1",
            "output": {"type": "approval", "decision": {"type": "approve"}},
        },
    })
    approvals.answer(res_params)

    # Check that child session is unblocked back to RunningSessionStatus
    assert isinstance(projector.child_sessions["child-c1"].status, RunningSessionStatus)


def test_child_sessions_save_restore_and_rewind():
    import asyncio
    from ovat.agent.events import Event, SUBAGENT_STARTED
    from ovat.tui.server import saved
    from ovat.tui.server.methods.session import session_rewind
    from ovat.tui.server.projector import Projector
    from ovat.tui.vibe.app_server._effect_models import SubagentEffectDetail, SubagentEffectInput
    from ovat.tui.vibe.utils.tool_presentation import EffectCallDisplay
    from ovat.tui.vibe.app_server.models import (
        BlockedSessionStatus,
        IdleSessionStatus,
        PublicEffectEntry,
        PublicEntryGenerationStatus,
        PublicMessageEntry,
        PublicSession,
        PublicTurn,
        RunningEffectState,
        TextContentBlock,
    )
    from ovat.tui.vibe.app_server.protocol import SessionRewindParams

    session = PublicSession(id="p1", status=IdleSessionStatus(), created_at=0, updated_at=0)
    projector = Projector(session, lambda m, p: None)
    projector.agent_event(Event(SUBAGENT_STARTED, {"call_id": "c1", "agent": "researcher"}), {})
    projector.agent_event(Event(SUBAGENT_STARTED, {"call_id": "c2", "agent": "writer"}), {})
    # Mark c2 blocked to verify restore turns it idle
    projector.child_sessions["child-c2"] = projector.child_sessions["child-c2"].model_copy(
        update={"status": BlockedSessionStatus(active_turn_id="t1", callback_id="cb1", reason="approval")}
    )

    class FakeAgent:
        def memory(self): return []
        def truncate_memory(self, cut): pass

    class FakeCtx:
        def __init__(self, p):
            self.projector = p
            self.agent = FakeAgent()
            self.workflow = "w.yml"
            self.info = {"engine": "native"}
            self.anchors = {"u1": 0, "u2": 1}
            self.store = None
            self.is_ready = True
            self.turns = type("Turns", (), {"busy": False})()
        def require_session(self, sid): return self.projector

    ctx = FakeCtx(projector)

    # 1. Verify saved.record includes child sessions
    rec = saved.record(ctx)
    assert "child-c1" in rec["child_sessions"]
    assert "child-c2" in rec["child_sessions"]

    # 2. Verify child_sessions_of restores non-terminal status as idle
    restored_cs = saved.child_sessions_of(rec)
    assert isinstance(restored_cs["child-c2"].status, IdleSessionStatus)

    # 3. Verify rewind only keeps child sessions referenced by kept entries
    # Add messages and a delegate effect for c1 only
    u1 = PublicMessageEntry(
        id="u1", session_id="p1", created_at=0, updated_at=0,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        role="user", content=[TextContentBlock(text="run c1")]
    )
    eff1 = PublicEffectEntry(
        id="eff1", session_id="p1", created_at=1, updated_at=1,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="delegate",
        state=RunningEffectState(),
        detail=SubagentEffectDetail(
            tool_name="delegate",
            agent_name="researcher",
            display=EffectCallDisplay(summary="researcher: t", status_text="running"),
            input=SubagentEffectInput(task="t", agent="researcher"),
            child_session_id="child-c1"
        )
    )
    u2 = PublicMessageEntry(
        id="u2", session_id="p1", created_at=2, updated_at=2,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        role="user", content=[TextContentBlock(text="run c2")]
    )
    eff2 = PublicEffectEntry(
        id="eff2", session_id="p1", created_at=3, updated_at=3,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="delegate",
        state=RunningEffectState(),
        detail=SubagentEffectDetail(
            tool_name="delegate",
            agent_name="writer",
            display=EffectCallDisplay(summary="writer: t2", status_text="running"),
            input=SubagentEffectInput(task="t2", agent="writer"),
            child_session_id="child-c2"
        )
    )
    projector.history = [u1, eff1, u2, eff2]
    projector.turns = [PublicTurn.model_validate({"id": "t1", "sessionId": "p1", "inputEntryId": "u1", "status": "completed", "startedAt": 0, "completedAt": 0})]

    # Rewind to u2 (so u1 and eff1 remain, while u2 and eff2 are dropped)
    res = asyncio.run(session_rewind(ctx, SessionRewindParams(session_id="p1", entry_id="u2", inplace=True)))
    # Kept history has u1 and eff1
    assert len(projector.history) == 2
    # Only child-c1 is kept; child-c2 is dropped because eff2 was rewound!
    assert "child-c1" in projector.child_sessions
    assert "child-c2" not in projector.child_sessions




def test_a_child_stop_flag_wakes_a_waiter_when_the_parent_is_set():
    """is_set() looked at the parent, but threading.Event.wait() reads its
    own flag and slept through a parent cancel."""
    from ovat.agent.events import CombinedStopEvent
    parent = threading.Event()
    child = CombinedStopEvent(parent)
    threading.Timer(0.1, parent.set).start()
    started = time.monotonic()
    assert child.wait(5) is True
    assert time.monotonic() - started < 1
    assert CombinedStopEvent(threading.Event()).wait(0.1) is False


def test_a_child_that_reported_no_tokens_shows_none_not_zero():
    """Absent is not zero. A child whose engine reported no token counts
    (an API that sends no usage, a framework engine) was shown as
    "0 tokens", which reads as measured."""
    from ovat.agent.events import Event, SUBAGENT_DONE, SUBAGENT_STARTED
    from ovat.tui.server.projector import Projector
    from ovat.tui.vibe.app_server.models import IdleSessionStatus, PublicSession
    from ovat.tui.vibe.cli.textual_ui.widgets.chat_input.subagent_list import \
        SubagentList

    def finish(totals):
        projector = Projector(PublicSession(
            id="p", status=IdleSessionStatus(), created_at=0, updated_at=0),
            lambda m, p: None)
        projector.agent_event(Event(SUBAGENT_STARTED, {
            "call_id": "c1", "agent": "researcher"}), {})
        projector.agent_event(Event(SUBAGENT_DONE, {
            "call_id": "c1", "agent": "researcher", "answer": "a",
            "status": "ok", "totals": totals}), {})
        return projector.child_sessions["child-c1"], projector.session

    def row(child):
        widget = SubagentList.__new__(SubagentList)
        widget._selected_session_id = widget._marker_session_id = None
        return widget._child_row(child).plain

    unknown, parent = finish({"turns": 1})
    assert unknown.context_usage is None
    assert "tokens" not in row(unknown)
    assert parent.token_usage is None              # nothing added
    known, parent = finish({"prompt_tokens": 30, "completion_tokens": 12})
    assert known.context_usage.total_tokens == 42
    assert "42 tokens" in row(known)
    assert parent.token_usage.total_tokens == 42


def test_an_approval_blocks_the_child_that_asked_not_its_namesake():
    """Two runs of one subagent were told apart by name only, so the
    first running one was marked blocked whichever had asked."""
    from ovat.agent.events import Event, SUBAGENT_STARTED
    from ovat.agent.permissions import ApprovalBroker, ApprovalRequest
    from ovat.tui.server.approvals import Approvals
    from ovat.tui.server.projector import Projector
    from ovat.tui.vibe.app_server.models import (BlockedSessionStatus,
                                                 IdleSessionStatus,
                                                 PublicSession,
                                                 RunningSessionStatus)

    projector = Projector(PublicSession(id="p", status=IdleSessionStatus(),
                                        created_at=0, updated_at=0),
                          lambda m, p: None)
    for call in ("c1", "c2"):
        projector.agent_event(Event(SUBAGENT_STARTED, {
            "call_id": call, "agent": "researcher"}), {})

    class Ctx:
        def __init__(self):
            self.projector = projector
            self.agent = type("A", (), {"broker": ApprovalBroker()})()
            self.turns = type("T", (), {"active_id": "turn-1"})()
            self.capabilities = type("C", (), {"callback_kinds": ["approval"]})()
            self.rpc = type("R", (), {"request": lambda *a: None})()
    Approvals(Ctx()).request(ApprovalRequest(
        id=1, tool="shell", arguments={}, agent="researcher",
        agent_call="c2"))
    assert isinstance(projector.child_sessions["child-c2"].status,
                      BlockedSessionStatus)
    assert isinstance(projector.child_sessions["child-c1"].status,
                      RunningSessionStatus)


def test_an_api_subagent_can_turn_thinking_off(monkeypatch):
    """Qwen3.5 behind OVMS thinks by default and, at temperature 0, spent
    every token on it. enable_thinking: false reaches the request."""
    monkeypatch.setenv("CLOUD_KEY", "k")
    runner, sent = _api_runner(monkeypatch, api_key_env="CLOUD_KEY",
                               enable_thinking=False)
    runner._bus.approver = lambda name, args, reason: True
    runner.run("cloud", "say PONG")
    assert sent[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}}


def test_an_api_subagent_sends_no_thinking_field_unless_asked(monkeypatch):
    monkeypatch.setenv("CLOUD_KEY", "k")
    runner, sent = _api_runner(monkeypatch, api_key_env="CLOUD_KEY")
    runner._bus.approver = lambda name, args, reason: True
    runner.run("cloud", "say PONG")
    assert sent[0]["extra_body"] is None


def test_workflow_subagent_inherits_parent_bus_context(monkeypatch, tmp_path):
    """A2A hop and seen-agent tracking live in bus.context; a workflow
    subagent must inherit a copy so it cannot reset hop limits."""
    runner, bus, built = _workflow_runner(monkeypatch, tmp_path)
    bus.context = {"a2a_hops": 2, "a2a_seen": ("agent-a", "agent-b")}
    runner.run("researcher", "check hops")
    child = built[0]
    assert child.events.context == {
        "a2a_hops": 2, "a2a_seen": ("agent-a", "agent-b")}
    assert child.events.context is not bus.context


def test_cancel_child_unblocks_pending_approval_and_question_for_that_child_only(
        monkeypatch, tmp_path):
    """Cancelling one running subagent must immediately wake its blocked
    approval or question prompt without cancelling a sibling's prompt."""
    from ovat.agent.events import CURRENT_CALL_ID, CURRENT_SUBAGENT
    from ovat.agent.questions import QuestionBroker

    runner, bus, _ = _workflow_runner(monkeypatch, tmp_path)
    approver = ApprovalBroker(on_request=lambda r: None, timeout=5.0)
    asker = QuestionBroker(on_request=lambda r: None, timeout=5.0)
    bus.approver = approver
    bus.asker = asker

    results = {}

    def child_waiting_approval():
        tok = CURRENT_SUBAGENT.set(("researcher", "call-c1"))
        try:
            results["c1_approval"] = approver("shell", {"cmd": "ls"}, "")
        finally:
            CURRENT_SUBAGENT.reset(tok)

    def sibling_waiting_approval():
        tok = CURRENT_SUBAGENT.set(("researcher", "call-c2"))
        try:
            results["c2_approval"] = approver("shell", {"cmd": "pwd"}, "")
        finally:
            CURRENT_SUBAGENT.reset(tok)

    def child_waiting_question():
        tok = CURRENT_SUBAGENT.set(("researcher", "call-c1"))
        try:
            results["c1_question"] = asker("Which env?", ["dev", "prod"])
        finally:
            CURRENT_SUBAGENT.reset(tok)

    # Register call-c1 as an active child run on the runner
    runner._child_stops["call-c1"] = threading.Event()
    runner._child_stops["call-c2"] = threading.Event()

    t1 = threading.Thread(target=child_waiting_approval)
    t2 = threading.Thread(target=sibling_waiting_approval)
    t3 = threading.Thread(target=child_waiting_question)
    t1.start()
    t2.start()
    t3.start()

    deadline = time.monotonic() + 2.0
    while len(approver.pending()) < 2 or len(asker.pending()) < 1:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert runner.cancel_child("call-c1") is True
    t1.join(timeout=0.5)
    t3.join(timeout=0.5)
    assert not t1.is_alive()
    assert not t3.is_alive()
    assert results["c1_approval"] is False
    assert results["c1_question"] is None

    # Sibling call-c2 is still waiting and unaffected
    assert len(approver.pending()) == 1
    assert approver.pending()[0].agent_call == "call-c2"
    approver.resolve(approver.pending()[0].id, permissions.ONCE)
    t2.join(timeout=0.5)
    assert results["c2_approval"] is True

