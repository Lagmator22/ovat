# tests/test_a2a_core.py
"""OVAT's own A2A logic, with no SDK and no network.

Everything in ovat/a2a/ outside sdk/ is plain Python on purpose (see the
package docstring), so it is tested here with fakes: a fake agent, a
Reporter that records what it is told, a peer that replies from a script.
"""
import asyncio
import os
import stat
import sys
import threading
import time

import pytest

from ovat.a2a import exposure, serving, tokens
from ovat.a2a.audit import AuditLog
from ovat.a2a.card import card_dict
from ovat.a2a.protocol import (PeerError, RemoteCard, RemoteReply,
                               RemoteSkill, TaskState)
from ovat.a2a.service import A2AService, Task, shape_sources
from ovat.agent.events import REMOTE_STATUS, EventBus, observed
from ovat.config.workflow import A2AServerConfig, WorkflowConfig
from ovat.tools import a2a_call


# ---- fakes ---------------------------------------------------------------

class AllowAll:
    """Admission that accepts everything (policy has its own tests)."""

    def check(self, ask):
        return None


class RefuseAll:
    def check(self, ask):
        return "no thanks (test)"


class Recorder:
    """A Reporter that writes down every call, in order."""

    def __init__(self):
        self.calls = []

    async def accepted(self):
        self.calls.append(("accepted",))

    async def working(self, text, event=None):
        self.calls.append(("working", text, event))

    async def answer(self, text, sources):
        self.calls.append(("answer", text, sources))

    async def completed(self, trace):
        self.calls.append(("completed", trace))

    async def failed(self, text, trace=None):
        self.calls.append(("failed", text))

    async def rejected(self, text):
        self.calls.append(("rejected", text))

    def kinds(self):
        return [c[0] for c in self.calls]


class FakeAgent:
    """Shaped like a built agent: run(), last_trace, events."""

    def __init__(self, answer="the answer", failed=False, raises=None,
                 sources=(), delay=0.0, tool=True):
        self.events = EventBus()
        self.answer, self.failed, self.raises = answer, failed, raises
        self.sources, self.delay, self.tool = list(sources), delay, tool
        self.questions = []
        self.seen_context = None
        self.closed = False
        self._search = observed("search_docs", lambda query: "chunks",
                                self.events)

    def run(self, text):
        self.questions.append(text)
        self.seen_context = dict(self.events.context)
        if self.tool:
            self._search(query=text)
        end = time.monotonic() + self.delay
        while time.monotonic() < end:
            if self.events.stop.is_set():
                return "Error: cancelled by the caller."
            time.sleep(0.01)
        if self.raises:
            raise self.raises
        self.last_trace = {"turns": [{"tool_calls": [
            {"name": "search_docs", "sources": self.sources}]}],
            "totals": {"failed": self.failed, "turns": 1}}
        return self.answer


def _service(agent_factory=FakeAgent, admission=None, audit=None, **settings):
    built = []

    def build():
        agent = agent_factory()
        built.append(agent)
        return agent

    def close(agent):
        agent.closed = True

    svc = A2AService(build, A2AServerConfig(**settings),
                     admission=admission or AllowAll(), audit=audit,
                     close=close)
    return svc, built


def _task(text="q", caller="planner", context="c1", task_id="t1", **kw):
    return Task(task_id=task_id, context_id=context, caller=caller,
                text=text, **kw)


def _run(svc, task, reporter=None):
    reporter = reporter or Recorder()
    asyncio.run(svc.run(task, reporter))
    return reporter


# ---- exposure ------------------------------------------------------------

def _tools(**kinds):
    return {name: {"schema": {"function": {"description": f"{name} tool"}},
                   "function": lambda **k: None, "kind": kind}
            for name, kind in kinds.items()}


def test_default_exposure_hides_paths_mcp_and_onward_agents():
    exp = exposure.decide(_tools(search_docs="builtin", transcribe="builtin",
                                 describe_image="builtin",
                                 fs_write="mcp_stdio", other="a2a"), None)
    assert exp.exposed == ["search_docs"]
    assert set(exp.hidden) == {"transcribe", "describe_image", "fs_write",
                               "other"}


def test_an_explicit_list_wins_and_warns_about_path_tools():
    exp = exposure.decide(_tools(search_docs="builtin", transcribe="builtin"),
                          ["transcribe"])
    assert exp.exposed == ["transcribe"]
    assert "search_docs" in exp.hidden
    assert any("file path" in w for w in exp.warnings)


def test_an_explicit_list_naming_a_missing_tool_is_an_error():
    with pytest.raises(ValueError, match="nope"):
        exposure.decide(_tools(search_docs="builtin"), ["nope"])


def test_the_coding_tools_are_never_offered_to_a_remote_caller():
    names = ("read_file", "write_file", "edit", "grep", "shell")
    tools = _tools(search_docs="builtin", **{n: "builtin" for n in names})
    exp = exposure.decide(tools, None)
    assert exp.exposed == ["search_docs"]
    assert set(names) <= set(exp.hidden)
    # Not even a line the owner wrote on purpose can hand them over.
    for name in names:
        with pytest.raises(ValueError, match="never offered"):
            exposure.decide(tools, ["search_docs", name])
    # An MCP server's own tool that happens to be called shell is a
    # different thing and follows the MCP rule.
    mcp = exposure.decide(_tools(shell="mcp_stdio"), ["shell"])
    assert mcp.exposed == ["shell"]


def test_select_builds_the_agent_with_exposed_tools_only():
    tools = _tools(search_docs="builtin", transcribe="builtin")
    assert list(exposure.decide(tools, None).select(tools)) == ["search_docs"]


# ---- card ----------------------------------------------------------------

def test_the_card_never_carries_the_system_prompt():
    cfg = WorkflowConfig(model={"name": "m"}, agent={
        "system_prompt": "SECRET instructions about payroll"})
    card = card_dict(cfg, A2AServerConfig(), _tools(search_docs="builtin"),
                     "http://127.0.0.1:9100/", "docs", "1.2.0")
    assert "SECRET" not in str(card)
    assert card["name"] == "docs"
    assert [s["id"] for s in card["skills"]] == ["search_docs"]
    assert all(s["tags"] for s in card["skills"])


def test_a_card_with_no_tools_still_has_a_skill():
    card = card_dict(WorkflowConfig(model={"name": "m"}), A2AServerConfig(),
                     {}, "http://x/", "n", "1")
    assert card["skills"] and card["skills"][0]["tags"]


def test_yaml_skills_and_description_replace_the_generated_ones():
    server = A2AServerConfig(description="Handbook answers.", skills=[
        {"id": "hb", "name": "Handbook", "description": "Policy Q&A"}])
    card = card_dict(WorkflowConfig(model={"name": "m"}), server,
                     _tools(search_docs="builtin"), "http://x/", "n", "1")
    assert card["description"] == "Handbook answers."
    assert card["skills"][0]["id"] == "hb"


# ---- tokens and audit ----------------------------------------------------

def test_the_generated_token_is_stable_until_rotated(tmp_path):
    path = tmp_path / "a2a" / "token"
    first = tokens.load_or_create(path)
    assert first and tokens.load_or_create(path) == first
    assert tokens.rotate(path) != first


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_the_token_file_is_owner_only(tmp_path):
    path = tmp_path / "token"
    tokens.load_or_create(path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_named_callers_read_their_tokens_from_the_environment(tmp_path):
    table = tokens.callers({"planner": "TOK_P", "cloud": "TOK_C"},
                           tmp_path / "t", environ={"TOK_P": "p1",
                                                    "TOK_C": "c1"})
    assert table == {"p1": "planner", "c1": "cloud"}


def test_an_unset_token_variable_is_named(tmp_path):
    with pytest.raises(ValueError, match="TOK_P"):
        tokens.callers({"planner": "TOK_P"}, tmp_path / "t", environ={})


def test_two_callers_with_one_token_are_refused(tmp_path):
    with pytest.raises(ValueError, match="same token"):
        tokens.callers({"a": "X", "b": "Y"}, tmp_path / "t",
                       environ={"X": "same", "Y": "same"})


def test_a_local_peer_gets_this_folders_server_token_and_no_one_else_does(
        tmp_path):
    """A caller with no token_env, talking to `ovat a2a serve` on this
    machine, sends the token that server generated. The check used to be
    `"localhost" in url`, which also matched http://localhost.evil.example
    and would have sent the token there."""
    token_file = tmp_path / "token"
    token_file.write_text("secret\n")
    def peer(url, env=None, environ={}):
        return tokens.peer_token(url, env, token_file, environ)
    assert peer("http://127.0.0.1:9100") == "secret"
    assert peer("http://localhost:9100/a2a") == "secret"
    assert peer("http://[::1]:9100") == "secret"
    for elsewhere in ("http://localhost.evil.example",
                      "http://evil.example/?u=127.0.0.1",
                      "http://127.0.0.1.evil.example:9100",
                      "http://user:localhost@evil.example"):
        assert peer(elsewhere) == "", elsewhere
    # a named variable always wins, and is never replaced by the file
    assert peer("http://127.0.0.1:9100", "T", {"T": "mine"}) == "mine"
    assert peer("http://127.0.0.1:9100", "T", {}) == ""
    token_file.unlink()
    assert peer("http://127.0.0.1:9100") == ""


def test_the_audit_log_appends_and_reads_back(tmp_path):
    log = AuditLog(tmp_path / "sub" / "audit.jsonl")
    log.write({"caller": "a", "answer": "x"})
    log.write({"caller": "b", "answer": "y"})
    assert [r["caller"] for r in log.tail(5)] == ["a", "b"]
    assert "time" in log.tail(1)[0]


# ---- sources -------------------------------------------------------------

def test_sources_become_file_names_even_from_windows_paths():
    paths = [r"C:\Users\devcloud\hr\leave.md", "/home/u/docs/travel.md",
             "/other/leave.md"]
    assert shape_sources(paths, "names") == ["leave.md", "travel.md"]
    assert shape_sources(paths, "full") == paths
    assert shape_sources(paths, "none") == []


# ---- the service ---------------------------------------------------------

def test_a_good_run_reports_in_order_and_logs_it(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    svc, _ = _service(lambda: FakeAgent(sources=["/p/a.md"]), audit=audit)
    rec = _run(svc, _task())
    assert rec.kinds() == ["accepted", "working", "working", "answer",
                           "completed"]
    assert rec.calls[1][1] == "calling search_docs"
    assert rec.calls[3][1:] == ("the answer", ["a.md"])
    record = audit.tail(1)[0]
    assert record["state"] == "completed"
    assert record["caller"] == "planner"
    assert record["tools"] == ["search_docs"]
    assert svc.status()["done"] == 1


def test_tool_arguments_never_reach_the_remote_caller():
    svc, _ = _service()
    rec = _run(svc, _task(text="my secret path /Users/me/x"))
    events = [c[2] for c in rec.calls if c[0] == "working"]
    assert events and all("arguments" not in e for e in events)


def test_a_failed_run_is_failed_with_the_loops_sentence():
    svc, _ = _service(lambda: FakeAgent(answer="Error: no answer",
                                        failed=True))
    rec = _run(svc, _task())
    assert rec.calls[-1] == ("failed", "Error: no answer")


def test_an_exception_is_failed_without_its_text():
    svc, _ = _service(lambda: FakeAgent(
        raises=RuntimeError("C:\\Users\\devcloud\\secret.db is locked")))
    rec = _run(svc, _task())
    kind, text = rec.calls[-1]
    assert kind == "failed"
    assert "secret" not in text and "devcloud" not in text


def test_a_refusal_builds_nothing_and_is_only_rejected(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    svc, built = _service(admission=RefuseAll(), audit=audit)
    rec = _run(svc, _task())
    assert rec.calls == [("rejected", "no thanks (test)")]
    assert built == []
    assert audit.tail(1)[0]["state"] == "rejected"


def test_one_conversation_keeps_its_agent_and_callers_never_share():
    svc, built = _service()
    _run(svc, _task(context="c1", caller="a", task_id="1"))
    _run(svc, _task(context="c1", caller="a", task_id="2"))
    _run(svc, _task(context="c1", caller="b", task_id="3"))   # same id!
    assert len(built) == 2
    assert built[0].questions == ["q", "q"]
    assert built[1].questions == ["q"]


def test_the_oldest_idle_conversation_is_closed_when_full():
    svc, built = _service(max_contexts=2)
    for i in range(3):
        _run(svc, _task(context=f"c{i}", task_id=str(i)))
    assert [a.closed for a in built] == [True, False, False]


def test_the_agent_sees_the_next_hop_and_this_server_in_the_chain():
    svc, built = _service()
    _run(svc, _task(hops=1, chain=["upstream"]))
    seen = built[0].seen_context
    assert seen["hops"] == 2
    assert seen["chain"] == ["upstream", svc.server_id]
    assert built[0].events.context == {}           # cleared afterwards


def test_a_timeout_fails_the_task_and_stops_the_agent():
    svc, built = _service(lambda: FakeAgent(delay=5.0), task_timeout_s=0.2)
    rec = _run(svc, _task())
    assert rec.calls[-1][0] == "failed"
    assert "task_timeout_s" in rec.calls[-1][1]
    assert built[0].events.stop.is_set()


def test_cancel_stops_a_running_agent():
    svc, built = _service(lambda: FakeAgent(delay=5.0))

    async def scenario():
        rec = Recorder()
        job = asyncio.ensure_future(svc.run(_task(task_id="t9"), rec))
        while not built or not built[0].questions:
            await asyncio.sleep(0.01)
        assert svc.cancel("t9") is True
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        return rec

    started = time.monotonic()
    asyncio.run(scenario())
    assert built[0].events.stop.is_set()
    assert time.monotonic() - started < 3       # it did not run 5 s
    assert svc.cancel("t9") is False             # no longer running


def test_one_model_at_a_time_and_the_second_is_told_it_waits():
    svc, built = _service(lambda: FakeAgent(delay=0.3))

    async def scenario():
        first, second = Recorder(), Recorder()
        await asyncio.gather(
            svc.run(_task(context="a", task_id="1"), first),
            svc.run(_task(context="b", task_id="2"), second))
        return first, second

    first, second = asyncio.run(scenario())
    waits = [c for r in (first, second) for c in r.calls
             if c[0] == "working" and "waiting" in c[1]]
    assert len(waits) == 1 and "1 ahead" in waits[0][1]


class ApprovalRecorder(Recorder):
    async def input_required(self, call_id, tool, args):
        self.calls.append(("input_required", call_id, tool, args))


class AsksFirst(FakeAgent):
    """Asks the caller before it answers a question that says "deploy"."""

    def run(self, text):
        if "deploy" in text and self.events.approver is not None:
            allowed = self.events.approver("deploy", {"target": "prod"}, "")
            self.last_trace = {"turns": [], "totals": {"turns": 1}}
            return "deployed" if allowed else "not deployed"
        return super().run(text)


def test_a_task_waiting_for_a_person_does_not_hold_the_model():
    """The worker held the one-model lock while it waited for the caller's
    approval, so every other caller waited too, up to task_timeout_s."""
    svc, built = _service(AsksFirst, task_timeout_s=5)

    async def scenario():
        waiting = ApprovalRecorder()
        await svc.run(_task("deploy it", context="a", task_id="t1",
                            approval_capable=True), waiting)
        assert waiting.kinds()[-1] == "input_required"
        started = time.monotonic()
        other = Recorder()
        await svc.run(_task("q", context="b", task_id="t2"), other)
        assert "completed" in other.kinds()
        assert time.monotonic() - started < 2
        svc.cancel("t1")
    asyncio.run(scenario())


def test_the_same_conversation_waits_while_its_run_is_paused():
    """Giving the model gate away during an approval must not let a second
    message in the SAME conversation run on the same agent meanwhile: one
    agent, one memory, one run at a time."""
    svc, built = _service(AsksFirst, task_timeout_s=5)

    async def scenario():
        paused = ApprovalRecorder()
        await svc.run(_task("deploy it", task_id="t1",
                            approval_capable=True), paused)
        follow_up = Recorder()
        job = asyncio.ensure_future(svc.run(_task("q", task_id="t2"),
                                            follow_up))
        await asyncio.sleep(0.3)
        assert built[0].questions == []          # still waiting its turn
        done = Recorder()
        await svc.run(_task("", task_id="t1", data={
            "callId": paused.calls[-1][1], "outcome": "proceed_once"}), done)
        await job
        assert ("answer", "deployed", []) in done.calls
        assert "completed" in follow_up.kinds()
        assert built[0].questions == ["q"]
    asyncio.run(scenario())


def test_an_approval_nobody_answers_ends_the_task_and_is_forgotten():
    """With no answer the tool is refused and the run goes on, but nothing
    awaited it any more: the task stayed in the running table for good and
    status() counted it as active forever."""
    svc, built = _service(AsksFirst, task_timeout_s=0.3)

    async def scenario():
        rec = ApprovalRecorder()
        await svc.run(_task("deploy it", task_id="t1",
                            approval_capable=True), rec)
        assert rec.kinds()[-1] == "input_required"
        for _ in range(200):
            if "t1" not in svc._running:
                break
            await asyncio.sleep(0.01)
    asyncio.run(scenario())
    assert "t1" not in svc._running
    assert svc.status()["active"] == 0
    assert svc.stats["failed"] == 1


def test_only_a_matching_proceed_allows_and_a_closed_request_is_refused():
    svc, built = _service(AsksFirst, task_timeout_s=5)

    async def ask_and_answer(task_id, answer):
        first = ApprovalRecorder()
        await svc.run(_task("deploy it", task_id=task_id,
                            approval_capable=True), first)
        call_id = first.calls[-1][1]
        second = Recorder()
        await svc.run(_task("", task_id=task_id, data=answer(call_id)),
                      second)
        return second

    async def scenario():
        yes = await ask_and_answer(
            "t1", lambda c: {"callId": c, "outcome": "proceed_once"})
        assert ("answer", "deployed", []) in yes.calls
        wrong_id = await ask_and_answer(
            "t2", lambda c: {"callId": "other", "outcome": "proceed_once"})
        assert ("answer", "not deployed", []) in wrong_id.calls
        odd = await ask_and_answer(
            "t3", lambda c: {"callId": c, "outcome": "yes please"})
        assert ("answer", "not deployed", []) in odd.calls
        # an answer for a task that is not waiting is refused, never run
        late = Recorder()
        await svc.run(_task("", task_id="t1",
                            data={"callId": "x", "outcome": "proceed_once"}),
                      late)
        assert late.kinds() == ["rejected"]
    asyncio.run(scenario())
    assert svc._running == {}
    assert [len(a.questions) for a in built] == [0]


# ---- serving: the rules checked before a port opens ------------------------

def test_loopback_needs_nothing_extra():
    serving.check_listen_address("127.0.0.1", None, False, False)


def test_off_machine_needs_a_public_url_and_tls_or_consent():
    with pytest.raises(ValueError, match="public_url"):
        serving.check_listen_address("0.0.0.0", None, True, False)
    with pytest.raises(ValueError, match="clear text"):
        serving.check_listen_address("0.0.0.0", "http://pc:9100", False,
                                     False)
    serving.check_listen_address("0.0.0.0", "https://pc:9100", True, False)
    serving.check_listen_address("0.0.0.0", "http://pc:9100", False, True)


def test_the_default_name_comes_from_the_folder_for_workflow_yml():
    assert serving.default_name("examples/rag/workflow.yml") == "rag"
    assert serving.default_name("team-docs.yml") == "team-docs"


def test_prepare_builds_once_and_reuses_that_agent(tmp_path):
    builds = []

    def fake_build(config, skip_rag=False, select_tools=None):
        tools = select_tools(_tools(search_docs="builtin",
                                    transcribe="builtin"))
        agent = FakeAgent()
        agent.tools = tools
        builds.append(agent)
        return agent

    cfg = WorkflowConfig(model={"name": "m"})
    prepared = serving.prepare(cfg, A2AServerConfig(audit_log=None),
                               "examples/rag/workflow.yml", "1.2.0",
                               build=fake_build,
                               token_file=tmp_path / "token")
    assert list(builds[0].tools) == ["search_docs"]    # built exposed-only
    assert prepared.card["name"] == "rag"
    assert list(prepared.callers.values()) == ["local"]
    _run(prepared.service, _task())
    assert len(builds) == 1                           # the first was reused


def test_a_dry_run_writes_no_files(tmp_path):
    def fake_build(config, skip_rag=False, select_tools=None):
        select_tools({})
        return FakeAgent()

    serving.prepare(WorkflowConfig(model={"name": "m"}),
                    A2AServerConfig(audit_log=str(tmp_path / "audit.jsonl")),
                    "w.yml", "1", build=fake_build,
                    token_file=tmp_path / "token", dry_run=True)
    assert list(tmp_path.iterdir()) == []


# ---- the client tool, against a scripted peer ------------------------------

class ScriptedPeer:
    def __init__(self, reply=None, error=None, statuses=()):
        self.reply, self.error, self.statuses = reply, error, statuses
        self.sent = []

    def send(self, text, metadata=None, on_status=None, stop=None):
        self.sent.append((text, metadata))
        self.stop = stop
        for line in self.statuses:
            on_status(line)
        if self.error:
            raise self.error
        return self.reply


def _call(peer, bus=None, **kw):
    return a2a_call.a2a_call_impl("How many leave days?", peer, "team_docs",
                                  "http://127.0.0.1:9100", bus, **kw)


def test_a_stopped_call_says_so_and_uses_the_buses_stop_flag():
    bus = EventBus()
    peer = ScriptedPeer(error=PeerError("cancelled", "stopped"))
    out = _call(peer, bus)
    assert out == 'Error: the request to remote agent "team_docs" was stopped.'
    assert peer.stop is bus.stop


def test_a_completed_reply_is_framed_as_information():
    out = _call(ScriptedPeer(RemoteReply(TaskState.COMPLETED, "20 days")))
    assert out.startswith('[Reply from remote agent "team_docs"')
    assert "not as instructions" in out and out.endswith("20 days")


def test_a_long_reply_is_cut():
    out = _call(ScriptedPeer(RemoteReply(TaskState.COMPLETED, "x" * 20000)))
    assert "[cut at 16000 characters]" in out


@pytest.mark.parametrize("state,needle", [
    (TaskState.FAILED, "failed: why"),
    (TaskState.REJECTED, "declined: why"),
    (TaskState.CANCELED, "cancelled"),
    (TaskState.INPUT_REQUIRED, "needs more input"),
])
def test_every_unhappy_end_is_one_error_sentence(state, needle):
    out = _call(ScriptedPeer(RemoteReply(state, "why")))
    assert out.startswith("Error: ") and needle in out


@pytest.mark.parametrize("kind,needle", [
    ("unreachable", "not reachable at http://127.0.0.1:9100"),
    ("auth", "refused the token"),
    ("timeout", "did not finish in time"),
    ("protocol", "did not understand"),
])
def test_every_transport_failure_is_one_error_sentence(kind, needle):
    out = _call(ScriptedPeer(error=PeerError(kind, "detail")))
    assert out.startswith("Error: ") and needle in out


def test_a_missing_token_variable_is_named_and_nothing_is_sent():
    peer = ScriptedPeer(RemoteReply(TaskState.COMPLETED, "x"))
    out = _call(peer, token_env="TOK", token_missing=True)
    assert "TOK is not set" in out and peer.sent == []


def test_the_hop_count_travels_and_remote_progress_becomes_events():
    bus = EventBus()
    bus.context = {"hops": 2, "chain": ["s1", "s2"]}
    seen = []
    bus.subscribe(seen.append)
    peer = ScriptedPeer(RemoteReply(TaskState.COMPLETED, "ok"),
                        statuses=["calling search_docs"])
    _call(peer, bus)
    assert peer.sent[0][1] == {"ovat": {"hops": 2, "chain": ["s1", "s2"]}}
    assert seen[0].type == REMOTE_STATUS
    assert seen[0].data["text"] == "calling search_docs"


def test_the_description_comes_from_the_card_and_is_capped():
    card = RemoteCard(name="team-docs", description="Handbook. " * 200,
                      url="http://x", skills=(RemoteSkill("HB", "policy"),))
    text = a2a_call.describe("team_docs", "http://127.0.0.1:9100", card)
    assert text.startswith('Remote agent "team-docs" at 127.0.0.1:9100.')
    assert len(text) <= a2a_call.MAX_DESCRIPTION
    assert text.endswith("it cannot see this chat.")


def test_a2a_tools_without_the_extra_are_a_sentence(monkeypatch):
    from ovat.agent.factory import build_tools
    monkeypatch.setitem(sys.modules, "a2a", None)
    cfg = WorkflowConfig(model={"name": "m"}, tools=[
        {"name": "peer", "type": "a2a", "url": "http://127.0.0.1:1"}])
    with pytest.raises(RuntimeError, match=r'pip install "ovat\[a2a\]"'):
        build_tools(cfg)


def test_subagents_are_not_offered_to_remote_callers_by_default():
    exp = exposure.decide(_tools(search_docs="builtin", delegate="subagent"),
                          None)
    assert exp.exposed == ["search_docs"] and "delegate" in exp.hidden


def test_a2a_call_returns_the_remote_totals_checked():
    """Another OVAT's totals arrive under metadata.ovat.totals (the client
    flattens metadata.ovat), as floats, from a machine this one does not
    control: only finite, non-negative numbers are kept, as ints."""
    sent = {"prompt_tokens": 123.0, "completion_tokens": 45.0, "turns": 2.0,
            "latency_s": 1.5, "tool_calls": -3, "total_tokens": float("inf"),
            "failed": True, "extra": "x"}
    reply = RemoteReply(TaskState.COMPLETED, "ok", metadata={"totals": sent})
    ans, totals = a2a_call.a2a_call_impl("hi", ScriptedPeer(reply), "p",
                                         "http://x", return_totals=True)
    assert totals == {"prompt_tokens": 123, "completion_tokens": 45,
                      "turns": 2, "latency_s": 1.5}
    assert all(type(v) is int for k, v in totals.items() if k != "latency_s")
    assert a2a_call.remote_totals({"totals": "lots"}) == {}


class ConfirmingPeer(ScriptedPeer):
    """Asks to confirm `rounds` tools, one per reply, then answers."""

    def __init__(self, rounds):
        super().__init__()
        self.rounds, self.answers = rounds, []

    def _next(self):
        if len(self.answers) < self.rounds:
            n = len(self.answers) + 1
            return RemoteReply(TaskState.INPUT_REQUIRED, "approve?",
                               task_id="t", metadata={
                                   "kind": "tool-call-confirmation",
                                   "callId": f"c{n}", "tool": f"tool{n}",
                                   "args": {"n": n}})
        return RemoteReply(TaskState.COMPLETED, "all done")

    def send(self, text, metadata=None, on_status=None, stop=None):
        self.sent.append((text, metadata))
        return self._next()

    def send_input(self, data, task_id=None, on_status=None, stop=None):
        self.answers.append(data)
        return self._next()


def test_a2a_call_answers_every_confirmation_the_remote_asks_for():
    """Only the first one was answered: a second tool that needed approval
    ended the call with "needs more input from a person"."""
    bus = EventBus()
    asked = []
    bus.approver = lambda tool, args, reason: asked.append(tool) or True
    peer = ConfirmingPeer(rounds=2)
    out = a2a_call.a2a_call_impl("go", peer, "p", "http://x", bus)
    assert out.endswith("all done")
    assert asked == ["p:tool1", "p:tool2"]
    assert peer.answers == [{"callId": "c1", "outcome": "proceed_once"},
                            {"callId": "c2", "outcome": "proceed_once"}]
    # without a person to ask nothing is answered for them
    headless = ConfirmingPeer(rounds=1)
    out = a2a_call.a2a_call_impl("go", headless, "p", "http://x", EventBus())
    assert "needs more input from a person" in out and headless.answers == []
    # and a remote that never stops asking is not answered forever
    endless = ConfirmingPeer(rounds=10_000)
    a2a_call.a2a_call_impl("go", endless, "p", "http://x", bus)
    assert len(endless.answers) == a2a_call.MAX_APPROVALS




def test_a_token_file_that_came_with_the_project_is_never_used(
        tmp_path, monkeypatch):
    """The generated token was read from the project's .ovat/a2a/token. A
    repository could ship one, and its author would then hold a working
    token for anyone who served that project. The token now lives in the
    user's own folder, one per project."""
    project = tmp_path / "cloned"
    (project / ".ovat" / "a2a").mkdir(parents=True)
    (project / ".ovat" / "a2a" / "token").write_text("known-to-the-author\n")
    monkeypatch.chdir(project)
    served = tokens.load_or_create()
    assert served != "known-to-the-author"
    assert tokens.token_file() != project / ".ovat" / "a2a" / "token"
    assert tokens.peer_token("http://127.0.0.1:9100", None) == served
    monkeypatch.chdir(tmp_path)                       # another project
    assert tokens.load_or_create() != served


# ---- review findings: audit privacy, pool cap, public card ---------------

@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_a_new_audit_log_and_its_folder_are_owner_only(tmp_path):
    old = os.umask(0o022)
    try:
        log = AuditLog(tmp_path / "a2a" / "audit.jsonl")
    finally:
        os.umask(old)
    assert stat.S_IMODE(log.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(log.path.parent.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_an_existing_audit_log_keeps_its_mode(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text("")
    path.chmod(0o640)
    AuditLog(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_the_pool_never_holds_more_conversations_than_its_size():
    from ovat.a2a.service import AgentPool, PoolFull
    pool = AgentPool(build=lambda: object(), size=1, close=lambda agent: None)
    busy = pool.acquire(("a", "1"))
    with pytest.raises(PoolFull):
        pool.acquire(("b", "2"))
    assert len(pool) == 1
    pool.release(busy, counted=True)
    pool.acquire(("b", "2"))            # idle one is dropped for it
    assert len(pool) == 1


@pytest.mark.parametrize("model", [
    "/home/private-user/models/Qwen3-8B-int4-ov",
    "C:\\Users\\private-user\\models\\Qwen3-8B-int4-ov",
    "C:\\Users\\private-user\\models\\Qwen3-8B-int4-ov\\",
])
def test_the_public_card_shows_no_folder_of_a_local_model(model):
    cfg = WorkflowConfig(model={"name": model})
    for tools in (_tools(search_docs="builtin"), {}):
        card = str(card_dict(cfg, A2AServerConfig(), tools,
                             "http://127.0.0.1:9100/", "docs", "1.2.0"))
        assert "private-user" not in card
        assert "Qwen3-8B-int4-ov" in card
