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

    def send(self, text, metadata=None, on_status=None):
        self.sent.append((text, metadata))
        for line in self.statuses:
            on_status(line)
        if self.error:
            raise self.error
        return self.reply


def _call(peer, bus=None, **kw):
    return a2a_call.a2a_call_impl("How many leave days?", peer, "team_docs",
                                  "http://127.0.0.1:9100", bus, **kw)


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


def test_a2a_call_returns_trace_totals_from_server_metadata():
    totals = {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168}
    reply = RemoteReply(TaskState.COMPLETED, "ok", metadata={"ovat": {"trace": totals}})
    peer = ScriptedPeer(reply)
    ans, returned_totals = a2a_call.a2a_call_impl("hi", peer, "p", "http://x", return_totals=True)
    assert returned_totals == totals

