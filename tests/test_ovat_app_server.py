# tests/test_ovat_app_server.py
"""OVAT's app server, driven by the copied Vibe client.

The server (ovat/tui/server/) is the other end of the wire the copied TUI
speaks. These tests start it with `start_ovat_app_server` and a fake
AgentSession (the same idea as tests/test_agent_tui.py's FakeSession: a turn
is played on a thread, exactly as the real session reports one), then use
Vibe's REAL client, `AppServerSession` and its `resources`, to talk to it.
So a passing test means the copied client accepted what the server sent: the
client validates every response with its protocol models and refuses an event
that is out of order.

Every wait is bounded (the Windows runner is slow) and nothing uses
pytest-asyncio: each test is one `asyncio.run`.
"""
import asyncio
import os
import subprocess
import sys
import threading
import time
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from ovat.agent.compaction import CompactionFailed, compacted_memory
from ovat.agent.events import (SUBAGENT_STATUS, TOOL_ENDED, TOOL_STARTED,
                               Event)
from ovat.agent.permissions import (ALWAYS, ONCE, REFUSE, SESSION,
                                    ApprovalBroker)
from ovat.agent.questions import QuestionBroker
from ovat.agent.session import Session
from ovat.agentic import session as s
from ovat.tui.server import saved, serve_on_memory, start_ovat_app_server
from ovat.tui.server.methods.session import history_page
from ovat.tui.vibe.app_server.client import AppServerClient
from ovat.tui.vibe.app_server.events import (CallbackRequested,
                                             HistoryEntryAdded,
                                             HistoryEntryUpdated,
                                             ServerWarning, StatsUpdated,
                                             TurnCompleted, TurnQueueUpdated,
                                             TurnStarted)
from ovat.tui.vibe.app_server.models import (ApprovalCallbackDetail,
                                             ImageAttachment,
                                             InlineImageSource,
                                             ApprovalCallbackOutput,
                                             ApprovalDecision,
                                             ApprovalDecisionType,
                                             BlockedEffectState,
                                             CancelledEffectState,
                                             CompletedEffectState,
                                             FailedEffectState,
                                             PublicCallbackEntry,
                                             PublicEffectEntry,
                                             PublicMessageEntry,
                                             PublicReasoningEntry,
                                             PublicTurnStatus,
                                             RunningEffectState,
                                             SkippedEffectState)
from ovat.tui.vibe.app_server.protocol import (AppServerResponseError,
                                               ClientCapabilities, ClientInfo,
                                               Notification, ProtocolErrorCode,
                                               ServerRequest)

T = 10.0                       # the longest any single wait may take

MODELS = [{"id": "Qwen3.5-4B-int4-ov", "label": "Qwen3.5-4B-int4-ov"},
          {"id": "Qwen3-8B-int4-ov", "label": "Qwen3-8B-int4-ov"}]

INFO = {"workflow": "w.yml", "name": "w", "model": "Qwen3.5-4B-int4-ov",
        "model_id": "Qwen3.5-4B-int4-ov", "models": MODELS,
        "default_model_id": "Qwen3.5-4B-int4-ov", "thinking": None,
        "device": "GPU", "provider": "ovms", "engine": "native",
        "tools": [{"name": "search_docs", "kind": "builtin",
                   "permission": "allow"},
                  {"name": "delegate", "kind": "subagent",
                   "permission": "allow"}],
        "subagents": [{"name": "researcher", "description": "reads docs",
                       "where": "workflow"}]}


def run(coro):
    asyncio.run(asyncio.wait_for(coro, 60))


class FakeAgent:
    """What AgentSession.agent offers that the server reads."""

    def __init__(self, native=True):
        self.tools = {"search_docs": {}, "delegate": {}}
        if native:
            self.max_iterations = 10
            self.llm = SimpleNamespace(max_tokens=4096)


class FakeSession:
    """Plays turns on threads, as AgentSession does, and records what the
    server did to it."""

    agent_class = FakeAgent

    def __init__(self, workflow, sink, cwd=None, *, native=True,
                 answer="New hires get 20 days.", reasoning="",
                 answer_deltas=(), reasoning_deltas=(), delta_pause=0.06,
                 context_window=0,
                 fail_build=None, hold_build=False, hold_turn=False,
                 approval=None, tool=True, shell_lines=("one", "two"),
                 shell_code=0, hold_shell=False, rebuild=False,
                 failed=False, totals=None, start_raises=False,
                 question=None, extra_tools=(),
                 hold_start=False, compact_error=None, fail_pick=False):
        self.workflow, self.sink = workflow, sink
        self.compact_error = compact_error      # a model that cannot summarize
        self.compacts = []                      # the instructions it was given
        self.cwd = cwd or os.getcwd()
        self.broker = ApprovalBroker(
            on_request=lambda r: sink(s.ApprovalNeeded(r)))
        self.resolved = []                      # (request id, decision)
        real_resolve = self.broker.resolve

        def spy(request_id, decision):
            self.resolved.append((request_id, decision))
            return real_resolve(request_id, decision)
        self.broker.resolve = spy
        self.questions = QuestionBroker(
            on_request=lambda r: sink(s.QuestionNeeded(r)))
        self.question, self.extra_tools = question, tuple(extra_tools)
        self.answered = "not asked"             # what ask_user got back
        self.native, self.answer, self.reasoning = native, answer, reasoning
        self.answer_deltas = tuple(answer_deltas)
        self.reasoning_deltas = tuple(reasoning_deltas)
        self.delta_pause = delta_pause
        self.context_window = context_window
        self.fail_build, self.tool, self.approval = fail_build, tool, approval
        self.shell_lines, self.shell_code = shell_lines, shell_code
        self.rebuild, self.failed = rebuild, failed
        self.start_raises = start_raises
        # The real session clears the stop flag as its worker starts, so a
        # cancel that lands before that is lost. hold_start plays that gap.
        self.start_gate = threading.Event()
        if not hold_start:
            self.start_gate.set()
        self.hold_start = hold_start
        self.totals_per_turn = totals if totals is not None else {
            "tool_calls": 1, "prompt_tokens": 100, "completion_tokens": 20,
            "latency_s": 0.1}
        self.build_gate = threading.Event()
        self.turn_gate = threading.Event()
        self.shell_gate = threading.Event()
        if not hold_build:
            self.build_gate.set()
        if not hold_turn:
            self.turn_gate.set()
        if not hold_shell:
            self.shell_gate.set()
        self.stop = threading.Event()
        self.busy = False
        self.agent = None
        self.info = {}
        self.serving = False
        self.server_calls = []                  # "start" / "stop" for /serve
        self.last_trace, self.last_answer = {}, ""
        self.totals = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.asked, self.subagent_calls = [], []
        self.steered = []
        self.engines = []
        # /model and /thinking: every switch asked for, the model and the
        # thinking switch the session runs now, and whether a switch fails.
        self.picks = []
        self.forgotten = []             # session picks /config dropped
        self.fail_pick = fail_pick
        self.model_id, self.thinking = INFO["model_id"], None
        self.default_model_id = INFO["default_model_id"]
        self.engine = INFO["engine"]
        self.turn_count = 0
        self.approved = None
        self.closed = False
        self.cleared = 0
        self.cancels = 0
        self.shell_busy = False
        self.shell_killed = threading.Event()
        self._lock = threading.Lock()
        # What the model remembers, kept as the native loop keeps it: a real
        # Session, a user message per question, an answer per finished turn.
        self.remembered = Session(system_prompt="be brief")

    def _thread(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True,
                                  name="fake-session")
        thread.start()
        return thread

    # ---- lifecycle -------------------------------------------------------------

    def start(self, engine=None):
        if self.start_raises:
            raise RuntimeError("no worker thread for you")
        self.engines.append(engine)

        def go():
            self.build_gate.wait(T)
            if self.fail_build or self.workflow.endswith("bad.yml"):
                self.sink(s.BuildFailed(
                    self.fail_build
                    or f"Could not build {self.workflow}: boom"))
                return
            # as the real session: the old agent is replaced only now
            self.agent = self.agent_class(self.native)
            self.info = self._describe(engine)
            self.sink(s.Ready(self.info))
        return self._thread(go)

    def _describe(self, engine=None):
        return dict(INFO, workflow=self.workflow,
                    engine=engine or self.engine,
                    context_window=self.context_window,
                    model_id=self.model_id,
                    model=self.model_id, thinking=self.thinking,
                    default_model_id=self.default_model_id)

    def switch_model(self, model=None, enable_thinking=None,
                     follow_workflow=False):
        self.picks.append((model, enable_thinking))

        def go():
            self.build_gate.wait(T)
            if self.fail_pick:
                self.sink(s.BuildFailed("Could not build w.yml: no such "
                                        "model"))
                return
            if os.path.isfile(self.workflow):
                from ovat.config.workflow import load_workflow
                config = load_workflow(self.workflow)
                self.model_id = config.model.name
                self.default_model_id = config.model.name
                self.thinking = config.model.enable_thinking
                if follow_workflow:
                    self.engine = config.agent.type
            if model is not None:
                self.model_id = model or self.default_model_id
            if enable_thinking is not None:
                self.thinking = enable_thinking
            # as a real rebuild: the new agent remembers nothing until the
            # server hands the old memory back
            self.remembered = Session(system_prompt="be brief")
            self.agent = self.agent_class(self.native)
            self.info = self._describe()
            self.sink(s.Ready(self.info))
        return self._thread(go)

    def forget_picks(self, *names):
        self.forgotten.extend(names)

    def switch_workflow(self, path):
        self.workflow = path
        return self.start()

    def new_conversation(self):
        self.cleared += 1
        self.remembered = Session(system_prompt="be brief")
        if not self.rebuild:
            # as the native loop does: forget in place and say so
            self.sink(s.Notice("info", "New conversation."))
            return None
        return self.start()

    def close(self):
        self.closed = True
        for gate in (self.build_gate, self.turn_gate, self.shell_gate):
            gate.set()
        self.stop.set()
        self.broker.cancel_all()

    # ---- turns -----------------------------------------------------------------

    def ask(self, text):
        with self._lock:
            if self.busy or self.agent is None:
                self.sink(s.Notice("warn", "Still working; Esc stops it."))
                return None
            self.busy = True
        self.asked.append(text)
        if self.native:
            self.remembered.add_user(text)
        return self._thread(self._turn, text)

    def ask_subagent(self, name, task):
        with self._lock:
            if self.busy or self.agent is None:
                return None
            self.busy = True
        self.subagent_calls.append((name, task))
        return self._thread(self._turn, f"/ask {name} {task}")

    # ---- the model's memory (native only, as AgentSession) --------------------

    def memory(self):
        if not self.native or self.agent is None:
            return None
        return self.remembered.snapshot()

    def restore_memory(self, messages):
        if not self.native:
            return False
        self.remembered = Session(system_prompt="be brief")
        self.remembered.messages.extend(
            m for m in messages if m.get("role") != "system")
        return True

    def truncate_memory(self, length):
        if not self.native:
            return False
        self.remembered.truncate(length)
        return True

    def steer(self, text):
        """As AgentSession.steer: native and busy only. The real loop takes
        the text before its next model call and reports where it landed; the
        fake takes it at once."""
        if not self.native or not self.busy:
            return False
        self.steered.append(text)
        index = len(self.remembered.snapshot())
        self.remembered.add_user(text)
        self.sink(s.Steered(text, index))
        return True

    def compact(self, instructions=""):
        """As the real one: native only, the memory becomes the summary
        envelope plus an acknowledgement, a failure leaves it alone."""
        if not self.native:
            raise CompactionFailed("Compacting needs the native engine")
        if self.compact_error:
            raise CompactionFailed(self.compact_error)
        self.compacts.append(instructions)
        memory = self.remembered.snapshot()
        summary = "they asked: " + ", ".join(
            m["content"] for m in memory if m["role"] == "user")
        self.restore_memory(compacted_memory(memory, summary))
        return summary

    def _turn(self, text):
        self.start_gate.wait(T)
        self.stop.clear()
        self.turn_count += 1
        call_id = f"c{self.turn_count}"
        self.sink(s.TurnStarted(text))
        if self.hold_start:
            # a long model call that only a second cancel (after the start)
            # can stop
            deadline = time.monotonic() + 3
            while self.cancels < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
        if self.tool:
            self.sink(s.AgentActivity(Event(
                TOOL_STARTED, {"call_id": call_id, "name": "search_docs",
                               "arguments": {"query": text}})))
        if self.approval is not None:
            self.approved = self.broker(
                "search_docs", self.approval, "sends the question away")
        for n, extra in enumerate(self.extra_tools, start=1):
            # A tool that reports a `display`, as the real observed() does.
            extra_id = f"x{n}"
            self.sink(s.AgentActivity(Event(TOOL_STARTED, {
                "call_id": extra_id, "name": extra["name"],
                "arguments": extra["arguments"]})))
            self.sink(s.AgentActivity(Event(TOOL_ENDED, {
                "call_id": extra_id, "name": extra["name"], "status": "ok",
                "duration_s": 0.1, "result_chars": 5,
                "display": extra["display"]})))
        if self.question is not None:
            self.sink(s.AgentActivity(Event(TOOL_STARTED, {
                "call_id": "q1", "name": "ask_user",
                "arguments": self.question})))
            self.answered = self.questions(
                self.question["question"], self.question.get("choices", []))
            self.sink(s.AgentActivity(Event(TOOL_ENDED, {
                "call_id": "q1", "name": "ask_user", "status": "ok",
                "duration_s": 0.1, "result_chars": 5, "display": {
                    "kind": "ask_user", "answer": self.answered,
                    "cancelled": self.answered is None}})))
        self.turn_gate.wait(T)
        if self.tool:
            denied = self.approval is not None and not self.approved
            self.sink(s.AgentActivity(Event(
                TOOL_ENDED, {"call_id": call_id, "name": "search_docs",
                             "status": "denied" if denied else "ok",
                             "duration_s": 0.4, "result_chars": 9})))
        for piece in self.reasoning_deltas:
            if self.stop.is_set():
                break
            self.sink(s.ReasoningDelta(piece))
            if self.delta_pause:
                time.sleep(self.delta_pause)
        for piece in self.answer_deltas:
            if self.stop.is_set():
                break
            self.sink(s.AnswerDelta(piece))
            if self.delta_pause:
                time.sleep(self.delta_pause)
        cancelled = self.stop.is_set()
        if self.native and not text.startswith("/ask "):
            # a subagent turn (`/ask`) leaves the main model's memory alone
            self.remembered.add_assistant(self.answer)
        self.totals["turns"] += 1
        self.totals["prompt_tokens"] += self.totals_per_turn.get(
            "prompt_tokens") or 0
        self.totals["completion_tokens"] += self.totals_per_turn.get(
            "completion_tokens") or 0
        self.busy = False
        self.sink(s.TurnDone(
            "Error: cancelled by the caller." if cancelled else self.answer,
            0.5, dict(self.totals_per_turn), [], self.failed, cancelled,
            self.reasoning))

    def start_server(self):
        self.server_calls.append("start")
        self.serving = True
        self.sink(s.Notice("info", "a2a: serving on 127.0.0.1:9999"))

    def stop_server(self):
        self.server_calls.append("stop")
        self.serving = False

    def cancel(self):
        self.cancels += 1
        stopped = False
        if self.busy:
            self.stop.set()
            self.turn_gate.set()
            stopped = True
        self.broker.cancel_all()
        self.questions.cancel_all()
        if self.shell_busy:
            self.shell_killed.set()
            self.shell_gate.set()
            stopped = True
        return stopped

    # ---- the `!` shell ---------------------------------------------------------

    def run_shell(self, command):
        with self._lock:
            if self.shell_busy:
                self.sink(s.Notice("warn", "A command is still running."))
                return None
            self.shell_busy = True

        def go():
            for line in self.shell_lines:
                self.sink(s.ShellLine(line))
            self.shell_gate.wait(T)
            killed = self.shell_killed.is_set()
            self.shell_busy = False
            self.sink(s.ShellDone(command, -15 if killed else self.shell_code,
                                  0.1))
        return self._thread(go)


@asynccontextmanager
async def served(workflow="w.yml", **script):
    """The server, the copied client's session, the fake. Always shut down."""
    made = []

    def factory(workflow, sink, cwd=None):
        fake = FakeSession(workflow, sink, cwd, **script)
        made.append(fake)
        return fake

    session, handle = await asyncio.wait_for(
        start_ovat_app_server(workflow, session_factory=factory), T)
    try:
        yield session, handle, made[0]
    finally:
        await asyncio.wait_for(session.close(), T)
        await asyncio.wait_for(handle.close(), T)


async def events_until(session, stop, timeout=T):
    """The unsolicited events, up to and including the first `stop` one."""
    seen = []

    async def go():
        async with aclosing(session.events()) as events:
            async for event in events:
                seen.append(event)
                if stop(event):
                    return
    await asyncio.wait_for(go(), timeout)
    return seen


def is_completed(event):
    return isinstance(event, TurnCompleted)


def kinds(events):
    return [type(e).__name__ for e in events]


def completed_turn(events):
    return next(e.turn for e in events if isinstance(e, TurnCompleted))


async def poll(check, timeout=T):
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "condition not met in time"
        await asyncio.sleep(0.02)


# ---- the handshake and the session ---------------------------------------------

def test_the_handshake_opens_a_session_the_client_accepts():
    async def go():
        async with served() as (session, handle, fake):
            assert session.session_id
            assert session.cwd == fake.cwd
            assert handle.agent_session is fake
            await asyncio.wait_for(
                session.resources.runtime.wait_until_ready(), T)
            runtime = session.resources.runtime
            assert runtime.ready
            config = session.resources.config.current
            assert config.active_model.name == "Qwen3.5-4B-int4-ov"
            # No update check, no telemetry: nothing goes over the network.
            assert not config.enable_update_checks
            assert not config.enable_telemetry
            assert [t.name for t in runtime.tools] == ["search_docs",
                                                       "delegate"]
            assert not any(t.is_custom for t in runtime.tools)
    run(go())


def test_runtime_and_stats_read_back_as_valid_models():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            runtime = session.resources.runtime
            await runtime.refresh()                       # runtime/read
            assert runtime.stats.steps == 0
            assert runtime.context_window == 0            # not known
            assert runtime.mcp.sources == []
            assert runtime.skills == []
            assert runtime.session_log.session_id == session.session_id
            assert not runtime.session_log.persisted
            # stats/read and tools/list have no resource of their own; ask.
            client = session._connection.current
            stats = await client.request("stats/read",
                                         {"sessionId": session.session_id})
            assert stats["stats"]["steps"] == 0
            assert stats["contextWindow"] == 0
            tools = await client.request("tools/list",
                                         {"sessionId": session.session_id})
            assert [t["name"] for t in tools["tools"]] == [
                "search_docs", "delegate"]
    run(go())


def test_agents_are_the_two_profiles_plus_the_workflows_subagents():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            agents = session.resources.agents
            await agents.read()
            assert [a.name for a in agents.all] == [
                "default", "auto-approve", "researcher"]
            assert agents.active.name == "default"
            assert agents.next().name == "auto-approve"
            assert not fake.broker.auto_approve
            await agents.switch("auto-approve")
            # /yolo is this switch: the broker now says yes to every ask.
            assert fake.broker.auto_approve
            assert agents.active.name == "auto-approve"
            await agents.switch("default")
            assert not fake.broker.auto_approve
            with pytest.raises(AppServerResponseError) as err:
                await agents.switch("researcher")
            assert err.value.error.code is ProtocolErrorCode.NOT_FOUND
    run(go())


def test_a_build_failure_is_announced_once_and_fails_a_turn():
    """Readiness itself does not fail: the copied app treats a failing wait
    as fatal ("press any key to exit"), and a workflow that does not build
    can be fixed with /workflow. The reason is a ServerError event, and a
    prompt sent meanwhile fails with it."""
    async def go():
        async with served(fail_build="Could not build w.yml: boom") as (
                session, handle, fake):
            await asyncio.wait_for(
                session.resources.runtime.wait_until_ready(), T)
            assert not session.resources.runtime.ready
            await session.enqueue("hello")
            seen = await events_until(session, is_completed)
            errors = [e.params.error.message for e in seen
                      if type(e).__name__ == "ServerError"]
            assert errors == ["Could not build w.yml: boom"]
            turn = completed_turn(seen)
            assert turn.status is PublicTurnStatus.FAILED
            assert "boom" in turn.error.message
            assert fake.asked == []
    run(go())


def test_a_prompt_sent_while_the_agent_builds_waits_and_then_runs():
    async def go():
        async with served(hold_build=True) as (session, handle, fake):
            await session.enqueue("early question")
            started = await events_until(
                session, lambda e: isinstance(e, TurnStarted))
            assert kinds(started)[-1] == "TurnStarted"
            await asyncio.sleep(0.2)
            assert fake.asked == []                  # still waiting
            assert session.turn_active
            fake.build_gate.set()
            seen = await events_until(session, is_completed)
            assert fake.asked == ["early question"]
            assert completed_turn(seen).status is PublicTurnStatus.COMPLETED
    run(go())


# ---- /engine and /workflow through the handle -----------------------------------

def test_switching_the_engine_builds_again_and_the_runtime_follows():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            heard = []
            handle.subscribe(heard.append)
            await asyncio.wait_for(handle.switch_engine("react"), T)
            await poll(lambda: any(isinstance(e, s.Ready) for e in heard))
            assert fake.engines == [None, "react"]
            effective = await session.resources.config.read_effective()
            assert effective.config["engine"] == "react"
            assert session.resources.runtime.ready
    run(go())


def test_a_prompt_sent_while_the_engine_rebuilds_waits_for_the_new_agent():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            fake.build_gate.clear()
            await asyncio.wait_for(handle.switch_engine("react"), T)
            await session.enqueue("while it builds")
            await events_until(session, lambda e: isinstance(e, TurnStarted))
            await asyncio.sleep(0.2)
            assert fake.asked == []                  # still waiting
            fake.build_gate.set()
            seen = await events_until(session, is_completed)
            assert fake.asked == ["while it builds"]
            assert completed_turn(seen).status is PublicTurnStatus.COMPLETED
    run(go())


def test_switching_the_workflow_loads_the_new_file():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await asyncio.wait_for(handle.switch_workflow("other.yml"), T)
            await poll(lambda: fake.workflow == "other.yml"
                       and session.resources.runtime.ready)
            effective = await session.resources.config.read_effective()
            assert effective.config["workflow"] == "other.yml"
    run(go())


def test_a_workflow_that_does_not_build_leaves_the_old_agent_working():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await asyncio.wait_for(handle.switch_workflow("bad.yml"), T)
            # the error is announced (the app draws it), nothing is fatal
            seen = await events_until(
                session, lambda e: type(e).__name__ == "ServerError")
            assert "bad.yml" in seen[-1].params.error.message
            await poll(lambda: fake.workflow == "w.yml")   # the old one is kept
            await session.enqueue("still there?")
            done = await events_until(session, is_completed)
            assert completed_turn(done).status is PublicTurnStatus.COMPLETED
            assert fake.asked == ["still there?"]
    run(go())


def test_switching_is_refused_while_a_turn_runs():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("long one")
            await events_until(session, lambda e: isinstance(e, TurnStarted))
            with pytest.raises(Exception) as err:
                await handle.switch_engine("react")
            assert "turn" in str(err.value)
            assert fake.engines == [None]
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


def test_a_listener_hears_the_reports_until_it_unsubscribes():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            heard = []
            stop = handle.subscribe(heard.append)
            await session.enqueue("one")
            await events_until(session, is_completed)
            assert [type(e).__name__ for e in heard] == [
                "TurnStarted", "AgentActivity", "AgentActivity", "TurnDone"]
            stop()
            await session.enqueue("two")
            await events_until(session, is_completed)
            assert len(heard) == 4
    run(go())


def test_continue_and_resume_say_when_there_is_nothing_to_open():
    async def go():
        for kwargs, says in (({"continue_session": True},
                              "no saved session to continue"),
                             ({"resume_session_id": "nope"},
                              "No saved session nope")):
            with pytest.raises(AppServerResponseError) as err:
                await asyncio.wait_for(start_ovat_app_server(
                    "w.yml", session_factory=FakeSession, **kwargs), T)
            assert err.value.error.code is ProtocolErrorCode.NOT_FOUND
            assert says in str(err.value)
        await poll(lambda: not _leftover_threads())
    run(go())


# ---- a turn ---------------------------------------------------------------------

def test_a_turn_started_through_the_client_streams_entries_in_order():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            events = []

            async def drive():
                async with aclosing(session.act("What is the leave?")) as gen:
                    async for event in gen:
                        events.append(event)
            await asyncio.wait_for(drive(), T)

            names = kinds(events)
            assert names[0] == "TurnStarted"
            tool_added = next(
                e for e in events if isinstance(e, HistoryEntryAdded)
                and isinstance(e.entry, PublicEffectEntry))
            tool_done = next(
                e for e in events if isinstance(e, HistoryEntryUpdated)
                and isinstance(e.entry, PublicEffectEntry))
            answer = next(
                e for e in events if isinstance(e, HistoryEntryAdded)
                and isinstance(e.entry, PublicMessageEntry)
                and e.entry.role == "assistant")
            user = next(
                e for e in events if isinstance(e, HistoryEntryAdded)
                and isinstance(e.entry, PublicMessageEntry)
                and e.entry.role == "user")
            # running first, then completed, then the answer
            assert isinstance(tool_added.entry.state, RunningEffectState)
            assert tool_added.entry.id == tool_done.entry.id == "c1"
            assert isinstance(tool_done.entry.state, CompletedEffectState)
            assert (events.index(user) < events.index(tool_added)
                    < events.index(tool_done) < events.index(answer))
            assert user.entry.text == "What is the leave?"
            assert answer.entry.text == "New hires get 20 days."
            assert all(e.entry.turn_id == events[0].turn.id
                       for e in (tool_added, user, answer))
            assert any(isinstance(e, StatsUpdated) for e in events)
            # TurnCompleted ends act(); the client's own state has it
            turn = session.state.turns[-1]
            assert turn.status is PublicTurnStatus.COMPLETED
            assert not session.turn_active
            assert fake.asked == ["What is the leave?"]
            # the client's mirror of the history is the server's
            ids = [e.id for e in session.history]
            assert ids == [user.entry.id, "c1", answer.entry.id]
    run(go())


def test_the_queue_path_starts_a_turn_the_way_the_app_does():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            queued = await session.enqueue("hello")
            seen = await events_until(session, is_completed)
            names = kinds(seen)
            # the item is queued, promoted, and then the turn starts
            queue_events = [e for e in seen if isinstance(e, TurnQueueUpdated)]
            assert [len(e.queue.items) for e in queue_events] == [1, 0]
            started = next(e for e in seen if isinstance(e, TurnStarted))
            assert started.turn.queue_item_id == queued.id
            assert names.index("TurnStarted") > names.index("TurnQueueUpdated")
            assert completed_turn(seen).status is PublicTurnStatus.COMPLETED
            assert session.turn_queue.items == []
            # the user's entry carries the id the app gave it
            user = next(e.entry for e in seen
                        if isinstance(e, HistoryEntryAdded)
                        and isinstance(e.entry, PublicMessageEntry)
                        and e.entry.role == "user")
            assert user.id == queued.entries[0].entry_id or user.id
    run(go())


def test_a_prompt_sent_while_busy_waits_in_the_queue_and_runs_next():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("first")
            await poll(lambda: fake.asked == ["first"])
            second = await session.enqueue("second")
            await poll(lambda: any(i.id == second.id
                                   for i in session.turn_queue.items))
            queue = await session.read_turn_queue()
            assert [i.id for i in queue.items] == [second.id]
            assert not queue.paused
            fake.turn_gate.set()
            await events_until(session, is_completed)         # the first
            seen = await events_until(
                session, lambda e: is_completed(e)
                and e.turn.queue_item_id == second.id)
            assert fake.asked == ["first", "second"]
            assert (await session.read_turn_queue()).items == []
    run(go())


def test_a_queued_prompt_can_be_replaced_and_removed():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("first")
            await poll(lambda: fake.asked == ["first"])
            item = await session.enqueue("draft")
            await poll(lambda: session.turn_queue.items)
            changed = await session.replace_queued_turn(item.id, "final")
            assert changed.id == item.id
            queue = await session.read_turn_queue()
            assert queue.items[0].entries[0].content[0].text == "final"
            assert await session.remove_queued_turn(item.id)
            assert (await session.read_turn_queue()).items == []
            fake.turn_gate.set()
            await events_until(session, is_completed)
            assert fake.asked == ["first"]
    run(go())


def test_turn_start_is_refused_while_a_turn_runs():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("first")
            await poll(lambda: fake.asked == ["first"])
            client = session._connection.current
            with pytest.raises(AppServerResponseError) as err:
                await client.request("turn/start", {
                    "sessionId": session.session_id,
                    "message": [{"type": "text", "text": "second"}]})
            assert err.value.error.code is ProtocolErrorCode.CONFLICT
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


def test_a_thinking_models_reasoning_is_its_own_entry_before_the_answer():
    async def go():
        async with served(reasoning="<think>first I check</think>") as (
                session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("why")
            seen = await events_until(session, is_completed)
            added = [e.entry for e in seen if isinstance(e, HistoryEntryAdded)]
            reasoning = next(e for e in added
                             if isinstance(e, PublicReasoningEntry))
            answer = next(e for e in added if isinstance(e, PublicMessageEntry)
                          and e.role == "assistant")
            assert reasoning.text == "first I check"          # no tags
            assert added.index(reasoning) < added.index(answer)
    run(go())


def test_a_failed_turn_completes_failed_with_the_sentence_to_show():
    async def go():
        async with served(answer="Error: OVMS is not answering.",
                          failed=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("hello")
            seen = await events_until(session, is_completed)
            turn = completed_turn(seen)
            assert turn.status is PublicTurnStatus.FAILED
            # the app writes "Error: " in front itself (seen doubled on the
            # AI PC: "Error: Error: ...")
            assert turn.error.message == "OVMS is not answering."
            assert not any(isinstance(e, HistoryEntryAdded)
                           and isinstance(e.entry, PublicMessageEntry)
                           and e.entry.role == "assistant" for e in seen)
    run(go())


def test_a_reply_cut_at_the_token_cap_is_marked_as_cut():
    async def go():
        totals = {"prompt_tokens": 5, "completion_tokens": 4096,
                  "truncated": True}
        async with served(totals=totals) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("write a lot")
            seen = await events_until(session, is_completed)
            assert completed_turn(seen).stop_reason.value == "length"
    run(go())


def test_stats_follow_the_turns():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            for text in ("one", "two"):
                await session.enqueue(text)
                await events_until(session, is_completed)
            stats = session.resources.runtime.stats
            assert stats.steps == 2
            assert stats.session_prompt_tokens == 200
            assert stats.session_completion_tokens == 40
            assert stats.context_tokens == 100         # the last prompt
            assert stats.tool_calls_succeeded == 2
            assert stats.last_turn_duration == 0.5
    run(go())


def test_an_at_mention_of_a_subagent_goes_straight_to_it():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("@researcher find the leave policy")
            await events_until(session, is_completed)
            assert fake.subagent_calls == [("researcher",
                                            "find the leave policy")]
            assert fake.asked == []
            # a file mention, and an unknown name, are ordinary prompts
            await session.enqueue("read @docs/leave.md please")
            await events_until(session, is_completed)
            await session.enqueue("@nobody hello")
            await events_until(session, is_completed)
            assert fake.asked == ["read @docs/leave.md please",
                                  "@nobody hello"]
    run(go())


def test_subagent_progress_lines_are_appended_under_the_running_call():
    async def go():
        class Progress(FakeSession):
            def _turn(self, text):
                self.stop.clear()
                self.sink(s.TurnStarted(text))
                self.sink(s.AgentActivity(Event(
                    TOOL_STARTED, {"call_id": "d1", "name": "delegate",
                                   "arguments": {"agent": "researcher",
                                                 "task": "look"}})))
                self.sink(s.AgentActivity(Event(
                    SUBAGENT_STATUS, {"agent": "researcher",
                                      "text": "calling search_docs"})))
                self.sink(s.AgentActivity(Event(
                    TOOL_ENDED, {"call_id": "d1", "name": "delegate",
                                 "status": "ok", "duration_s": 1.0,
                                 "result_chars": 3})))
                self.busy = False
                self.sink(s.TurnDone("done", 1.0, {}, [], False, False))

        made = []

        def factory(workflow, sink, cwd=None):
            made.append(Progress(workflow, sink, cwd))
            return made[0]
        session, handle = await start_ovat_app_server(
            "w.yml", session_factory=factory)
        try:
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("delegate it")
            seen = await events_until(session, is_completed)
            updates = [e for e in seen if isinstance(e, HistoryEntryUpdated)]
            appended = [op.value for u in updates for op in u.patch
                        if op.op == "append"]
            assert appended == ["researcher: calling search_docs"]
            added = next(e.entry for e in seen
                         if isinstance(e, HistoryEntryAdded)
                         and isinstance(e.entry, PublicEffectEntry))
            assert added.detail.kind.value == "subagent"
        finally:
            await session.close()
            await handle.close()
    run(go())


def test_an_agent_that_raises_on_ask_fails_the_turn_and_the_server_goes_on():
    async def go():
        class Broken(FakeSession):
            def ask(self, text):
                if text == "break it":
                    raise RuntimeError("the worker could not start")
                return super().ask(text)

        made = []

        def factory(workflow, sink, cwd=None):
            made.append(Broken(workflow, sink, cwd))
            return made[0]
        session, handle = await start_ovat_app_server(
            "w.yml", session_factory=factory)
        try:
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("break it")
            seen = await events_until(session, is_completed)
            turn = completed_turn(seen)
            assert turn.status is PublicTurnStatus.FAILED
            assert "the worker could not start" in turn.error.message
            await session.enqueue("fine")
            seen = await events_until(session, is_completed)
            assert completed_turn(seen).status is PublicTurnStatus.COMPLETED
        finally:
            await session.close()
            await handle.close()
    run(go())


def test_a_message_id_used_twice_does_not_stop_the_second_turn():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            for _ in range(2):
                async with aclosing(session.act(
                        "again", client_message_id="same-id")) as gen:
                    async for _event in gen:
                        pass
            users = [e for e in session.history
                     if isinstance(e, PublicMessageEntry) and e.role == "user"]
            assert len(users) == 2 and len({u.id for u in users}) == 2
            assert fake.asked == ["again", "again"]
    run(go())


def test_an_at_mention_sent_before_the_agent_is_built_still_finds_the_subagent():
    async def go():
        async with served(hold_build=True) as (session, handle, fake):
            await session.enqueue("@researcher find the policy")
            await poll(lambda: session.turn_active)
            fake.build_gate.set()
            await events_until(session, is_completed)
            assert fake.subagent_calls == [("researcher", "find the policy")]
    run(go())


# ---- interrupting ---------------------------------------------------------------

def test_interrupt_cancels_a_running_turn():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("long one")
            await poll(lambda: fake.busy)
            await poll(lambda: session.turn_active)
            await asyncio.wait_for(session.interrupt(), T)
            seen = await events_until(session, is_completed)
            turn = completed_turn(seen)
            assert turn.status is PublicTurnStatus.INTERRUPTED
            assert fake.cancels >= 1
            assert fake.stop.is_set()
            # the tool call that never reported an end is closed, not left
            # spinning
            settled = [e for e in seen if isinstance(e, HistoryEntryUpdated)
                       and isinstance(e.entry, PublicEffectEntry)]
            assert settled
            assert not any(isinstance(e, HistoryEntryAdded)
                           and isinstance(e.entry, PublicMessageEntry)
                           and e.entry.role == "assistant" for e in seen)
            assert not session.turn_active
    run(go())


def test_interrupt_while_the_agent_is_still_building_ends_the_turn():
    async def go():
        async with served(hold_build=True) as (session, handle, fake):
            await session.enqueue("early")
            await poll(lambda: session.turn_active)
            await asyncio.wait_for(session.interrupt(), T)
            seen = await events_until(session, is_completed)
            assert completed_turn(seen).status is PublicTurnStatus.INTERRUPTED
            fake.build_gate.set()
            await asyncio.sleep(0.1)
            assert fake.asked == []
    run(go())


def test_interrupt_pauses_the_queue_and_resume_runs_the_next_prompt():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("first")
            await poll(lambda: fake.asked == ["first"])
            second = await session.enqueue("second")
            await poll(lambda: session.turn_queue.items)
            await poll(lambda: session.turn_active)
            await session.interrupt()
            seen = await events_until(
                session, lambda e: isinstance(e, TurnQueueUpdated)
                and e.queue.paused)
            assert completed_turn(seen).status is PublicTurnStatus.INTERRUPTED
            await asyncio.sleep(0.2)
            assert fake.asked == ["first"]               # waiting for a yes
            fake.turn_gate.set()
            queue = await session.resume_turn_queue()
            assert not queue.paused
            await events_until(
                session, lambda e: is_completed(e)
                and e.turn.queue_item_id == second.id)
            assert fake.asked == ["first", "second"]
    run(go())


def test_a_stale_or_missing_turn_id_is_a_clear_error():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            client = session._connection.current
            with pytest.raises(AppServerResponseError) as err:
                await client.request("turn/interrupt", {
                    "sessionId": session.session_id,
                    "expectedTurnId": "nothing"})
            assert str(err.value) == "No active turn"
            assert err.value.error.code is ProtocolErrorCode.CONFLICT
            await session.enqueue("first")
            await poll(lambda: fake.busy)
            with pytest.raises(AppServerResponseError) as err:
                await client.request("turn/interrupt", {
                    "sessionId": session.session_id,
                    "expectedTurnId": "other"})
            assert err.value.error.code is ProtocolErrorCode.STALE_TURN
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


# ---- steering ---------------------------------------------------------------------

async def _running_turn(session, fake):
    """Start a turn that stays open; returns its id."""
    await session.resources.runtime.wait_until_ready()
    await session.enqueue("first")
    await poll(lambda: fake.busy and session.state.turns
               and session.state.turns[-1].status.value == "in_progress")
    return session.state.turns[-1].id


def test_steering_a_running_turn_adds_the_message_and_reaches_the_agent():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            turn_id = await _running_turn(session, fake)
            await session.inject_user_context(
                "use the short form", as_message=True,
                client_message_id="steer-1", require_active_turn=True)
            assert fake.steered == ["use the short form"]
            entry = next(e for e in session.history if e.id == "steer-1")
            assert entry.role == "user" and entry.source == "turn_steer"
            assert entry.turn_id == turn_id
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


def test_steer_answers_a_missing_or_stale_turn_and_an_empty_message():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            client = session._connection.current
            body = {"sessionId": session.session_id,
                    "expectedTurnId": "nothing",
                    "message": [{"type": "text", "text": "hi"}]}
            with pytest.raises(AppServerResponseError) as err:
                await client.request("turn/steer", body)
            assert err.value.error.code is ProtocolErrorCode.CONFLICT
            assert str(err.value) == "No active turn"
            turn_id = await _running_turn(session, fake)
            with pytest.raises(AppServerResponseError) as err:
                await client.request("turn/steer", body)
            assert err.value.error.code is ProtocolErrorCode.STALE_TURN
            body.update(expectedTurnId=turn_id,
                        message=[{"type": "text", "text": "  "}])
            with pytest.raises(AppServerResponseError) as err:
                await client.request("turn/steer", body)
            assert err.value.error.code is ProtocolErrorCode.INVALID_PARAMS
            assert fake.steered == []
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


def test_steering_a_framework_engine_or_a_subagent_is_refused_in_plain_words():
    async def go():
        async with served(hold_turn=True, native=False) as (
                session, handle, fake):
            turn_id = await _running_turn(session, fake)
            with pytest.raises(AppServerResponseError) as err:
                await session.inject_user_context(
                    "more", as_message=True, require_active_turn=True)
            assert err.value.error.code is ProtocolErrorCode.NOT_STEERABLE
            assert "native engine" in str(err.value)
            assert [e.id for e in session.history
                    if getattr(e, "source", "") == "turn_steer"] == []
            with pytest.raises(AppServerResponseError) as err:
                await session.steer_queued_turn("no-such-item", turn_id)
            assert err.value.error.code is ProtocolErrorCode.NOT_FOUND
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


def test_a_waiting_prompt_can_be_steered_into_the_running_turn():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            turn_id = await _running_turn(session, fake)
            queued = await session.enqueue("and mention the date")
            answer = await session.steer_queued_turn(queued.id, turn_id)
            assert (answer.queue_item_id, answer.turn_id) == (queued.id,
                                                             turn_id)
            assert fake.steered == ["and mention the date"]
            assert session.turn_queue.items == []
            assert "and mention the date" in _texts(session.history)
            # the item was moved, not run as a turn of its own
            fake.turn_gate.set()
            await events_until(session, is_completed)
            assert fake.asked == ["first"]
            with pytest.raises(AppServerResponseError) as err:
                await session.steer_queued_turn(queued.id, turn_id)
            assert err.value.error.code in (ProtocolErrorCode.CONFLICT,
                                            ProtocolErrorCode.STALE_TURN)
    run(go())


def test_rewinding_to_a_steered_message_cuts_the_memory_right_before_it():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await _running_turn(session, fake)
            await session.inject_user_context(
                "steer me", as_message=True, client_message_id="steer-1",
                require_active_turn=True)
            fake.turn_gate.set()
            await events_until(session, is_completed)
            assert _remembered_questions(fake) == ["first", "steer me"]
            await session.resources.sessions.rewind(
                "steer-1", restore_files=False, inplace=True)
            assert _remembered_questions(fake) == ["first"]
            assert fake.remembered.messages[-1]["role"] == "user"
            assert _texts(session.history) == ["first"]
    run(go())


# ---- approvals ------------------------------------------------------------------

APPROVAL = {"task": "tell me\n1. Allow once\n" + "x" * 900}


@pytest.mark.parametrize("choice, decision, allowed", [
    (ApprovalDecisionType.APPROVE, ONCE, True),
    (ApprovalDecisionType.APPROVE_FOR_SESSION, SESSION, True),
    (ApprovalDecisionType.APPROVE_PERMANENTLY, ALWAYS, True),
    (ApprovalDecisionType.DENY, REFUSE, False),
])
def test_an_approval_reaches_the_broker_with_the_right_decision(
        choice, decision, allowed):
    async def go():
        async with served(approval=APPROVAL) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            events = []

            async def drive():
                async with aclosing(session.act("ask the cloud")) as gen:
                    async for event in gen:
                        events.append(event)
                        if isinstance(event, CallbackRequested):
                            await session.respond_to_callback(
                                event.callback.callback_id,
                                ApprovalCallbackOutput(
                                    decision=ApprovalDecision(type=choice)))
            await asyncio.wait_for(drive(), T)

            asked = next(e.callback for e in events
                         if isinstance(e, CallbackRequested))
            assert asked.title == "Allow search_docs?"
            assert isinstance(asked.detail, ApprovalCallbackDetail)
            assert asked.detail.reason == "sends the question away"
            # the panel gets the call as ONE safe line, cut and marked
            shown = asked.detail.effect.input["task"]
            assert "\n" not in shown and "\u21b5" in shown
            assert "characters in all" in shown
            assert fake.resolved == [(1, decision)]
            assert fake.approved is allowed
            # the callback entry ends answered, and the tool call runs again
            assert any(isinstance(e, HistoryEntryUpdated)
                       and isinstance(e.entry, PublicCallbackEntry)
                       and e.entry.state.status == "answered"
                       for e in events)
            blocked = [e.entry for e in events
                       if isinstance(e, HistoryEntryUpdated)
                       and isinstance(e.entry, PublicEffectEntry)
                       and isinstance(e.entry.state, BlockedEffectState)]
            assert blocked and blocked[0].state.callback_id == \
                asked.callback_id
            final = next(e for e in session.history if e.id == "c1")
            expected = SkippedEffectState if not allowed else \
                CompletedEffectState
            assert isinstance(final.state, expected)
            assert not session.state.active_callbacks
    run(go())


def test_the_panel_offers_always_allow():
    async def go():
        async with served(approval=APPROVAL) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            asked = []

            async def drive():
                async with aclosing(session.act("ask the cloud")) as gen:
                    async for event in gen:
                        if isinstance(event, CallbackRequested):
                            asked.append(event.callback)
                            await session.respond_to_callback(
                                event.callback.callback_id,
                                ApprovalCallbackOutput(decision=ApprovalDecision(
                                    type=ApprovalDecisionType.DENY)))
            await asyncio.wait_for(drive(), T)
            assert ApprovalDecisionType.APPROVE_PERMANENTLY in \
                asked[0].detail.choices
    run(go())


def test_a_session_approval_is_remembered_by_the_broker_for_the_next_call():
    async def go():
        async with served(approval=APPROVAL) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()

            async def one(text):
                async with aclosing(session.act(text)) as gen:
                    async for event in gen:
                        if isinstance(event, CallbackRequested):
                            await session.respond_to_callback(
                                event.callback.callback_id,
                                ApprovalCallbackOutput(decision=ApprovalDecision(
                                    type=ApprovalDecisionType.
                                    APPROVE_FOR_SESSION)))
            await asyncio.wait_for(one("first"), T)
            await asyncio.wait_for(one("second"), T)      # no second prompt
            assert fake.resolved == [(1, SESSION)]
    run(go())


def test_interrupting_closes_an_open_approval_and_denies_it():
    async def go():
        async with served(approval=APPROVAL) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("ask the cloud")
            seen = await events_until(
                session, lambda e: isinstance(e, CallbackRequested))
            callback = seen[-1].callback
            await session.interrupt()
            more = await events_until(session, is_completed)
            assert completed_turn(more).status is PublicTurnStatus.INTERRUPTED
            assert fake.approved is False
            closed = [e.entry for e in more
                      if isinstance(e, HistoryEntryUpdated)
                      and isinstance(e.entry, PublicCallbackEntry)]
            assert closed and closed[0].state.status == "cancelled"
            # a late answer is told the callback is closed, not "unknown"
            with pytest.raises(AppServerResponseError) as err:
                await session.respond_to_callback(
                    callback.callback_id, ApprovalCallbackOutput(
                        decision=ApprovalDecision(
                            type=ApprovalDecisionType.APPROVE)))
            assert err.value.error.code is ProtocolErrorCode.CALLBACK_CLOSED
    run(go())


def test_an_unknown_callback_is_not_found():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            with pytest.raises(AppServerResponseError) as err:
                await session.respond_to_callback(
                    "approval-99", ApprovalCallbackOutput(
                        decision=ApprovalDecision(
                            type=ApprovalDecisionType.APPROVE)))
            assert err.value.error.code is ProtocolErrorCode.NOT_FOUND
    run(go())


def test_a_client_that_cannot_answer_approvals_gets_a_no():
    async def go():
        made = []

        def factory(workflow, sink, cwd=None):
            made.append(FakeSession(workflow, sink, cwd, approval=APPROVAL))
            return made[0]
        session, handle = await start_ovat_app_server(
            "w.yml", session_factory=factory,
            capabilities=ClientCapabilities(callback_kinds=[]))
        try:
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("ask the cloud")
            seen = await events_until(session, is_completed)
            assert made[0].approved is False
            warning = next(e for e in seen if isinstance(e, ServerWarning))
            assert "cannot ask" in warning.params.warning.message
        finally:
            await session.close()
            await handle.close()
    run(go())


# ---- what is not supported ------------------------------------------------------

def test_an_unsupported_method_answers_a_readable_error():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.skills.read_installed()
            assert str(err.value) == "not supported in OVAT: skills/installed"
            assert err.value.error.code is ProtocolErrorCode.NOT_IMPLEMENTED
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.vibe_code.open_projects()
            assert "not supported in OVAT: vibeCode/projects/open" in str(
                err.value)
            # a name the protocol does not know at all
            client = session._connection.current
            with pytest.raises(AppServerResponseError) as err:
                await client.request("nope/nothing", {})
            assert str(err.value) == "not supported in OVAT: nope/nothing"
            assert err.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND
            # and the server is still fine afterwards
            await session.resources.runtime.refresh()
    run(go())


def test_the_quiet_answers_for_account_workspace_and_feedback():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            r = session.resources
            assert (await r.account.read()).status.value == "missing_key"
            assert await r.identity.read() is None
            auth = await r.provider_auth.read()
            assert auth.provider_name == "ovms"
            assert auth.api_base is None
            assert await r.feedback.should_show() is False
            assert (await r.workspace.trust_status()).status == "trusted"
            assert (await r.workspace.untrusted_config_dirs()).dirs == []
            prepared = await r.workspace.prepare_prompt("read @a.md now")
            assert prepared.prompt_text == "read @a.md now"
            assert prepared.display_text == "read @a.md now"
            # accepted and dropped: nothing leaves the process
            r.telemetry.record("vibe.something", {"a": 1})
            await r.telemetry.flush()
            await r.feedback.record("snoozed")
            schema = await r.config.read_schema()
            assert schema.config_schema["title"] == "WorkflowConfig"
            effective = await r.config.read_effective()
            assert effective.config["model"]["name"] == "Qwen3.5-4B-int4-ov"
            assert effective.config["engine"] == "native"
    run(go())


def test_the_effective_config_never_carries_a_secret():
    async def go():
        # A session whose info held more than the banner needs (an MCP
        # server's env): the server forwards only the fields it names.
        secret_info = dict(INFO, tools=[{
            "name": "mcp_tool", "kind": "mcp_stdio", "permission": "allow",
            "env": {"API_KEY": "hunter2"}}])

        class Secretive(FakeSession):
            def start(self, engine=None):
                def go():
                    self.agent = FakeAgent()
                    self.sink(s.Ready(secret_info))
                return self._thread(go)
        session, handle = await start_ovat_app_server(
            "w.yml", session_factory=Secretive)
        try:
            await session.resources.runtime.wait_until_ready()
            effective = await session.resources.config.read_effective()
            assert effective.config["tools"] == [{
                "name": "mcp_tool", "kind": "mcp_stdio",
                "permission": "allow"}]
            assert "hunter2" not in repr(effective.config)
        finally:
            await session.close()
            await handle.close()
    run(go())


def test_settings_update_changes_the_native_engine_or_says_it_cannot():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.resources.sessions.update_settings(
                max_turns=3, max_tokens=512)
            assert fake.agent.max_iterations == 3
            assert fake.agent.llm.max_tokens == 512
        async with served(native=False) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.sessions.update_settings(max_turns=3)
            assert "max_turns" in str(err.value)
    run(go())


def test_the_same_enqueue_sent_twice_queues_once():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("first")
            await poll(lambda: fake.asked == ["first"])
            one = await session.enqueue("again", idempotency_key="k1")
            two = await session.enqueue("again", idempotency_key="k1")
            assert one.id == two.id
            assert len((await session.read_turn_queue()).items) == 1
            with pytest.raises(AppServerResponseError) as err:
                await session.enqueue("different", idempotency_key="k1")
            assert err.value.error.code is ProtocolErrorCode.CONFLICT
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


def test_a_full_queue_and_context_entries_are_refused_with_a_reason():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("first")
            await poll(lambda: fake.asked == ["first"])
            for n in range(32):
                await session.enqueue(f"waiting {n}")
            with pytest.raises(AppServerResponseError) as err:
                await session.enqueue("one too many")
            assert "Turn queue is full (32 items)" in str(err.value)
            client = session._connection.current
            with pytest.raises(AppServerResponseError) as err:
                await client.request("session/turn/enqueue", {
                    "sessionId": session.session_id,
                    "entries": [{"role": "context", "content": [
                        {"type": "text", "text": "background"}]}]})
            assert err.value.error.code is ProtocolErrorCode.INVALID_PARAMS
            assert "Context entries are not supported" in str(err.value)
            fake.turn_gate.set()
    run(go())


def test_small_reads_before_and_after_a_session_exists(monkeypatch):
    # diagnostics/list runs the doctor's checks; this machine's are not the point
    from ovat.cli import diagnostics
    monkeypatch.setattr(diagnostics, "run_checks", lambda config_path=None: [])
    async def go():
        client, handle, made = await raw_client()
        try:
            await handshake(client)
            # the app reads its config before it has a session
            config = await client.request("config/read", {})
            assert config["config"]["activeModel"]["name"] == "w"   # the file
            assert config["config"]["enableUpdateChecks"] is False
            agents = await client.request("agents/list", {})
            assert agents["active"]["name"] == "default"
            # and anything that needs a session says so, readably
            with pytest.raises(AppServerResponseError) as err:
                await client.request("runtime/read", {"sessionId": "x"})
            assert err.value.error.code is ProtocolErrorCode.CONFLICT
            assert "Start, resume, or continue a session" in str(err.value)

            start = await client.request("session/start", {})
            session_id = start["state"]["session"]["id"]
            ready = await client.request("session/ready/read",
                                         {"sessionId": session_id})
            assert ready["ready"] in (True, False)
            await client.request("session/ready/wait", {"sessionId": session_id})
            assert (await client.request("session/ready/read", {
                "sessionId": session_id}))["ready"] is True
            log = await client.request("session/log/read",
                                       {"sessionId": session_id})
            assert log["log"]["sessionId"] == session_id
            assert log["log"]["persisted"] is False
            diag = await client.request("diagnostics/list",
                                        {"sessionId": session_id})
            assert diag == {"issues": [], "hooksCount": 0}
            # resuming the open session returns it; any other id is unknown
            again = await client.request("session/resume",
                                         {"sessionId": session_id})
            assert again["state"]["session"]["id"] == session_id
            with pytest.raises(AppServerResponseError) as err:
                await client.request("session/resume", {"sessionId": "old"})
            assert err.value.error.code is ProtocolErrorCode.NOT_FOUND
            # a second session on the same server is refused
            with pytest.raises(AppServerResponseError) as err:
                await client.request("session/start", {})
            assert "already open" in str(err.value)
        finally:
            await client.close()
            await handle.close()
    run(go())


# ---- history, clear, read -------------------------------------------------------

def test_clearing_the_history_starts_a_fresh_conversation_without_gaps():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("one")
            await events_until(session, is_completed)
            assert len(session.history) == 3
            before_id = session.session_id
            await asyncio.wait_for(session.clear_history(), T)
            assert fake.cleared == 1
            assert session.history == []
            # a new conversation is a new session; the old one stays saved,
            # because the app tells the person how to resume it
            assert session.session_id != before_id
            assert before_id in [
                x.id for x in await session.resources.sessions.list()]
            # the next turn's events still number on from where they were: a
            # gap would make the client resync and the history would double
            await session.enqueue("two")
            await events_until(session, is_completed)
            assert [e.id for e in session.history if isinstance(
                e, PublicMessageEntry) and e.role == "user"] and \
                len(session.history) == 3
            # and the server agrees with the client's mirror
            read = await session.resources.sessions.get_session_history(
                session.session_id)
            assert [e.id for e in read] == [e.id for e in session.history]
    run(go())


def test_clearing_waits_for_a_framework_engine_to_rebuild():
    async def go():
        async with served(rebuild=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await asyncio.wait_for(session.clear_history(), T)
            assert fake.cleared == 1
            assert session.resources.runtime.ready is True
    run(go())


def test_clearing_is_refused_while_a_turn_runs():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("first")
            await poll(lambda: fake.busy)
            await poll(lambda: session.turn_active)
            with pytest.raises(AppServerResponseError) as err:
                await session.clear_history()
            assert "Wait for the current turn" in str(err.value)
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


def test_history_pages_walk_backwards_and_forwards():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            for text in ("one", "two", "three"):
                await session.enqueue(text)
                await events_until(session, is_completed)
            ids = [e.id for e in session.history]
            assert len(ids) == 9
            latest = await session.resources.sessions.list_history(limit=4)
            assert [e.id for e in latest.items] == ids[-4:]
            older = await session.resources.sessions.list_history(
                before=latest.items[0].id, limit=4)
            assert [e.id for e in older.items] == ids[1:5]
            assert older.next_cursor == ids[1]
            oldest = await session.resources.sessions.list_history(
                before=older.items[0].id, limit=4)
            assert [e.id for e in oldest.items] == ids[:1]
            assert oldest.next_cursor is None
            newer = await session.resources.sessions.list_history(
                after=ids[3], limit=2)
            assert [e.id for e in newer.items] == ids[4:6]
    run(go())


def test_history_page_cursors():
    class E:                                      # an entry with just an id
        def __init__(self, id, turn_id=None):
            self.id, self.turn_id = id, turn_id
    entries = [E(str(i), "t" if i % 2 else None) for i in range(6)]
    page, older, newer = history_page(entries, turn_id=None, before=None,
                                      after=None, limit=2)
    assert [e.id for e in page] == ["4", "5"] and older == "4" and \
        newer is None
    page, older, newer = history_page(entries, turn_id="t", before=None,
                                      after=None, limit=10)
    assert [e.id for e in page] == ["1", "3", "5"] and older is None


def test_session_read_returns_the_state_and_other_ids_are_not_found():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("hi")
            await events_until(session, is_completed)
            state = await session.refresh_state() \
                if hasattr(session, "refresh_state") else session.state
            assert state.session.id == session.session_id
            assert state.session.preview == "hi"
            assert state.session.model == "Qwen3.5-4B-int4-ov"
            assert state.session.status.type == "idle"
            assert [t.status for t in state.turns] == [
                PublicTurnStatus.COMPLETED]
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.sessions.get_session_history("nope")
            assert err.value.error.code is ProtocolErrorCode.NOT_FOUND
            # saved after its turn, so it is the one to continue
            assert [x.id for x in await session.resources.sessions.list()] \
                == [session.session_id]
            assert await session.resources.sessions.resolve_continue_session(
            ) == session.session_id
    run(go())


# ---- the `!` shell --------------------------------------------------------------

def test_shell_lines_arrive_as_one_entry_that_grows_and_completes():
    async def go():
        async with served(shell_lines=("alpha", "beta")) as (
                session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            events = []

            async def drive():
                async for event in session.resources.shell.run("echo hi"):
                    events.append(event)
            await asyncio.wait_for(drive(), T)

            first = events[0]
            assert isinstance(first, HistoryEntryAdded)
            assert first.entry.detail.input.command == "echo hi"
            assert isinstance(first.entry.state, RunningEffectState)
            final = events[-1].entry
            assert isinstance(final.state, CompletedEffectState)
            assert final.state.output_text == "alpha\nbeta\n"
            assert final.state.output["stdout"] == "alpha\nbeta\n"
            streamed = "".join(op.value for e in events
                               if isinstance(e, HistoryEntryUpdated)
                               for op in e.patch if op.op == "append")
            assert streamed == "alpha\nbeta\n"
            assert all(e.entry.id == first.entry.id for e in events)
    run(go())


def test_a_failing_shell_command_ends_failed_with_its_status():
    async def go():
        async with served(shell_code=3) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            events = [e async for e in session.resources.shell.run("false")]
            final = events[-1].entry
            assert isinstance(final.state, FailedEffectState)
            assert "status 3" in final.state.error.message
    run(go())


def test_interrupting_a_shell_command_ends_it_cancelled():
    async def go():
        async with served(hold_shell=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            events = []

            async def drive():
                async for event in session.resources.shell.run("sleep 99"):
                    events.append(event)
            task = asyncio.create_task(drive())
            await poll(lambda: fake.shell_busy)
            client = session._connection.current
            operation_id = events[0].entry.id if events else None
            if operation_id is None:
                await poll(lambda: events)
                operation_id = events[0].entry.id
            await client.request("session/shellCommand", {
                "sessionId": session.session_id, "action": "interrupt",
                "operationId": operation_id})
            await asyncio.wait_for(task, T)
            assert isinstance(events[-1].entry.state, CancelledEffectState)
            assert fake.shell_killed.is_set()
    run(go())


def test_a_second_shell_command_while_one_runs_is_refused():
    async def go():
        async with served(hold_shell=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            events = []

            async def drive():
                async for event in session.resources.shell.run("sleep 99"):
                    events.append(event)
            task = asyncio.create_task(drive())
            await poll(lambda: fake.shell_busy)
            with pytest.raises(AppServerResponseError) as err:
                async for _ in session.resources.shell.run("echo again"):
                    pass
            assert err.value.error.code is ProtocolErrorCode.CONFLICT
            fake.shell_gate.set()
            await asyncio.wait_for(task, T)
    run(go())


def test_a_cancel_that_lands_before_the_worker_starts_is_repeated():
    """AgentSession clears its stop flag as a turn starts, so an Esc pressed
    in that gap would be lost. The server cancels again once the session
    reports TurnStarted."""
    async def go():
        async with served(hold_start=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("early")
            await poll(lambda: fake.asked == ["early"])
            await poll(lambda: session.turn_active)
            await asyncio.wait_for(session.interrupt(), T)
            assert fake.cancels == 1                    # the first, too early
            fake.start_gate.set()                       # the worker starts
            seen = await events_until(session, is_completed)
            assert fake.cancels == 2                    # and the repeat
            assert completed_turn(seen).status is PublicTurnStatus.INTERRUPTED
    run(go())


def test_a_command_with_a_lot_of_output_is_sent_in_batches():
    async def go():
        lines = tuple(f"line {n}" for n in range(300))
        async with served(shell_lines=lines) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            events = [e async for e in session.resources.shell.run("seq")]
            appends = [op.value for e in events
                       if isinstance(e, HistoryEntryUpdated)
                       for op in e.patch if op.op == "append"]
            assert "".join(appends) == "".join(f"{l}\n" for l in lines)
            assert len(appends) < len(lines)            # not one per line
    run(go())


# ---- the wire itself ------------------------------------------------------------

async def raw_client(**script):
    made = []

    def factory(workflow, sink, cwd=None):
        made.append(FakeSession(workflow, sink, cwd, **script))
        return made[0]
    transport, handle = serve_on_memory("w.yml", session_factory=factory)
    client = AppServerClient(transport)
    return client, handle, made


async def handshake(client, callbacks=("approval",)):
    await client.start()
    reply = await client.initialize(
        ClientInfo(name="test", version="0"),
        ClientCapabilities(callback_kinds=list(callbacks)))
    await client.notify("initialized")
    return reply


def test_requests_before_the_handshake_are_refused():
    async def go():
        client, handle, made = await raw_client()
        try:
            await client.start()
            with pytest.raises(AppServerResponseError) as err:
                await client.request("session/start", {})
            assert err.value.error.code is ProtocolErrorCode.NOT_INITIALIZED
            reply = await client.initialize(ClientInfo(name="t", version="0"))
            assert reply.server_info.name == "ovat-app-server"
            assert reply.process_id == os.getpid()
            with pytest.raises(AppServerResponseError) as err:
                await client.initialize(ClientInfo(name="t", version="0"))
            assert err.value.error.code is ProtocolErrorCode.INVALID_REQUEST
        finally:
            await client.close()
            await handle.close()
    run(go())


def test_bad_params_come_back_as_invalid_params_with_the_field_path():
    async def go():
        client, handle, made = await raw_client()
        try:
            await handshake(client)
            with pytest.raises(AppServerResponseError) as err:
                await client.request("session/start", {
                    "agentConfig": {"cwd": ""}, "historyLimit": 0})
            error = err.value.error
            assert error.code is ProtocolErrorCode.INVALID_PARAMS
            paths = {tuple(i["path"]) for i in error.data["issues"]}
            assert ("historyLimit",) in paths
            assert ("agentConfig", "cwd") in paths
            assert "historyLimit" in str(err.value)
            assert made == []                    # no session was made
        finally:
            await client.close()
            await handle.close()
    run(go())


def test_the_wire_shapes_match_what_vibes_server_sends():
    """camelCase keys, an `error` of exactly code/message/data, string
    ids from the client and integer ids from the server."""
    async def go():
        client, handle, made = await raw_client()
        try:
            await handshake(client)
            start = await client.request("session/start", {})
            assert set(start) == {"state", "lastEventId"}
            state = start["state"]
            assert state["format"] == "vibe.public-session-state/v1"
            assert state["eventId"] == 0
            assert state["session"]["status"] == {"type": "idle"}
            assert "createdAt" in state["session"]
            assert "turnQueue" in state and "activeCallbacks" in state
            # an error is exactly {code, message, data}
            raw = {}
            try:
                await client.request("nope/nothing", {})
            except AppServerResponseError as exc:
                raw = exc.error.model_dump(mode="json", by_alias=True)
            assert set(raw) == {"code", "message", "data"}
            assert raw["code"] == "method_not_found"
        finally:
            await client.close()
            await handle.close()
    run(go())


def test_every_event_is_numbered_one_after_the_other():
    async def go():
        client, handle, made = await raw_client(approval=APPROVAL)
        try:
            await handshake(client)
            start = await client.request("session/start", {})
            session_id = start["state"]["session"]["id"]
            await client.request("session/ready/wait",
                                 {"sessionId": session_id})
            await client.request("session/turn/enqueue", {
                "sessionId": session_id,
                "entries": [{"role": "user", "content": [
                    {"type": "text", "text": "go"}]}]})
            numbers, methods = [], []
            async for message in client.incoming():
                if isinstance(message, ServerRequest):
                    assert isinstance(message.id, int)         # server ids
                    assert message.method == "callback/call"
                    callback = message.params["callback"]
                    await client.respond(message.id, {
                        "callbackId": callback["callbackId"],
                        "accepted": True})
                    await client.request("callback/result", {
                        "sessionId": session_id, "result": {
                            "callbackId": callback["callbackId"],
                            "output": {"type": "approval",
                                       "decision": {"type": "approve"}}}})
                    continue
                assert isinstance(message, Notification)
                methods.append(message.method)
                if "eventId" in message.params:
                    numbers.append(message.params["eventId"])
                if message.method == "turn/completed":
                    break
            assert numbers == list(range(1, len(numbers) + 1))
            assert {"turn/queueUpdated", "turn/started", "history/entryAdded",
                    "history/entryUpdated", "session/statsUpdated",
                    "turn/completed"} <= set(methods)
        finally:
            await client.close()
            await handle.close()
    run(go())


def test_the_sessions_new_conversation_notice_is_not_shown_as_a_warning():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await asyncio.wait_for(session.clear_history(), T)
            await session.enqueue("two")
            seen = await events_until(session, is_completed)
            assert not any(isinstance(e, ServerWarning) for e in seen)
    run(go())


def test_other_notices_reach_the_client_as_warnings_and_errors():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            fake.sink(s.Notice("warn", "a2a server stopped"))
            fake.sink(s.Notice("info", "a2a: serving"))
            fake.sink(s.Notice("error", "it broke"))
            seen = await events_until(
                session, lambda e: type(e).__name__ == "ServerError")
            warnings = [e.params.warning for e in seen
                        if isinstance(e, ServerWarning)]
            assert [(w.message, w.code) for w in warnings] == [
                ("a2a server stopped", None), ("a2a: serving", "info")]
            assert seen[-1].params.error.message == "it broke"
    run(go())


def test_images_are_dropped_with_a_warning_and_the_text_still_runs():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            image = ImageAttachment(source=InlineImageSource(data="eA=="),
                                    alias="a.png", mime_type="image/png")
            events = []
            async with aclosing(session.act("look", images=[image])) as gen:
                async for event in gen:
                    events.append(event)
            assert any(isinstance(e, ServerWarning)
                       and "images" in e.params.warning.message
                       for e in events)
            assert fake.asked == ["look"]
    run(go())


def test_an_agent_that_cannot_start_answers_not_ready_instead_of_hanging():
    async def go():
        async with served(start_raises=True) as (session, handle, fake):
            await asyncio.wait_for(
                session.resources.runtime.wait_until_ready(), T)
            assert not session.resources.runtime.ready
            await session.enqueue("hello")
            seen = await events_until(session, is_completed)
            assert "Could not start the agent" in completed_turn(
                seen).error.message
    run(go())


def test_a_client_that_refuses_the_callback_delivery_denies_the_approval():
    async def go():
        client, handle, made = await raw_client(approval=APPROVAL)
        try:
            await handshake(client)
            start = await client.request("session/start", {})
            session_id = start["state"]["session"]["id"]
            await client.request("session/ready/wait",
                                 {"sessionId": session_id})
            await client.request("session/turn/enqueue", {
                "sessionId": session_id,
                "entries": [{"role": "user", "content": [
                    {"type": "text", "text": "go"}]}]})
            async for message in client.incoming():
                if isinstance(message, ServerRequest):
                    await client.respond(message.id, {
                        "callbackId": message.params["callback"]["callbackId"],
                        "accepted": False})
                if (isinstance(message, Notification)
                        and message.method == "turn/completed"):
                    break
            assert made[0].approved is False
            assert made[0].resolved == [(1, REFUSE)]
        finally:
            await client.close()
            await handle.close()
    run(go())


def test_a_client_can_switch_off_a_kind_of_notification():
    async def go():
        seen = {}
        for label, disabled in (("on", []), ("off", ["runtime/updated"])):
            client, handle, made = await raw_client()
            try:
                await client.start()
                await client.initialize(
                    ClientInfo(name="t", version="0"),
                    ClientCapabilities(callback_kinds=["approval"],
                                       disabled_notifications=disabled))
                await client.notify("initialized")
                start = await client.request("session/start", {})
                session_id = start["state"]["session"]["id"]
                await client.request("session/ready/wait",
                                     {"sessionId": session_id})
                # a request is a barrier: everything sent before its answer
                # is already queued on the client
                await client.request("runtime/read", {"sessionId": session_id})
                methods = []
                incoming = client.incoming()
                try:
                    while True:
                        try:
                            message = await asyncio.wait_for(
                                incoming.__anext__(), 0.3)
                        except asyncio.TimeoutError:
                            break
                        methods.append(message.method)
                finally:
                    await incoming.aclose()
                seen[label] = methods
            finally:
                await client.close()
                await handle.close()
        assert "runtime/updated" in seen["on"]
        assert "runtime/updated" not in seen["off"]
        assert "session/updated" in seen["off"]       # events never switch off
    run(go())


def test_closing_with_an_approval_open_leaves_nothing_waiting():
    async def go():
        async with served(approval=APPROVAL) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("ask the cloud")
            await events_until(
                session, lambda e: isinstance(e, CallbackRequested))
        assert fake.closed
        await poll(lambda: not _leftover_threads())
        assert fake.approved is False
    run(go())


def test_the_bridge_names_the_session_and_the_turn():
    from ovat.tui.bridge import Bridge as ClassicBridge
    from ovat.tui.server.bridge import Bridge
    assert ClassicBridge is Bridge            # the classic app's import path
    bridge = Bridge("s1")
    bridge.turn_id = "t1"
    added = bridge.convert(Event(TOOL_STARTED, {
        "call_id": "k1", "name": "search_docs", "arguments": {"q": "x"}}), {})
    assert (added.entry.session_id, added.entry.turn_id) == ("s1", "t1")
    assert bridge.running("search_docs").id == "k1"
    assert bridge.running("other") is None
    assert bridge.answer("hi").entry.turn_id == "t1"
    assert bridge.reasoning("why").entry.text == "why"
    plain = Bridge()                          # the classic app's use
    assert plain.answer("x").entry.session_id == "ovat"
    assert plain.answer("x").entry.turn_id is None


def _bridge_run(tmp_path, monkeypatch, name, arguments):
    """Run one real coding tool and return the card the bridge settles."""
    from ovat.agent import factory
    from ovat.agent.events import EventBus
    from ovat.config.workflow import WorkflowConfig
    from ovat.tui.server.bridge import Bridge
    monkeypatch.chdir(tmp_path)
    bus = EventBus()
    bus.approver = lambda tool, args, reason: True
    events = []
    bus.subscribe(events.append)
    tools = factory.observe_tools(factory.build_tools(
        WorkflowConfig(model={"name": "m"}, tools=[{"name": name}]),
        events=bus), bus)
    tools[name]["function"](**arguments)
    bridge = Bridge("s1")
    started = bridge.convert(events[0], {})
    updated = bridge.convert(events[-1], {})
    # What goes over the wire must survive being written out as JSON.
    updated.entry.model_dump_json(by_alias=True)
    return started.entry.detail, updated.entry.state


def test_an_edit_reaches_the_diff_card_with_its_occurrences(
        tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("a = 1\nb = 2\n")
    detail, state = _bridge_run(tmp_path, monkeypatch, "edit", {
        "file_path": "a.py", "old_string": "b = 2", "new_string": "b = 3"})
    assert detail.kind == "file_edit"
    assert detail.input.old_string == "b = 2"
    assert state.status == "completed"
    # The wire form is camelCase, like every Vibe effect output.
    assert state.output["occurrences"] == [
        {"startLine": 2, "oldText": "b = 2", "newText": "b = 3"}]
    assert state.display.message == "a.py"


def test_a_shell_command_reaches_the_card_with_its_output(
        tmp_path, monkeypatch):
    detail, state = _bridge_run(tmp_path, monkeypatch, "shell", {
        "command": f'"{sys.executable}" -c "print(42)"'})
    assert detail.kind == "shell" and "print(42)" in detail.input.command
    assert state.status == "completed"
    assert state.output["stdout"].strip() == "42"
    assert state.output["output"].strip() == "42"


def test_a_shell_command_that_fails_keeps_its_output_on_the_failed_card(
        tmp_path, monkeypatch):
    detail, state = _bridge_run(tmp_path, monkeypatch, "shell", {
        "command": f'"{sys.executable}" -c "import sys; print(1); '
                   f'sys.exit(2)"'})
    assert state.status == "failed"
    assert "code 2" in state.error.message
    assert state.output["stdout"].strip() == "1"


def test_write_read_and_grep_cards_carry_their_outputs(tmp_path, monkeypatch):
    detail, state = _bridge_run(tmp_path, monkeypatch, "write_file", {
        "file_path": "n/new.txt", "content": "hi"})
    assert detail.kind == "file_write"
    assert state.output == {"filePath": "n/new.txt", "content": "hi",
                            "fileExisted": False, "previousContent": None}
    (tmp_path / "r.txt").write_text("one\ntwo\n")
    detail, state = _bridge_run(tmp_path, monkeypatch, "read_file", {
        "file_path": "r.txt"})
    assert detail.kind == "file_read" and state.output["numLines"] == 2
    detail, state = _bridge_run(tmp_path, monkeypatch, "grep", {
        "pattern": "two"})
    assert detail.kind == "file_search"
    assert state.output["matchCount"] == 1
    assert state.output["parsedMatches"][0]["line"] == 2


def test_a_refused_coding_call_shows_the_sentence_not_a_traceback(
        tmp_path, monkeypatch):
    detail, state = _bridge_run(tmp_path, monkeypatch, "read_file", {
        "file_path": "../outside.txt"})
    assert state.status == "failed"
    assert "outside the project folder" in state.error.message


# ---- shutdown -------------------------------------------------------------------

def _leftover_threads():
    return [t.name for t in threading.enumerate()
            if t.name.startswith(("fake-", "ovat-"))
            and t is not threading.current_thread()]


def test_shutdown_leaves_no_task_and_no_thread_running():
    async def go():
        before = {t for t in asyncio.all_tasks()}
        async with served(hold_turn=True, hold_shell=True) as (
                session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("first")           # a turn left running
            await poll(lambda: fake.busy)
        # served() closed the session and the handle
        assert fake.closed
        assert not handle.running
        await poll(lambda: not _leftover_threads())
        rest = {t for t in asyncio.all_tasks()} - before \
            - {asyncio.current_task()}
        assert rest == set(), rest
    run(go())


def test_session_stop_ends_the_server_even_while_the_client_stays_connected():
    async def go():
        client, handle, made = await raw_client()
        try:
            await handshake(client)
            start = await client.request("session/start", {})
            session_id = start["state"]["session"]["id"]
            await client.request("session/ready/wait", {"sessionId": session_id})
            stopped = await client.request("session/stop",
                                           {"sessionId": session_id})
            assert stopped == {"closed": True}        # the answer still came
            await poll(lambda: not handle.running)
            assert made[0].closed
        finally:
            await client.close()
            await handle.close()
    run(go())


def test_session_stop_alone_ends_the_server_and_closes_the_agent():
    async def go():
        made = []

        def factory(workflow, sink, cwd=None):
            made.append(FakeSession(workflow, sink, cwd, hold_turn=True))
            return made[0]
        session, handle = await start_ovat_app_server(
            "w.yml", session_factory=factory)
        await session.resources.runtime.wait_until_ready()
        await session.enqueue("long one")
        await poll(lambda: made[0].busy)
        await asyncio.wait_for(session.close(), T)
        # nobody calls handle.close(): stopping the session is enough
        await poll(lambda: not handle.running)
        assert made[0].closed
        await poll(lambda: not _leftover_threads())
        await handle.close()
    run(go())


def test_a_client_that_just_disconnects_also_ends_the_server():
    async def go():
        client, handle, made = await raw_client()
        await handshake(client)
        await client.request("session/start", {})
        await poll(lambda: made)
        await client.close()                # no session/stop
        await poll(lambda: not handle.running)
        assert made[0].closed
        await poll(lambda: not _leftover_threads())
    run(go())


def test_closing_the_handle_alone_also_stops_everything():
    async def go():
        made = []

        def factory(workflow, sink, cwd=None):
            made.append(FakeSession(workflow, sink, cwd, hold_turn=True))
            return made[0]
        session, handle = await start_ovat_app_server(
            "w.yml", session_factory=factory)
        await session.resources.runtime.wait_until_ready()
        await session.enqueue("first")
        await poll(lambda: made[0].busy)
        await asyncio.wait_for(handle.close(), T)         # the client is alive
        assert made[0].closed and not handle.running
        await asyncio.wait_for(handle.close(), T)         # and twice is fine
        await poll(lambda: not _leftover_threads())
        session.begin_close()
        await session.close()
    run(go())


def test_the_server_does_not_import_textual():
    code = ("import sys\n"
            "import ovat.tui.server, ovat.tui.server.methods\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('textual', 'pyfiglet', 'rich_pixels'))\n"
            "print('BAD', bad)\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(__file__).resolve().parent.parent),
         os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=120)
    assert "BAD []" in out.stdout, out.stdout + out.stderr


def test_streamed_reasoning_and_answer_deltas_become_incremental_history_patches():
    async def go():
        async with served(
                tool=False,
                reasoning_deltas=("check ", "handbook"),
                answer_deltas=("New hires ", "get 20 ", "days."),
                reasoning="<think>check handbook</think>",
                answer="New hires get 20 days.") as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("How many days?")
            seen = await events_until(session, is_completed)
            added = [e.entry for e in seen if isinstance(e, HistoryEntryAdded)]
            updated = [e for e in seen if isinstance(e, HistoryEntryUpdated)]
            assistants = [e for e in session.history
                          if isinstance(e, PublicMessageEntry)
                          and e.role == "assistant"]
            reasonings = [e for e in session.history
                          if isinstance(e, PublicReasoningEntry)]
            assert len(assistants) == 1
            assert len(reasonings) == 1
            assert assistants[0].text == "New hires get 20 days."
            assert reasonings[0].text == "check handbook"
            assert any(isinstance(e, PublicMessageEntry)
                       and e.role == "assistant"
                       and e.generation_status.value == "in_progress"
                       for e in added)
            assert any(any(op.op == "append" and op.path == "/content/0/text"
                           for op in u.patch) for u in updated)
    run(go())


def test_context_window_is_published_in_runtime_and_stats_updates():
    async def go():
        async with served(context_window=32768) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            runtime = session.resources.runtime
            await runtime.refresh()
            assert runtime.context_window == 32768
            assert (session.resources.config.current
                    .active_model.max_context_length == 32768)
            await session.enqueue("How many days?")
            seen = await events_until(session, is_completed)
            stats_ev = next(e for e in seen if isinstance(e, StatsUpdated))
            assert stats_ev.params.context_window == 32768
            assert stats_ev.params.stats.context_tokens == 100
            stats = await session._connection.current.request(
                "stats/read", {"sessionId": session.session_id})
            assert stats["contextWindow"] == 32768
    run(go())




# ---- saved conversations ---------------------------------------------------------

def _texts(history, role="user"):
    return [e.text for e in history
            if isinstance(e, PublicMessageEntry) and e.role == role]


def _remembered_questions(fake):
    return [m["content"] for m in fake.remembered.messages
            if m["role"] == "user"]


async def _ask(session, text):
    await session.enqueue(text)
    await events_until(session, is_completed)


def test_a_conversation_is_saved_after_each_turn_and_reopens_on_a_new_server(
        saved_sessions):
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            await _ask(session, "two")
            first_id = session.session_id
            entries = [e.id for e in session.history]
            remembered = fake.memory()
        assert saved_sessions.load(first_id)["memory"] == remembered
        made = []

        def factory(workflow, sink, cwd=None):
            made.append(FakeSession(workflow, sink, cwd))
            return made[0]
        # the short id the app prints after /clear is enough
        session, handle = await asyncio.wait_for(start_ovat_app_server(
            "w.yml", session_factory=factory,
            resume_session_id=first_id[:8]), T)
        try:
            assert session.session_id == first_id
            assert [e.id for e in session.history] == entries
            await session.resources.runtime.wait_until_ready()
            # the memory goes in once the agent is built to take it
            await poll(lambda: made[0].remembered.snapshot() == remembered)
            await _ask(session, "three")
            assert _texts(session.history) == ["one", "two", "three"]
        finally:
            await asyncio.wait_for(session.close(), T)
            await asyncio.wait_for(handle.close(), T)
    run(go())


def test_continue_opens_the_most_recent_conversation(saved_sessions):
    for sid, stamp in (("older", 1), ("newer", 2)):
        saved_sessions.save({"id": sid, "title": f"the {sid} one",
                             "updated_at": stamp, "history": [], "turns": [],
                             "memory": [], "anchors": {}})

    async def go():
        session, handle = await asyncio.wait_for(start_ovat_app_server(
            "w.yml", session_factory=FakeSession, continue_session=True), T)
        try:
            assert session.session_id == "newer"
            assert session.state.session.title == "the newer one"
        finally:
            await asyncio.wait_for(session.close(), T)
            await asyncio.wait_for(handle.close(), T)
    run(go())


def test_rewind_cuts_the_memory_where_that_turn_began_even_after_a_subagent():
    """`@researcher ...` adds a user entry to the transcript and nothing to
    the model's memory, so the third user entry is not the third user
    message. Rewinding to "two" must leave the model remembering "one"."""
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            for text in ("one", "@researcher look it up", "two", "three"):
                await _ask(session, text)
            assert fake.subagent_calls == [("researcher", "look it up")]
            target = next(e for e in session.history
                          if isinstance(e, PublicMessageEntry)
                          and e.text == "two")
            result = await session.resources.sessions.rewind(
                target.id, restore_files=False, inplace=True)
            assert result.message == "two"
            assert _texts(session.history) == ["one",
                                               "@researcher look it up"]
            assert _remembered_questions(fake) == ["one"]
            assert fake.remembered.messages[-1]["role"] == "assistant"
            # and the conversation goes on from there
            await _ask(session, "two again")
            assert _remembered_questions(fake) == ["one", "two again"]
    run(go())


def test_rewind_into_a_new_session_keeps_the_original_saved(saved_sessions):
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            await _ask(session, "two")
            original = session.session_id
            target = next(
                e for e in session.history
                if isinstance(e, PublicMessageEntry) and e.text == "two")
            await session.resources.sessions.rewind(
                target.id, restore_files=False, inplace=False)
            assert session.session_id != original
            assert _texts(saved.history_of(saved_sessions.load(original))) \
                == ["one", "two"]
            assert _texts(saved.history_of(
                saved_sessions.load(session.session_id))) == ["one"]
    run(go())


def test_rewinding_to_the_first_message_in_place_forgets_the_saved_file(
        saved_sessions):
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            sid = session.session_id
            assert saved_sessions.load(sid) is not None
            first = session.history[0]
            await session.resources.sessions.rewind(
                first.id, restore_files=False, inplace=True)
            assert session.history == []
            assert _remembered_questions(fake) == []
            assert saved_sessions.load(sid) is None
    run(go())


def test_rewind_is_refused_on_an_engine_whose_memory_ovat_cannot_cut():
    async def go():
        async with served(native=False) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            before = [e.id for e in session.history]
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.sessions.rewind(
                    session.history[0].id, restore_files=False, inplace=True)
            assert err.value.error.code is ProtocolErrorCode.NOT_IMPLEMENTED
            assert "native engine" in str(err.value)
            assert [e.id for e in session.history] == before
    run(go())


def _checkpoints(session):
    return [e for e in session.history if e.type == "checkpoint"]


def test_compact_replaces_the_memory_and_leaves_a_checkpoint(saved_sessions):
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            await _ask(session, "two")
            texts = _texts(session.history)
            summary = await session.compact("keep the numbers")
            assert summary == "they asked: one, two"
            assert fake.compacts == ["keep the numbers"]
            # the model remembers a summary now, not the two questions
            assert _remembered_questions(fake)[0].startswith(
                "You are continuing a trajectory")
            assert len(fake.memory()) == 3      # system, summary, ack
            # the person's transcript is untouched, with a checkpoint added
            assert _texts(session.history) == texts
            (mark,) = _checkpoints(session)
            assert (mark.kind, mark.message) == ("compaction",
                                                 "Context compacted")
            assert mark.details == {"summaryLength": len(summary)}
            # and it is saved that way, with no rewind anchors left
            rec = saved_sessions.load(session.session_id)
            assert rec["memory"] == fake.memory() and rec["anchors"] == {}
            assert [e["type"] for e in rec["history"]].count("checkpoint") == 1
            await _ask(session, "three")        # the conversation goes on
            assert _remembered_questions(fake)[-1] == "three"
    run(go())


def test_compact_that_fails_says_why_and_changes_nothing():
    async def go():
        async with served(compact_error="The model returned an empty "
                                        "summary.") as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            memory, history = fake.memory(), [e.id for e in session.history]
            with pytest.raises(AppServerResponseError) as err:
                await session.compact()
            assert err.value.error.code is ProtocolErrorCode.COMPACTION_FAILED
            assert "empty summary" in str(err.value)
            assert fake.memory() == memory
            assert [e.id for e in session.history] == history
    run(go())


def test_compact_is_refused_on_a_framework_engine_and_during_a_turn():
    async def go():
        async with served(native=False) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            with pytest.raises(AppServerResponseError) as err:
                await session.compact()
            assert err.value.error.code is ProtocolErrorCode.COMPACTION_FAILED
            assert "native engine" in str(err.value)
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("long one")
            await events_until(session, lambda e: isinstance(e, TurnStarted))
            with pytest.raises(AppServerResponseError) as err:
                await session.compact()
            assert err.value.error.code is ProtocolErrorCode.CONFLICT
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


def test_rewind_after_a_compaction_works_and_before_it_is_refused():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            before = next(e for e in session.history
                          if isinstance(e, PublicMessageEntry))
            await session.compact()
            await _ask(session, "two")
            await _ask(session, "three")
            memory = fake.memory()
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.sessions.rewind(
                    before.id, restore_files=False, inplace=True)
            assert err.value.error.code is ProtocolErrorCode.INVALID_PARAMS
            assert "compacted" in str(err.value)
            assert fake.memory() == memory      # refused, nothing cut
            target = next(e for e in session.history
                          if isinstance(e, PublicMessageEntry)
                          and e.text == "three")
            await session.resources.sessions.rewind(
                target.id, restore_files=False, inplace=True)
            # back to the summary and "two"
            assert _remembered_questions(fake)[1:] == ["two"]
            assert _remembered_questions(fake)[0].startswith(
                "You are continuing")
    run(go())


def test_branch_saves_a_copy_and_leaves_the_open_conversation_alone(
        saved_sessions):
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            await _ask(session, "two")
            before = [e.id for e in session.history]
            memory = fake.memory()
            response = await session.resources.sessions.fork(
                entry_id=None, attach=False)
            copy_id = response.state.session.id
            assert copy_id != session.session_id
            assert [e.id for e in session.history] == before
            assert fake.memory() == memory
            rec = saved_sessions.load(copy_id)
            assert _texts(saved.history_of(rec)) == ["one", "two"]
            assert rec["memory"] == memory
    run(go())


def test_rename_and_delete_change_what_is_saved(saved_sessions):
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            sid = session.session_id
            await session.resources.sessions.rename("Leave policy")
            assert saved_sessions.load(sid)["title"] == "Leave policy"
            listed = await session.resources.sessions.list()
            assert [(x.id, x.title) for x in listed] == [
                (sid, "Leave policy")]
            await session.resources.sessions.delete(sid)
            assert saved_sessions.load(sid) is None
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.sessions.delete("nope")
            assert err.value.error.code is ProtocolErrorCode.NOT_FOUND
    run(go())


def test_resume_switches_conversations_and_keeps_the_open_one_saved():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "first chat")
            first = session.session_id
            await asyncio.wait_for(session.clear_history(), T)
            await _ask(session, "second chat")
            second = session.session_id
            await asyncio.wait_for(session.resume(first), T)
            assert session.session_id == first
            assert _texts(session.history) == ["first chat"]
            assert _remembered_questions(fake) == ["first chat"]
            assert {first, second} <= {
                x.id for x in await session.resources.sessions.list()}
            # and it carries on there, numbered without a gap
            await _ask(session, "more")
            assert _texts(session.history) == ["first chat", "more"]
    run(go())


def test_a_save_that_fails_is_said_once_and_the_conversation_goes_on(
        saved_sessions, monkeypatch):
    def full(record):
        raise OSError("disk full")
    monkeypatch.setattr(saved_sessions, "save", full)

    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            seen = []
            # a turn's save (and so its warning) comes just after the turn
            # completes: the third turn is there to collect the second's
            for text in ("one", "two", "three"):
                await session.enqueue(text)
                seen += await events_until(session, is_completed)
            said = [e.params.warning.message for e in seen
                    if isinstance(e, ServerWarning)]
            assert len(said) == 1 and "disk full" in said[0]
            assert _texts(session.history) == ["one", "two", "three"]
    run(go())


# ---- /model and /thinking (config/model/write) ----------------------------------

def _config(session):
    return session.resources.config.current


def test_the_runtime_offers_the_models_the_session_can_switch_to():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            config = _config(session)
            assert [m.alias for m in config.models] == [
                "Qwen3.5-4B-int4-ov", "Qwen3-8B-int4-ov"]
            assert config.active_model.alias == "Qwen3.5-4B-int4-ov"
            assert not config.active_model_pinned
            assert config.active_model.thinking_levels == ["off", "high"]
            assert config.active_model.thinking == "off"
    run(go())


def test_no_setting_is_fixed_by_an_administrator():
    """The /model picker asks which fields are fixed before it opens. With no
    workflow file only the theme is listed, and the one layer a write may
    name is the workflow."""
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            response = await session.resources.config.read_fields()
            assert [f.name for f in response.fields] == ["theme"]
            assert response.targets == ["workflow"]
            assert all(f.origin != "admin" for f in response.fields)
    run(go())


def test_picking_a_model_builds_again_and_the_conversation_survives(
        saved_sessions, tmp_path):
    async def go():
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            await _ask(session, "two")
            status = await session.resources.config.write_model(
                model_alias="Qwen3-8B-int4-ov")
            assert status.name == "APPLIED"
            assert fake.picks == [(None, None)]
            config = _config(session)
            assert config.active_model.alias == "Qwen3-8B-int4-ov"
            assert not config.active_model_pinned
            # the new agent remembers what the old one did, and the
            # transcript is where it was
            assert _remembered_questions(fake) == ["one", "two"]
            assert _texts(session.history) == ["one", "two"]
            await _ask(session, "three")
            assert _remembered_questions(fake) == ["one", "two", "three"]
    run(go())


def test_the_rewind_points_stay_valid_after_a_model_switch(saved_sessions, tmp_path):
    async def go():
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            await _ask(session, "two")
            before = dict(handle._ctx.anchors)
            assert len(before) == 2
            await session.resources.config.write_model(
                model_alias="Qwen3-8B-int4-ov")
            assert handle._ctx.anchors == before
    run(go())


def test_the_empty_alias_goes_back_to_the_workflows_own_model(tmp_path):
    async def go():
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.resources.config.write_model(
                model_alias="Qwen3-8B-int4-ov")
            await session.resources.config.write_model(model_alias="")
            assert fake.picks[-1] == (None, None)
            assert not _config(session).active_model_pinned
            assert _config(session).active_model.alias == "Qwen3-8B-int4-ov"
    run(go())


def test_picking_the_model_already_running_builds_nothing(tmp_path):
    async def go():
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.resources.config.write_model(
                model_alias="Qwen3.5-4B-int4-ov", reasoning_effort="off")
            assert fake.picks == [(None, None)]
    run(go())


def test_a_model_the_session_does_not_offer_is_refused_in_words():
    async def go():
        async with served() as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.config.write_model(
                    model_alias="nope")
            assert err.value.error.code is ProtocolErrorCode.INVALID_PARAMS
            assert "Unknown model: nope" in err.value.error.message
            assert "Qwen3-8B-int4-ov" in err.value.error.message
            assert fake.picks == []
    run(go())


def test_thinking_is_the_enable_thinking_switch(tmp_path):
    async def go():
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.resources.config.set_thinking("high")
            assert fake.picks == [(None, None)]
            assert _config(session).active_model.thinking == "high"
            await session.resources.config.set_thinking("off")
            assert fake.picks[-1] == (None, None)
            assert _config(session).active_model.thinking == "off"
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.config.set_thinking("max")
            assert err.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    run(go())


def test_a_switch_is_refused_while_a_turn_runs():
    async def go():
        async with served(hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("slow")
            await events_until(session, lambda e: isinstance(e, TurnStarted))
            await poll(lambda: fake.busy)
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.config.write_model(
                    model_alias="Qwen3-8B-int4-ov")
            assert err.value.error.code is ProtocolErrorCode.CONFLICT
            assert fake.picks == []
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


def test_a_model_that_cannot_start_leaves_the_old_one_and_says_so(tmp_path):
    async def go():
        async with served(_wf(tmp_path), fail_pick=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.config.write_model(
                    model_alias="Qwen3-8B-int4-ov")
            assert "agent from before is still running" in err.value.error.message
            assert _config(session).active_model.alias == "Qwen3.5-4B-int4-ov"
            assert handle.ready
            # the old agent was never touched: still the same conversation
            assert _remembered_questions(fake) == ["one"]
            assert len(handle._ctx.anchors) == 1
    run(go())


def test_a_framework_engine_is_told_it_forgets_on_a_model_switch(tmp_path):
    async def go():
        async with served(_wf(tmp_path), native=False) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("one")
            await events_until(session, is_completed)
            await session.resources.config.write_model(
                model_alias="Qwen3-8B-int4-ov")
            seen = await events_until(
                session, lambda e: isinstance(e, ServerWarning))
            assert "keeps its own memory" in next(
                e.params.warning.message for e in seen
                if isinstance(e, ServerWarning))
    run(go())


def test_reload_rebuilds_only_when_the_workflow_file_changed(tmp_path):
    workflow = tmp_path / "w.yml"
    workflow.write_text("model: {name: Qwen3.5-4B-int4-ov}\n")

    async def go():
        async with served(str(workflow)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            await session.resources.config.reload()
            assert fake.picks == []                  # nothing to read again
            workflow.write_text("model: {name: other}\n")
            os.utime(workflow, ns=(1, time.time_ns() + 10**9))
            await session.resources.config.reload()
            assert len(fake.picks) == 1
            assert _remembered_questions(fake) == ["one"]
    run(go())


# ---- ask_user: the question callback --------------------------------------------

QUESTION = {"question": "Which database?", "choices": ["sqlite", "postgres"]}


def _answer_with(text=None, cancelled=False, other=False):
    from ovat.tui.vibe.app_server.models import (UserAnswer,
                                                 UserInputCallbackOutput,
                                                 UserQuestionResult)
    answers = [] if text is None else [UserAnswer(
        question="Which database?", answer=text, is_other=other)]
    return UserInputCallbackOutput(result=UserQuestionResult(
        answers=answers, cancelled=cancelled))


def test_a_question_round_trips_to_the_asking_tool():
    from ovat.tui.vibe.app_server.models import UserInputCallbackDetail

    async def go():
        async with served(question=QUESTION) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            events = []

            async def drive():
                async with aclosing(session.act("pick one")) as gen:
                    async for event in gen:
                        events.append(event)
                        if isinstance(event, CallbackRequested):
                            await session.respond_to_callback(
                                event.callback.callback_id,
                                _answer_with("sqlite"))
            await asyncio.wait_for(drive(), T)

            asked = next(e.callback for e in events
                         if isinstance(e, CallbackRequested))
            assert isinstance(asked.detail, UserInputCallbackDetail)
            question = asked.detail.request.questions[0]
            assert question.question == "Which database?"
            assert [o.label for o in question.options] == ["sqlite",
                                                           "postgres"]
            assert fake.answered == "sqlite"
            assert any(isinstance(e, HistoryEntryUpdated)
                       and isinstance(e.entry, PublicCallbackEntry)
                       and e.entry.state.status == "answered"
                       for e in events)
            card = next(e for e in session.history if e.id == "q1")
            assert isinstance(card.state, CompletedEffectState)
            assert card.state.display.message == "sqlite"
            assert not session.state.active_callbacks
    run(go())


def test_a_question_without_choices_still_shows_two_options():
    async def go():
        async with served(question={"question": "Name it?"}) as (
                session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("go")
            seen = await events_until(
                session, lambda e: isinstance(e, CallbackRequested))
            options = seen[-1].callback.detail.request.questions[0].options
            assert [o.label for o in options] == ["Not sure", "Skip"]
            await session.respond_to_callback(
                seen[-1].callback.callback_id,
                _answer_with("Widget", other=True))
            await events_until(session, is_completed)
            assert fake.answered == "Widget"
    run(go())


def test_the_person_cancelling_the_question_gives_the_tool_no_answer():
    async def go():
        async with served(question=QUESTION) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("pick one")
            seen = await events_until(
                session, lambda e: isinstance(e, CallbackRequested))
            await session.respond_to_callback(
                seen[-1].callback.callback_id, _answer_with(cancelled=True))
            await events_until(session, is_completed)
            assert fake.answered is None
    run(go())


def test_interrupting_closes_an_open_question():
    async def go():
        async with served(question=QUESTION) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("pick one")
            seen = await events_until(
                session, lambda e: isinstance(e, CallbackRequested))
            callback = seen[-1].callback
            await session.interrupt()
            more = await events_until(session, is_completed)
            assert completed_turn(more).status is PublicTurnStatus.INTERRUPTED
            assert fake.answered is None
            closed = [e.entry for e in more
                      if isinstance(e, HistoryEntryUpdated)
                      and isinstance(e.entry, PublicCallbackEntry)]
            assert closed and closed[0].state.status == "cancelled"
            with pytest.raises(AppServerResponseError) as err:
                await session.respond_to_callback(
                    callback.callback_id, _answer_with("sqlite"))
            assert err.value.error.code is ProtocolErrorCode.CALLBACK_CLOSED
    run(go())


def test_a_client_that_cannot_show_questions_gets_no_answer_at_once():
    async def go():
        made = []

        def factory(workflow, sink, cwd=None):
            made.append(FakeSession(workflow, sink, cwd, question=QUESTION))
            return made[0]
        session, handle = await start_ovat_app_server(
            "w.yml", session_factory=factory,
            capabilities=ClientCapabilities(callback_kinds=["approval"]))
        try:
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("pick one")
            seen = await events_until(session, is_completed)
            assert made[0].answered is None
            assert any(isinstance(e, ServerWarning) and "cannot show" in
                       e.params.warning.message for e in seen)
        finally:
            await asyncio.wait_for(session.close(), T)
            await asyncio.wait_for(handle.close(), T)
    run(go())


# ---- the todo and web tools as Vibe cards ---------------------------------------

def test_todo_and_web_results_become_vibe_cards():
    from ovat.tui.vibe.app_server.models import (TodoEffectDetail,
                                                 TodoEffectOutput,
                                                 WebFetchEffectDetail,
                                                 WebFetchEffectOutput,
                                                 WebSearchEffectDetail,
                                                 WebSearchEffectOutput)
    todos = [{"id": "1", "content": "read it", "status": "in_progress",
              "priority": "high"}]
    tools = [
        {"name": "todo", "arguments": {"action": "write", "todos": todos},
         "display": {"kind": "todo", "action": "write", "todos": todos}},
        {"name": "web_fetch", "arguments": {"url": "https://example.com"},
         "display": {"kind": "web_fetch", "url": "https://example.com/",
                     "content": "# Hi", "content_type": "text/html",
                     "was_truncated": False}},
        {"name": "web_search", "arguments": {"query": "ovms"},
         "display": {"kind": "web_search", "query": "ovms", "answer": "a",
                     "sources": [{"title": "T", "url": "https://x.org"}]}},
    ]

    async def go():
        async with served(extra_tools=tools) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "plan")
            todo, fetch, search = (next(e for e in session.history
                                        if e.id == f"x{n}")
                                   for n in (1, 2, 3))
            assert isinstance(todo.detail, TodoEffectDetail)
            assert todo.detail.input.action == "write"
            assert TodoEffectOutput.model_validate(
                todo.state.output).todos[0].content == "read it"
            assert todo.state.display.message == "1 todo"
            assert isinstance(fetch.detail, WebFetchEffectDetail)
            assert WebFetchEffectOutput.model_validate(
                fetch.state.output).url == "https://example.com/"
            assert isinstance(search.detail, WebSearchEffectDetail)
            out = WebSearchEffectOutput.model_validate(search.state.output)
            assert out.sources[0].url == "https://x.org"
            assert search.state.display.message == "ovms (1 result)"
    run(go())


# ---- /config and /theme (config/write, config/fields/read) -----------------------

WORKFLOW = """\
# my agent
model:
  name: Qwen3.5-4B-int4-ov   # the model
  device: GPU
tools:
  - name: search_docs
    type: builtin
agent:
  type: native
"""


def _wf(tmp_path, text=WORKFLOW):
    path = tmp_path / "w.yml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _op(path, value=None, op="set", target=None):
    from ovat.tui.vibe.app_server.protocol import ConfigWriteOpWire
    return ConfigWriteOpWire(op=op, path=path, value=value,
                             target_layer=target)


async def _write(session, *ops, reload=False):
    return await session.resources.config.write(
        list(ops), reason="test", reload_runtime=reload)


def test_the_settings_screen_lists_the_workflows_own_settings(tmp_path):
    async def go():
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            response = await session.resources.config.read_fields()
            by_name = {f.name: f for f in response.fields}
            device = by_name["device"]
            assert device.kind.value == "enum" and device.value == "GPU"
            assert device.enum_choices == ["CPU", "GPU", "NPU"]
            assert [v.layer for v in device.layer_values] == [
                "workflow", "default"]
            temperature = by_name["temperature"]
            assert [v.layer for v in temperature.layer_values] == ["default"]
            assert by_name["search_docs.permission"].path == \
                "/tools/search_docs/permission"
            assert by_name["engine"].enum_choices[0] == "native"
            assert by_name["theme"].value == "auto"
            # no Vibe-only setting is listed
            assert not {"voice_mode_enabled", "narrator_enabled", "log_level",
                        "enable_telemetry"} & set(by_name)
            assert response.targets == ["workflow"]
    run(go())


def test_a_model_picker_choice_is_saved_in_the_workflow_layer(tmp_path):
    async def go():
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.resources.config.write_model(
                model_alias="Qwen3-8B-int4-ov")
            fields = {f.name: f for f in
                      (await session.resources.config.read_fields()).fields}
            model = fields["active_model"]
            assert model.value == "Qwen3-8B-int4-ov"
            assert [(v.layer, v.value) for v in model.layer_values] == [
                ("workflow", "Qwen3-8B-int4-ov")]
            assert "name: Qwen3-8B-int4-ov" in Path(fake.workflow).read_text()
    run(go())


def test_a_setting_is_written_into_the_file_and_the_comments_stay(tmp_path):
    path = _wf(tmp_path)

    async def go():
        async with served(path) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            response = await _write(session,
                                    _op("/model/temperature", 0.3),
                                    _op("/tools/search_docs/permission",
                                        "ask"))
            assert response.applied
            text = open(path, encoding="utf-8").read()
            assert "# my agent" in text and "# the model" in text
            from ovat.config.workflow import load_workflow
            config = load_workflow(path)
            assert config.model.temperature == 0.3
            assert config.tools[0].permission == "ask"
            # written, not yet rebuilt: the screen sends config/reload on close
            assert fake.picks == []
            await session.resources.config.reload()
            assert fake.picks == [(None, None)]
    run(go())


def test_reload_runtime_builds_again_and_the_conversation_survives(tmp_path):
    path = _wf(tmp_path)

    async def go():
        async with served(path) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            response = await _write(session, _op("/model/device", "NPU"),
                                    reload=True)
            assert response.applied
            assert fake.picks == [(None, None)]
            assert _remembered_questions(fake) == ["one"]
            assert _texts(session.history) == ["one"]
            await _ask(session, "two")
            assert _remembered_questions(fake) == ["one", "two"]
    run(go())


def test_an_invalid_value_is_refused_with_the_message_and_changes_nothing(
        tmp_path):
    path = _wf(tmp_path)
    before = open(path, "rb").read()

    async def go():
        async with served(path) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            response = await _write(session, _op("/model/top_p", 5.0))
            assert response.rejected and not response.applied
            assert "model.top_p" in response.failures[0]
            assert open(path, "rb").read() == before
            seen = await events_until(
                session, lambda e: isinstance(e, ServerWarning))
            assert "model.top_p" in next(
                e.params.warning.message for e in seen
                if isinstance(e, ServerWarning))
            assert fake.picks == []
    run(go())


def test_a_batch_with_one_bad_op_writes_none_of_it(tmp_path):
    path = _wf(tmp_path)
    before = open(path, "rb").read()

    async def go():
        async with served(path) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            response = await _write(session,
                                    _op("/model/temperature", 0.3),
                                    _op("/model/device", "TPU"))
            assert response.failures == [
                "device must be one of: CPU, GPU, NPU."]
            assert open(path, "rb").read() == before
    run(go())


def test_a_vibe_only_setting_is_answered_not_used_and_changes_nothing(
        tmp_path):
    path = _wf(tmp_path)
    before = open(path, "rb").read()

    async def go():
        async with served(path) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            for name in ("voice_mode_enabled", "log_level",
                         "enable_telemetry"):
                response = await _write(session, _op(f"/{name}", True))
                assert not response.applied and not response.rejected
                assert response.failures == [
                    f"'{name}' is not used by OVAT, so nothing was changed."]
            response = await _write(session, _op("/no_such_thing", 1))
            assert response.failures == ["Unknown setting: /no_such_thing"]
            response = await _write(session, _op("/model/device", "CPU",
                                                 target="project-toml"))
            assert "Unknown layer" in response.failures[0]
            assert open(path, "rb").read() == before
            with pytest.raises(AppServerResponseError) as err:
                await session.resources.config.update({"log_level": "debug"})
            assert "not used by OVAT" in str(err.value)
    run(go())


def test_a_setting_is_reset_by_removing_it(tmp_path):
    path = _wf(tmp_path)

    async def go():
        async with served(path) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _write(session, _op("/model/temperature", 0.3))
            await _write(session, _op("/model/temperature", op="remove",
                                      target="workflow"))
            assert "temperature" not in open(path, encoding="utf-8").read()
    run(go())


def test_saving_the_model_in_the_file_drops_the_sessions_own_pick(tmp_path):
    path = _wf(tmp_path)

    async def go():
        async with served(path) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            await session.resources.config.write_model(
                model_alias="Qwen3-8B-int4-ov")
            fake.picks.clear()
            response = await _write(session,
                                    _op("/active_model", "Qwen3-8B-int4-ov"),
                                    reload=True)
            assert response.applied
            assert "name: Qwen3-8B-int4-ov" in open(
                path, encoding="utf-8").read()
            assert fake.forgotten == ["name", "name"]
            assert fake.picks == [(None, None)]          # built from the file
            assert _remembered_questions(fake) == ["one"]
    run(go())


def test_the_default_model_choice_only_drops_the_pick_and_leaves_the_file(
        tmp_path):
    path = _wf(tmp_path)
    before = open(path, "rb").read()

    async def go():
        async with served(path) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            response = await _write(session, _op("/active_model", ""))
            assert response.applied
            assert open(path, "rb").read() == before
            assert fake.forgotten == ["name"]
            # the file did not change, but the agent is out of date anyway
            await session.resources.config.reload()
            assert fake.picks == [(None, None)]
    run(go())


def test_writing_settings_is_refused_while_a_turn_runs(tmp_path):
    path = _wf(tmp_path)
    before = open(path, "rb").read()

    async def go():
        async with served(path, hold_turn=True) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await session.enqueue("slow")
            await events_until(session, lambda e: isinstance(e, TurnStarted))
            await poll(lambda: fake.busy)
            with pytest.raises(AppServerResponseError) as err:
                await _write(session, _op("/model/device", "CPU"))
            assert err.value.error.code is ProtocolErrorCode.CONFLICT
            assert open(path, "rb").read() == before
            fake.turn_gate.set()
            await events_until(session, is_completed)
    run(go())


def test_a_workflow_that_will_not_load_is_a_sentence(tmp_path):
    path = _wf(tmp_path, "model: {name: m, devise: GPU}\n")

    async def go():
        async with served(path) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            response = await _write(session, _op("/model/device", "CPU"))
            assert not response.applied
            assert "Cannot read" in response.failures[0]
    run(go())


def test_the_theme_is_saved_in_the_ui_preferences_file(tmp_path):
    from ovat.agentic import prefs

    async def go():
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            cwd = handle._ctx.cwd
            prefs.update(cwd, config="keep.yml")
            response = await _write(session, _op("/theme", "nord"))
            assert response.applied
            assert _config(session).theme == "nord"
            assert prefs.load(cwd) == {"config": "keep.yml", "theme": "nord"}
            fields = {f.name: f for f in
                      (await session.resources.config.read_fields()).fields}
            assert [v.layer for v in fields["theme"].layer_values] == [
                "preferences", "default"]
            # a theme that does not exist is refused, the saved one stays
            response = await _write(session, _op("/theme", "neon-pink"))
            assert response.failures == ["Unknown theme: neon-pink"]
            assert prefs.load(cwd)["theme"] == "nord"
            # and removing it goes back to auto, keeping the other keys
            await _write(session, _op("/theme", op="remove"))
            assert prefs.load(cwd) == {"config": "keep.yml"}
            assert _config(session).theme == "auto"
            assert fake.picks == []            # a theme never rebuilds
    run(go())


def test_a_saved_theme_is_what_the_next_session_starts_with(tmp_path):
    from ovat.agentic import prefs

    async def go():
        prefs.update(os.getcwd(), theme="nord")
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            assert _config(session).theme == "nord"
    run(go())


def test_diagnostics_are_the_doctors_checks_that_are_not_ok(monkeypatch,
                                                            tmp_path):
    from ovat.cli import diagnostics
    from ovat.tui.vibe.app_server.protocol import DiagnosticsListResponse
    seen = []

    def checks(config_path=None):
        seen.append(config_path)
        return [diagnostics.Check("Python", "ok", "3.13"),
                diagnostics.Check("OVMS serving", "warn", "ovms not found"),
                diagnostics.Check("Workflow config", "fail", "no model")]
    monkeypatch.setattr(diagnostics, "run_checks", checks)
    path = _wf(tmp_path)

    async def go():
        async with served(path) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            client = session._connection.current
            raw = await client.request(
                "diagnostics/list",
                {"sessionId": handle._ctx.projector.session.id})
            response = DiagnosticsListResponse.model_validate(
                raw, by_alias=True)
            # worst first, ok ones left out, and the same checks /doctor runs
            assert [(i.file, i.message) for i in response.issues] == [
                ("Workflow config", "fail: no model"),
                ("OVMS serving", "warn: ovms not found")]
            assert response.hooks_count == 0
            assert seen == [path]
    run(go())


def test_saving_the_engine_uses_the_new_workflow_engine_and_keeps_memory(tmp_path):
    async def go():
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            await _ask(session, "one")
            result = await _write(session, _op("/agent/type", "react"), reload=True)
            assert result.applied
            assert handle._ctx.info["engine"] == "react"
            assert _remembered_questions(fake) == ["one"]
    run(go())


def test_subagent_done_updates_session_token_usage_when_initially_none(tmp_path):
    from ovat.agent.events import Event, SUBAGENT_STARTED, SUBAGENT_DONE

    async def go():
        async with served(_wf(tmp_path)) as (session, handle, fake):
            await session.resources.runtime.wait_until_ready()
            proj = handle._ctx.projector
            assert proj.session.token_usage is None
            proj.agent_event(Event(SUBAGENT_STARTED, {
                "call_id": "c123", "agent": "researcher", "kind": "workflow"
            }), {})
            proj.agent_event(Event(SUBAGENT_DONE, {
                "call_id": "c123", "status": "ok",
                "totals": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
            }), {})
            assert proj.session.token_usage is not None
            assert proj.session.token_usage.input_tokens == 50
            assert proj.session.token_usage.output_tokens == 20
            assert proj.session.token_usage.total_tokens == 70
            child = proj.child_sessions["child-c123"]
            assert child.context_usage is not None
            assert child.context_usage.total_tokens == 70
    run(go())
