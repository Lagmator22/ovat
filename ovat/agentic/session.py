# ovat/agentic/session.py
"""AgentSession: one interactive agent, driven by any front end.

Note to myself: everything slow happens on a worker thread (building the
agent loads models; a turn waits on OVMS; a `!` command can run for
minutes), and everything the user should SEE is reported through one
callback, `sink(event)`, with the small dataclasses below. The TUI's sink
posts them to its own thread; a test's sink appends them to a list. The
session never draws anything.

In C++ terms: a controller object that owns the model and a worker thread,
and publishes immutable messages to one observer.
"""
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ovat.agent.events import Event
from ovat.agent.permissions import ApprovalBroker, ApprovalRequest


# ---- what the session tells its front end ---------------------------------

@dataclass(frozen=True)
class Ready:
    """The agent is built. `info` is a plain dict for the banner and bar."""
    info: dict


@dataclass(frozen=True)
class BuildFailed:
    text: str


@dataclass(frozen=True)
class TurnStarted:
    text: str


@dataclass(frozen=True)
class AgentActivity:
    """A tool, remote-agent or subagent event from the running agent."""
    event: Event


@dataclass(frozen=True)
class ApprovalNeeded:
    request: ApprovalRequest


@dataclass(frozen=True)
class TurnDone:
    answer: str
    elapsed: float
    totals: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    failed: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class ShellLine:
    text: str


@dataclass(frozen=True)
class ShellDone:
    command: str
    code: int | None
    elapsed: float


@dataclass(frozen=True)
class Notice:
    level: str            # info | warn | error
    text: str


Sink = Callable[[object], None]


def find_workflow(cwd: str | None = None) -> str | None:
    """./workflow.yml (or .yaml) if there is one."""
    base = Path(cwd or os.getcwd())
    for name in ("workflow.yml", "workflow.yaml"):
        if (base / name).is_file():
            return str(base / name)
    return None


def _brief(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    first = text[0] if text else ""
    return f"{type(exc).__name__}: {first}"[:400] if first else type(exc).__name__


def describe_agent(config, agent, workflow: str) -> dict:
    """What the banner and the bottom bar show. Plain values only."""
    tools = getattr(agent, "tools", None) or {}
    return {
        "workflow": workflow,
        "name": Path(workflow).stem if workflow else "",
        # A local model is named by its folder; show the folder's name.
        "model": Path(config.model.name.replace("\\", "/")).name
        or config.model.name,
        "device": config.model.device,
        "provider": config.model.provider,
        "engine": config.agent.type,
        "tools": [{"name": n, "kind": spec.get("kind", "builtin"),
                   "permission": spec.get("permission", "allow")}
                  for n, spec in tools.items()],
        "subagents": [{"name": s.name, "description": s.description,
                       "where": ("workflow" if s.workflow else
                                 "api" if s.api else "a2a")}
                      for s in config.subagents],
    }


class AgentSession:
    def __init__(self, workflow: str | None, sink: Sink, *,
                 load=None, build=None, close=None,
                 cwd: str | None = None):
        from ovat.agent.factory import build_agent, close_agent
        from ovat.config.workflow import load_workflow
        self.workflow = workflow
        self.sink = sink
        self.cwd = cwd or os.getcwd()
        self._load = load or load_workflow
        self._build = build or build_agent
        self._close = close or close_agent
        self.config = None
        self.agent = None
        self.info: dict = {}
        self.broker = ApprovalBroker(
            on_request=lambda r: self._emit(ApprovalNeeded(r)))
        self.busy = False
        self.last_answer = ""
        self.last_trace: dict = {}
        self.totals = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()
        self._shell: subprocess.Popen | None = None
        self._server: subprocess.Popen | None = None
        self._unsubscribe = None

    # ---- lifecycle ---------------------------------------------------------

    def start(self, engine: str | None = None) -> threading.Thread:
        """Load and build on a worker; ends in Ready or BuildFailed."""
        return self._spawn(self._build_now, engine)

    def _build_now(self, engine: str | None) -> None:
        if not self.workflow:
            self._emit(BuildFailed(
                "No workflow here. Run `/workflow path/to/workflow.yml`, or "
                "start with `ovat agent path/to/workflow.yml`. `ovat init` "
                "writes a starter one."))
            return
        try:
            config = self._load(self.workflow)
            if engine:
                config = config.model_copy(update={
                    "agent": config.agent.model_copy(update={"type": engine})})
            agent = self._build(config)
        except Exception as exc:
            self._emit(BuildFailed(f"Could not build {self.workflow}: "
                                   f"{_brief(exc)}"))
            return
        self._drop_agent()
        self.config, self.agent = config, agent
        agent.events.approver = self.broker
        self._unsubscribe = agent.events.subscribe(
            lambda e: self._emit(AgentActivity(e)))
        self.info = describe_agent(config, agent, self.workflow)
        self._emit(Ready(self.info))

    def switch_workflow(self, path: str) -> threading.Thread:
        self.workflow = path
        return self.start()

    def new_conversation(self) -> threading.Thread | None:
        """Forget the conversation. The native loop just gets an empty
        Session (no model reload); the framework engines are rebuilt, the
        one way to forget that works on all of them."""
        self.broker.forget_session_grants()
        self.last_trace, self.last_answer = {}, ""
        from ovat.agent.loop import AgentLoop
        from ovat.agent.session import Session
        if isinstance(self.agent, AgentLoop):
            self.agent.session = Session(
                system_prompt=self.config.agent.system_prompt)
            self._emit(Notice("info", "New conversation."))
            return None
        engine = self.config.agent.type if self.config else None
        return self.start(engine)

    def close(self) -> None:
        self.cancel()
        self.stop_server()
        self._drop_agent()

    def _drop_agent(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        if self.agent is not None:
            try:
                self._close(self.agent)
            except Exception:
                pass
            self.agent = None

    # ---- turns -------------------------------------------------------------

    def ask(self, text: str) -> threading.Thread | None:
        """Run one turn on a worker. None if one is already running."""
        with self._lock:
            if self.busy:
                self._emit(Notice("warn", "Still working; Esc stops it."))
                return None
            if self.agent is None:
                self._emit(Notice("warn", "No agent is ready yet."))
                return None
            self.busy = True
        return self._spawn(self._turn, text)

    def ask_subagent(self, name: str, task: str) -> threading.Thread | None:
        """`/ask name task`: straight to one subagent, no main model."""
        tools = getattr(self.agent, "tools", None) or {}
        if "delegate" not in tools:
            self._emit(Notice("warn", "This workflow has no subagents."))
            return None
        with self._lock:
            if self.busy:
                self._emit(Notice("warn", "Still working; Esc stops it."))
                return None
            self.busy = True

        def run():
            answer = tools["delegate"]["function"](agent=name, task=task)
            # The "[Answer from subagent ...]" line frames the reply for a
            # MODEL; a person who asked directly just wants the reply.
            if answer.startswith("[Answer from subagent") and "\n" in answer:
                return answer.split("\n", 1)[1]
            return answer
        return self._spawn(self._turn, f"/ask {name} {task}", run)

    def _turn(self, text: str, run=None) -> None:
        agent = self.agent
        agent.events.stop.clear()
        self._emit(TurnStarted(text))
        started = time.monotonic()
        try:
            answer = run() if run else agent.run(text)
        except Exception as exc:
            answer = f"Error: {_brief(exc)}"
        elapsed = time.monotonic() - started
        trace = {} if run else (getattr(agent, "last_trace", None) or {})
        totals = dict(trace.get("totals") or {})
        cancelled = agent.events.stop.is_set()
        failed = bool(totals.get("failed")
                      or getattr(agent, "last_failed", False)
                      or (answer or "").startswith("Error: "))
        if not run:
            self.last_trace = trace
            self.totals["turns"] += 1
            for key in ("prompt_tokens", "completion_tokens"):
                if totals.get(key) is not None:
                    self.totals[key] += totals[key]
        self.last_answer = answer or ""
        with self._lock:
            self.busy = False
        self._emit(TurnDone(answer or "", elapsed, totals,
                            self._sources(trace), failed, cancelled))

    @staticmethod
    def _sources(trace: dict) -> list:
        found = []
        for turn in trace.get("turns", []) or []:
            for call in turn.get("tool_calls", []) or []:
                for src in call.get("sources") or []:
                    if src not in found:
                        found.append(src)
        return found

    def cancel(self) -> bool:
        """Esc: stop the turn, deny open approvals, end a `!` command."""
        stopped = False
        if self.agent is not None and self.busy:
            self.agent.events.stop.set()
            stopped = True
        self.broker.cancel_all()
        if self._shell is not None and self._shell.poll() is None:
            from ovat.cli.shell import kill
            try:
                kill(self._shell)
            except Exception:
                pass
            stopped = True
        return stopped

    # ---- `!` shell ---------------------------------------------------------

    def run_shell(self, command: str) -> threading.Thread | None:
        if self._shell is not None and self._shell.poll() is None:
            self._emit(Notice("warn", "A command is still running; Esc "
                                      "stops it."))
            return None
        return self._spawn(self._shell_now, command)

    def _shell_now(self, command: str) -> None:
        from ovat.cli.shell import iter_display_lines, spawn, venv_env
        started = time.monotonic()
        try:
            self._shell = spawn(command, self.cwd, venv_env(columns=100))
        except Exception as exc:
            self._emit(ShellLine(f"could not start: {_brief(exc)}"))
            self._emit(ShellDone(command, None, 0.0))
            return
        proc = self._shell
        try:
            for line in iter_display_lines(proc.stdout):
                self._emit(ShellLine(line))
        finally:
            if proc.stdout:
                proc.stdout.close()
        code = proc.wait()
        self._emit(ShellDone(command, code, time.monotonic() - started))

    # ---- the local A2A server (`/serve`) -----------------------------------

    @property
    def serving(self) -> bool:
        return self._server is not None and self._server.poll() is None

    def start_server(self) -> None:
        """`ovat a2a serve` for this workflow, as its OWN process: it keeps
        its own agent and memory, and survives nothing the TUI does."""
        if not self.workflow:
            self._emit(Notice("warn", "Load a workflow first."))
            return
        if self.serving:
            self._emit(Notice("info", "The A2A server is already running."))
            return
        cmd = [sys.executable, "-m", "ovat.cli.main", "a2a", "serve",
               self.workflow]
        from ovat.cli.shell import iter_display_lines, spawn, venv_env
        line = " ".join(shlex.quote(c) for c in cmd) if os.name != "nt" \
            else subprocess.list2cmdline(cmd)
        self._server = spawn(line, self.cwd, venv_env(columns=100))
        proc = self._server

        def pump():
            for text in iter_display_lines(proc.stdout):
                self._emit(Notice("info", f"a2a: {text}"))
            code = proc.wait()
            self._emit(Notice("warn" if code else "info",
                              f"a2a server stopped (exit {code})"))
        self._spawn(pump)

    def stop_server(self) -> None:
        if self.serving:
            from ovat.cli.shell import terminate
            try:
                terminate(self._server)
            except Exception:
                pass

    # ---- plumbing ----------------------------------------------------------

    def _emit(self, event) -> None:
        try:
            self.sink(event)
        except Exception:
            pass

    def _spawn(self, target, *args) -> threading.Thread:
        thread = threading.Thread(target=target, args=args, daemon=True,
                                  name="ovat-agent-session")
        thread.start()
        return thread
