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

from ovat.agent.events import (ANSWER_DELTA, REASONING_DELTA, STEER_TAKEN,
                               Event)
from ovat.agent.permissions import ApprovalBroker, ApprovalRequest
from ovat.agent.questions import QuestionBroker, QuestionRequest
from ovat.agentic import rules
from ovat.text import split_thinking


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
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class AnswerDelta:
    text: str


@dataclass(frozen=True)
class Steered:
    """The native loop took a steering message into the model's memory, at
    message number `index`."""
    text: str
    index: int


@dataclass(frozen=True)
class ApprovalNeeded:
    request: ApprovalRequest


@dataclass(frozen=True)
class QuestionNeeded:
    """The ask_user tool wants a person's answer (agent/questions.py)."""
    request: QuestionRequest


@dataclass(frozen=True)
class TurnDone:
    answer: str
    elapsed: float
    totals: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    failed: bool = False
    cancelled: bool = False
    # What a thinking model reasoned before it answered, kept apart so
    # `answer` is only the answer. "" when it did not narrate.
    reasoning: str = ""


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


def _context_window(config) -> int:
    from ovat.core.model_scout import read_context_window
    m = config.model
    if m.context_window:
        return int(m.context_window)
    if (m.device or "").upper() == "NPU":
        return int(m.ovms_max_prompt_len or 4096)
    roots = ([m.model_repository_path] if m.model_repository_path else []) + (
        list(getattr(config, "model_search_paths", None) or []))
    # `ovms --pull` lays a model out by its source (models/OpenVINO/<name>),
    # so that path is tried before the bare name.
    for name in filter(None, (m.source_model, m.name)):
        window = read_context_window(name, roots)
        if window:
            return window
    return 0


def _model_choices(config) -> list[dict]:
    """The /model picker's list, as plain values. Never raises: a list that
    cannot be had is only a picker with one choice."""
    try:
        from ovat.agentic import models
        return [{"id": c.id, "label": c.label} for c in models.choices(config)]
    except Exception:
        name = config.model.name
        return [{"id": name, "label": name}]


def describe_agent(config, agent, workflow: str,
                   default_model: str | None = None) -> dict:
    """What the banner and the bottom bar show. Plain values only.

    `default_model` is the model the workflow FILE names, when this session
    runs another one (see AgentSession.switch_model)."""
    tools = getattr(agent, "tools", None) or {}
    return {
        # For /model and /thinking: the model by the name it is set with (the
        # display name below drops a local folder's path), the list to pick
        # from, the workflow's own model, and the thinking switch (None: the
        # chat template decides).
        "model_id": config.model.name,
        "models": _model_choices(config),
        "default_model_id": default_model or config.model.name,
        "thinking": config.model.enable_thinking,
        "workflow": workflow,
        "name": Path(workflow).stem if workflow else "",
        # A local model is named by its folder; show the folder's name.
        "model": Path(config.model.name.replace("\\", "/")).name
        or config.model.name,
        "device": config.model.device,
        "provider": config.model.provider,
        "engine": config.agent.type,
        "context_window": _context_window(config),
        "tools": [{"name": n, "kind": spec.get("kind", "builtin"),
                   "permission": spec.get("permission", "allow")}
                  for n, spec in tools.items()],
        "subagents": [{"name": s.name, "description": s.description,
                       "where": ("workflow" if s.workflow else
                                 "api" if s.api else "a2a")}
                      for s in config.subagents],
    }


# How ovat/agent/subagents.py and ovat/tools/a2a_call.py open a reply.
_FRAMES = ("[Answer from subagent", "[Reply from ")


def _chat_message(m) -> bool:
    """A user, assistant or tool message as the native loop writes them."""
    if not isinstance(m, dict) or m.get("role") not in (
            "user", "assistant", "tool"):
        return False
    if not isinstance(m.get("content"), (str, type(None))):
        return False
    if m["role"] == "tool" and not isinstance(m.get("tool_call_id"), str):
        return False
    return isinstance(m.get("tool_calls", []), list)


class AgentSession:
    def __init__(self, workflow: str | None, sink: Sink, *,
                 load=None, build=None, close=None,
                 cwd: str | None = None):
        from ovat.agent.factory import build_agent, close_agent
        from ovat.agentic.mcp import MCPSelection
        from ovat.config.workflow import load_workflow
        self.workflow = workflow
        self.sink = sink
        self.cwd = cwd or os.getcwd()
        self._load = load or load_workflow
        self.mcp_selection = MCPSelection()
        self._build = build or (lambda config: build_agent(
            config, select_tools=self.mcp_selection.select_tools))
        self._close = close or close_agent
        self.config = None
        self.agent = None
        self.info: dict = {}
        # The project's "always allow" rules are preloaded here and nowhere
        # else: this session has a person to ask, `ovat run` does not.
        self.broker = ApprovalBroker(
            on_request=lambda r: self._emit(ApprovalNeeded(r)),
            always=rules.load(self.cwd), on_always=self._remember_always)
        self.questions = QuestionBroker(
            on_request=lambda r: self._emit(QuestionNeeded(r)))
        self.busy = False
        self.last_answer = ""
        self.last_trace: dict = {}
        self.totals = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()
        self._shell: subprocess.Popen | None = None
        self._server: subprocess.Popen | None = None
        self._unsubscribe = None
        # What this session runs instead of what the workflow file says
        # (`model.name`, `model.enable_thinking`). Kept here, not in the
        # config, because every rebuild (/engine, /clear on a framework
        # engine) loads the file again and must not forget the pick.
        self._overrides: dict = {}

    # ---- lifecycle ---------------------------------------------------------

    def start(self, engine: str | None = None) -> threading.Thread:
        """Load and build on a worker; ends in Ready or BuildFailed."""
        return self._spawn(self._build_now, engine, {})

    def switch_model(self, model: str | None = None,
                     enable_thinking: bool | None = None,
                     follow_workflow: bool = False) -> threading.Thread:
        """/model and /thinking: build again with another model and/or the
        thinking switch, for this session only (the workflow file is not
        touched). `model` "" goes back to the workflow's own; None leaves it.
        The engine stays the one running now. The pick counts only once the
        build succeeds, so a failed one leaves the old agent and old pick."""
        picked = {}
        if model is not None:
            picked["name"] = model
        if enable_thinking is not None:
            picked["enable_thinking"] = enable_thinking
        # A settings reload must also honour an edited agent.type. Ordinary
        # model picks retain a session's current engine.
        engine = (self.config.agent.type if self.config else None) \
            if not follow_workflow else None
        return self._spawn(self._build_now, engine, picked)

    def _remember_always(self, tool: str) -> None:
        """A person chose "always allow" for `tool`: keep it for the next
        session in this project (ovat/agentic/rules.py)."""
        try:
            rules.add(self.cwd, tool)
        except OSError as exc:
            self._emit(Notice(
                "warn", f"Could not save \"always allow {tool}\" "
                        f"({exc.strerror or exc}); it lasts for this "
                        f"session only."))

    def forget_picks(self, *names: str) -> None:
        """`/config` saved a model setting in the workflow file, so the
        /model or /thinking pick for it (`name`, `enable_thinking`) must stop
        winning over the file. Takes effect at the next build."""
        for name in names:
            self._overrides.pop(name, None)

    def _build_now(self, engine: str | None, picked: dict,
                   fresh: bool = False) -> None:
        if not self.workflow:
            self._emit(BuildFailed(
                "No workflow here. Run `/workflow path/to/workflow.yml`, or "
                "start with `ovat agent path/to/workflow.yml`. `ovat init` "
                "writes a starter one."))
            return
        try:
            config = self._load(self.workflow)
            default_model = config.model.name
            overrides = {**({} if fresh else self._overrides), **picked}
            if overrides.get("name") == "":
                del overrides["name"]               # back to the file's own
            if overrides:
                config = config.model_copy(update={
                    "model": config.model.model_copy(update=overrides)})
            if engine:
                config = config.model_copy(update={
                    "agent": config.agent.model_copy(update={"type": engine})})
            agent = self._build(self.mcp_selection.config_for_build(config))
        except Exception as exc:
            self._emit(BuildFailed(f"Could not build {self.workflow}: "
                                   f"{_brief(exc)}"))
            return
        self._drop_agent()
        self.config, self.agent = config, agent
        self._overrides = overrides
        agent.events.approver = self.broker
        agent.events.asker = self.questions

        def _on_bus(e: Event) -> None:
            if e.type == ANSWER_DELTA:
                self._emit(AnswerDelta(e.data.get("text") or ""))
            elif e.type == REASONING_DELTA:
                self._emit(ReasoningDelta(e.data.get("text") or ""))
            elif e.type == STEER_TAKEN:
                self._emit(Steered(e.data.get("text") or "",
                                   int(e.data.get("index") or 0)))
            else:
                self._emit(AgentActivity(e))

        self._unsubscribe = agent.events.subscribe(_on_bus)
        self.info = describe_agent(config, agent, self.workflow, default_model)
        self.refresh_mcp()
        self._emit(Ready(self.info))

    def refresh_mcp(self) -> None:
        if self.config is not None:
            self.info["mcp"] = self.mcp_selection.catalog(self.config, self.agent)

    def switch_workflow(self, path: str) -> threading.Thread:
        """Another workflow starts from its own file: no picks carry over."""
        self.workflow = path
        return self._spawn(self._build_now, None, {}, True)

    def new_conversation(self) -> threading.Thread | None:
        """Forget the conversation. The native loop just gets an empty
        Session (no model reload); the framework engines are rebuilt, the
        one way to forget that works on all of them."""
        self.broker.forget_session_grants()
        self.last_trace, self.last_answer = {}, ""
        from ovat.agent.loop import AgentLoop
        from ovat.agent.factory import compose_system_prompt
        from ovat.agent.session import Session
        if isinstance(self.agent, AgentLoop):
            self.agent.session = Session(
                system_prompt=compose_system_prompt(self.config, Path(self.cwd)))
            self._emit(Notice("info", "New conversation."))
            return None
        engine = self.config.agent.type if self.config else None
        return self.start(engine)

    # ---- the model's memory (native engine only) ---------------------------
    # Saved sessions, rewind and fork read and replace what the model
    # remembers. Only the native loop keeps that as a plain list of messages;
    # a framework engine keeps its own, out of reach, so these answer None or
    # False there and the caller says so.

    def _memory(self):
        from ovat.agent.loop import AgentLoop
        return (self.agent.session if isinstance(self.agent, AgentLoop)
                else None)

    def memory(self) -> list[dict] | None:
        """A copy of what the model remembers; None on a framework engine."""
        session = self._memory()
        return None if session is None else session.snapshot()

    def restore_memory(self, messages: list[dict]) -> bool:
        """Make the model remember `messages` (a resumed conversation).

        The saved system prompt is replaced by this workflow's own: a resumed
        conversation continues with the agent running now, not the one that
        happened to be running when it was saved. A file on disk is not
        trusted to say what the agent IS, so no saved message may be a
        system message, and anything not shaped like a chat message is left
        out."""
        session = self._memory()
        if session is None:
            return False
        from ovat.agent.factory import compose_system_prompt
        from ovat.agent.session import Session
        prompt = compose_system_prompt(self.config, Path(self.cwd)) if self.config else None
        fresh = Session(system_prompt=prompt)
        fresh.messages.extend(dict(m) for m in messages if _chat_message(m))
        self.agent.session = fresh
        return True

    def truncate_memory(self, length: int) -> bool:
        """Forget from message `length` on (rewind). False on a framework
        engine; ValueError when `length` is not a user-message boundary."""
        session = self._memory()
        if session is None:
            return False
        session.truncate(length)
        return True

    def compact(self, instructions: str = "") -> str:
        """/compact: replace what the model remembers with a summary of it.

        One extra model call, with no tools, over the memory written out as
        text (ovat/agent/compaction.py). BLOCKING: the app server runs it on
        a worker thread, so its event loop keeps answering. It holds `busy`
        for the whole call, so a turn cannot start in the middle of it and
        change the memory under it. Any failure (a framework engine, an OVMS
        error, an empty summary) raises CompactionFailed with one sentence
        and leaves the memory as it was. Returns the summary."""
        from ovat.agent.compaction import (CompactionFailed, compacted_memory,
                                           summary_messages, summary_of)
        memory = self.memory()
        if memory is None:
            raise CompactionFailed(
                "Compacting needs the native engine: it is the one whose "
                "memory OVAT can rewrite. `/engine native` switches.")
        with self._lock:
            if self.busy:
                raise CompactionFailed(
                    "Wait for the current turn to end, or interrupt it, "
                    "before compacting.")
            self.busy = True
        try:
            try:
                reply = self.agent.llm.chat(summary_messages(memory,
                                                             instructions))
            except Exception as exc:
                raise CompactionFailed(
                    f"The model could not write the summary, so nothing was "
                    f"compacted: {_brief(exc)}") from exc
            summary = summary_of(reply)
            self.restore_memory(compacted_memory(memory, summary))
            return summary
        finally:
            with self._lock:
                self.busy = False

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

    def steer(self, text: str) -> bool:
        """Give the running turn one more instruction (Ctrl+Enter). The native
        loop reads it before its next model call. False when no turn runs or
        the engine is not native: a framework engine has no step of its own
        to slip it into."""
        from ovat.agent.loop import AgentLoop
        with self._lock:
            if not self.busy or not isinstance(self.agent, AgentLoop):
                return False
        # The loop itself refuses once it has decided its answer is final.
        return self.agent.steer(text)

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
            # The "[Answer from subagent ...]" and "[Reply from remote agent
            # ...]" lines frame the reply for a MODEL; a person who asked
            # directly just wants the reply.
            if answer.startswith(_FRAMES) and "\n" in answer:
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
        # The agent's session keeps the raw reply; only what is SHOWN (and
        # copied) is split, so the front end never has to know the tag shapes.
        reasoning, answer = split_thinking(answer or "")
        self.last_answer = answer
        with self._lock:
            self.busy = False
        self._emit(TurnDone(answer, elapsed, totals, self._sources(trace),
                            failed, cancelled, reasoning))

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
        self.questions.cancel_all()
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
