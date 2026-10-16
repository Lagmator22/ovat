# ovat/tui/server/context.py
"""What the method handlers share: the agent session and the public session.

Note to myself: one server, one conversation. This object owns

- the AgentSession (OVAT's agent: built on a worker thread, reports through
  one sink callback from worker threads),
- the Projector (the session as the copied TUI reads it, numbered events),
- the turn queue, the open approvals and the running `!` command,

and it is the ONE place a report from a worker thread crosses onto the event
loop: `_sink` calls `loop.call_soon_threadsafe(dispatch, event)`, and
everything after that runs on the loop, in the order the session sent it.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ovat.agent.events import (REMOTE_STATUS, SUBAGENT_STATUS, TOOL_ENDED,
                              TOOL_STARTED)
from ovat.agentic import session as s
from ovat.tui.server import views
from ovat.tui.server.approvals import Approvals
from ovat.tui.server.projector import Projector, now_ms
from ovat.tui.server.rpc import RpcError, RpcServer
from ovat.tui.server.shell_runs import ShellRuns
from ovat.tui.server.turns import BuildFailedError, TurnManager
from ovat.tui.vibe.app_server.models import (IdleSessionStatus, PublicSession,
                                             PublicSessionState)
from ovat.tui.vibe.app_server.protocol import (AgentConfig, ClientCapabilities,
                                               ClientInfo, InitializeParams,
                                               ProtocolErrorCode,
                                               RuntimeSnapshot,
                                               RuntimeUpdatedParams)

# How long closing waits for an agent that is still being built, before it
# closes the session anyway.
BUILD_JOIN_SECONDS = 20.0
# And how long closing waits for the session to close, so a stuck OVMS
# shutdown cannot keep the app from exiting.
CLOSE_WAIT_SECONDS = 60.0


class ServerContext:
    def __init__(self, workflow: str | None, session_factory: Callable,
                 cwd: str | None, loop: asyncio.AbstractEventLoop) -> None:
        self.workflow = workflow
        self.session_factory = session_factory
        self.cwd = cwd
        self.loop = loop
        self.rpc: RpcServer | None = None
        self.client_info: ClientInfo | None = None
        self.capabilities = ClientCapabilities()
        self.agent: s.AgentSession | None = None
        self.projector: Projector | None = None
        self.info: dict = {}
        self.build_error: str | None = None
        self.ready = asyncio.Event()
        self.profile = views.DEFAULT_PROFILE
        self.stats = views.StatsBook()
        self.turns = TurnManager(self)
        self.approvals = Approvals(self)
        self.shell = ShellRuns(self)
        self.lifecycle = asyncio.Lock()
        self.closed = False
        self._build_thread: threading.Thread | None = None
        # Things in this process that want every report too (OVAT's own app:
        # its banner and bottom bar follow the agent). Called on the loop,
        # after the server has handled the report.
        self.listeners: list[Callable[[Any], None]] = []
        # While /engine or /workflow rebuilds: what to go back to if it fails.
        self._switch: _Switch | None = None
        # While > 0, plain notices are dropped (see new_conversation).
        self._quiet_notices = 0

    # ---- the handshake ---------------------------------------------------------

    def on_initialize(self, params: InitializeParams) -> None:
        self.client_info = params.client_info
        self.capabilities = params.capabilities

    def report_error(self, message: str) -> None:
        if self.projector is not None:
            self.projector.error(message)

    # ---- the session -----------------------------------------------------------

    def require_session(self, session_id: str | None) -> Projector:
        if self.projector is None:
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "Start, resume, or continue a session before "
                           "using this method")
        if session_id is not None and session_id != self.projector.session.id:
            raise RpcError(ProtocolErrorCode.NOT_FOUND,
                           f"Session not found: {session_id}")
        return self.projector

    def open_session(self, options: AgentConfig
                     ) -> tuple[PublicSessionState, Callable[[], None]]:
        """Create the session and its agent. The agent starts building once
        the response has gone out (`after`), so a report from it can never
        overtake the answer that tells the client the session exists."""
        if self.projector is not None:
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "A session is already open on this server")
        cwd = options.cwd or options.workdir or self.cwd
        agent = self.session_factory(self.workflow, self._sink, cwd=cwd)
        self.cwd = cwd or getattr(agent, "cwd", None)
        self.agent = agent
        named = views.profile(options.agent or "")
        self.profile = (views.AUTO_APPROVE_PROFILE if options.auto_approve
                        else named.name if named else views.DEFAULT_PROFILE)
        agent.broker.auto_approve = self.profile == views.AUTO_APPROVE_PROFILE
        stamp = now_ms()
        active, _ = views.agent_summaries(self.info, self.profile)
        session = PublicSession(
            id=str(uuid4()), status=IdleSessionStatus(), created_at=stamp,
            updated_at=stamp, cwd=self.cwd,
            workspace_roots=[self.cwd] if self.cwd else [], agent=active)
        self.projector = Projector(session, self.rpc.notify)
        self.ready.clear()

        def after() -> None:
            try:
                self._build_thread = agent.start()
            except Exception as exc:
                self.build_error = f"Could not start the agent: {exc}"
                self.ready.set()
        return self.projector.state(), after

    async def wait_ready(self) -> None:
        """Until the agent is built. Raises BuildFailedError with the reason
        the session gave when it could not be."""
        await self.ready.wait()
        if self.build_error is not None:
            raise BuildFailedError(self.build_error)

    @property
    def is_ready(self) -> bool:
        return self.ready.is_set() and self.build_error is None

    # ---- the pictures ----------------------------------------------------------

    def runtime(self) -> RuntimeSnapshot:
        return views.runtime_snapshot(self.info, self.workflow, self.profile,
                                      self.stats.snapshot())

    def refresh_runtime(self) -> None:
        """Tell the client the runtime changed (the agent finished building,
        the profile switched)."""
        if self.projector is None or self.rpc is None:
            return
        active, _ = views.agent_summaries(self.info, self.profile)
        model = self.info.get("model") or None
        self.projector.change_session(model=model, agent=active)
        self.rpc.notify("runtime/updated", RuntimeUpdatedParams(
            session_id=self.projector.session.id, runtime=self.runtime()))

    def record_totals(self, event: s.TurnDone) -> None:
        """Fold a finished turn into the numbers the stats report."""
        totals = event.totals or {}
        book = self.stats
        if self.agent is not None:
            session_totals = self.agent.totals
            book.turns = session_totals["turns"]
            book.prompt_tokens = session_totals["prompt_tokens"]
            book.completion_tokens = session_totals["completion_tokens"]
        if totals.get("prompt_tokens") is not None:
            book.last_prompt = totals["prompt_tokens"]
        book.last_completion = totals.get("completion_tokens")
        book.last_seconds = float(event.elapsed or 0.0)
        if self.projector is not None:
            book.tool_calls = dict(self.projector.tool_calls)

    # ---- reports from the agent session (worker threads) -----------------------

    def _sink(self, event: Any) -> None:
        try:
            self.loop.call_soon_threadsafe(self.dispatch, event)
        except RuntimeError:
            pass                    # the loop is closed: the server is gone

    def dispatch(self, event: Any) -> None:
        """One report from the session, now on the event loop. A failure in
        here is reported to the client and never kills the server."""
        if self.closed or self.projector is None:
            return
        handler = getattr(self, f"_on_{type(event).__name__}", None)
        if handler is not None:
            try:
                handler(event)
            except Exception as exc:
                self.report_error(f"{type(event).__name__}: "
                                  f"{type(exc).__name__}: {exc}")
        for listener in list(self.listeners):
            try:
                listener(event)
            except Exception as exc:
                self.report_error(f"{type(event).__name__}: a listener "
                                  f"failed: {type(exc).__name__}: {exc}")

    def _on_Ready(self, event: s.Ready) -> None:
        self.info = event.info
        self.build_error = None
        self._switch = None
        self.workflow = getattr(self.agent, "workflow", None) or self.workflow
        self.ready.set()
        self.refresh_runtime()

    def _on_BuildFailed(self, event: s.BuildFailed) -> None:
        switch, self._switch = self._switch, None
        if switch is not None and switch.agent_kept:
            # /engine or /workflow could not build the new one. The old agent
            # was never closed, so it stays, and its workflow with it.
            self.agent.workflow = switch.workflow
            self.build_error = None
        else:
            self.build_error = event.text
        self.ready.set()
        # Said once, where the person reads (the app draws it in the
        # transcript): readiness itself does not fail, see ready_wait.
        self.projector.error(event.text)

    def _on_TurnStarted(self, event: s.TurnStarted) -> None:
        self.turns.agent_started()

    def _on_ReasoningDelta(self, event: s.ReasoningDelta) -> None:
        self.turns.on_reasoning_delta(event.text)

    def _on_AnswerDelta(self, event: s.AnswerDelta) -> None:
        self.turns.on_answer_delta(event.text)

    def _on_AgentActivity(self, event: s.AgentActivity) -> None:
        if event.event.type not in (TOOL_STARTED, TOOL_ENDED, SUBAGENT_STATUS,
                                    REMOTE_STATUS):
            return
        if event.event.type == TOOL_STARTED:
            self.turns.flush_deltas(close_entries=True)
        kinds = {t["name"]: t["kind"] for t in self.info.get("tools", [])}
        self.projector.agent_event(event.event, kinds)

    def _on_ApprovalNeeded(self, event: s.ApprovalNeeded) -> None:
        self.approvals.request(event.request)

    def _on_TurnDone(self, event: s.TurnDone) -> None:
        self.turns.finish(event)

    def _on_ShellLine(self, event: s.ShellLine) -> None:
        self.shell.on_line(event)

    def _on_ShellDone(self, event: s.ShellDone) -> None:
        self.shell.on_done(event)

    def _on_Notice(self, event: s.Notice) -> None:
        if self._quiet_notices:
            return
        # The protocol has a warning and an error, and no plain "info". An
        # info notice goes out as a warning with the code "info", so a front
        # end that wants to draw it differently can.
        if event.level == "error":
            self.projector.error(event.text)
        else:
            self.projector.warning(
                event.text, code="info" if event.level == "info" else None)

    # ---- /engine and /workflow -------------------------------------------------

    async def rebuild(self, *, engine: str | None = None,
                      workflow: str | None = None) -> None:
        """Build the agent again with another engine or another workflow.

        The agent is "not ready" while it builds, so a prompt sent meanwhile
        waits for the new one instead of running on an agent that is about
        to be closed. A build that fails leaves the old agent working
        (the error is reported once); see _on_BuildFailed.
        """
        self.require_session(None)
        if self.turns.busy:
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "Wait for the current turn to end, or interrupt "
                           "it, before switching")
        async with self.lifecycle:
            agent = self.agent
            switch = _Switch(workflow=agent.workflow,
                             agent_kept=agent.agent is not None)
            self._switch = switch
            self.ready.clear()
            self.build_error = None
            try:
                self._build_thread = (agent.switch_workflow(workflow)
                                      if workflow is not None
                                      else agent.start(engine))
            except Exception as exc:
                self._switch = None
                agent.workflow = switch.workflow
                if not switch.agent_kept:
                    self.build_error = f"Could not start the agent: {exc}"
                self.ready.set()
                raise RpcError(ProtocolErrorCode.INTERNAL_ERROR,
                               f"Could not start the agent: {exc}") from None

    # ---- /clear ----------------------------------------------------------------

    async def new_conversation(self) -> None:
        if self.turns.busy:
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "Wait for the current turn to end, or interrupt it, "
                           "before clearing the conversation")
        self.approvals.forget()
        self.turns.reset()
        # The session says "New conversation." about itself; the app draws its
        # own confirmation, so that notice (delivered a moment later, in
        # order) is dropped.
        self._quiet_notices += 1
        try:
            rebuilt = self.agent.new_conversation()
        finally:
            self.loop.call_soon(self._unmute_notices)
        if rebuilt is not None:
            # A framework engine forgets by rebuilding; wait for it.
            self.ready.clear()
            self._build_thread = rebuilt
            await self.wait_ready()
        # The session's token totals carry on; the context is empty again.
        self.stats.last_prompt = None
        self.stats.last_completion = None
        self.projector.reset()

    def _unmute_notices(self) -> None:
        self._quiet_notices = max(0, self._quiet_notices - 1)

    # ---- shutdown --------------------------------------------------------------

    async def close(self) -> None:
        """Stop everything this server started. Safe to call twice."""
        if self.closed:
            return
        self.closed = True
        self.turns.close()
        self.shell.close()
        agent, build = self.agent, self._build_thread
        if agent is None:
            return
        done: asyncio.Future = self.loop.create_future()

        def work() -> None:
            try:
                if build is not None and build.is_alive():
                    build.join(BUILD_JOIN_SECONDS)
                agent.close()
            except Exception:
                pass
            finally:
                try:
                    self.loop.call_soon_threadsafe(_resolve, done)
                except RuntimeError:
                    pass

        closer = threading.Thread(target=work, daemon=True,
                                  name="ovat-server-close")
        closer.start()
        try:
            await asyncio.wait_for(done, CLOSE_WAIT_SECONDS)
        except asyncio.TimeoutError:
            return                  # a stuck close is left to its daemon thread
        closer.join(1.0)            # it is ending: its last act was `done`


@dataclass
class _Switch:
    """What /engine or /workflow replaces: the workflow to go back to, and
    whether an agent is alive that a failed build leaves in place."""
    workflow: str | None
    agent_kept: bool


def _resolve(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)
