# ovat/tui/server/__init__.py
"""OVAT's app server: the other end of the copied TUI's wire.

The copied Vibe TUI (ovat/tui/vibe/) never touches an agent. It talks
JSON-RPC to an "app server" through ovat/tui/vibe/app_server/session.py. This
package is that server, written for OVAT's own AgentSession
(ovat/agentic/session.py), so the TUI's code stays an unchanged copy:

    rpc.py         the JSON-RPC loop: framing, dispatch, errors, callbacks
    context.py     one conversation: the AgentSession and what shares it
    projector.py   agent reports -> the session the TUI reads, numbered events
    turns.py       a turn, the queue behind it, how it ends
    approvals.py   the ApprovalBroker as Vibe callbacks
    shell_runs.py  `!command`
    views.py       runtime, config and stats pictures
    methods/       one handler per JSON-RPC method
    bridge.py      tool, subagent and remote events -> history entries

Nothing here imports textual, and nothing talks to a network.

`start_ovat_app_server` wires it all in one process: a memory transport, the
server task, the copied client and its session.
"""
from __future__ import annotations

import asyncio
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

from ovat.agentic.session import AgentSession
from ovat.tui.server.context import ServerContext
from ovat.tui.server.methods import all_routes
from ovat.tui.server.rpc import RpcServer
from ovat.tui.vibe.app_server.client import AppServerClient
from ovat.tui.vibe.app_server.protocol import (ClientCapabilities, ClientInfo,
                                               SessionOptions)
from ovat.tui.vibe.app_server.session import AppServerSession
from ovat.tui.vibe.app_server.transport import memory_transport_pair


class OvatServerHandle:
    """The server side of a running app server. Hold it next to the
    AppServerSession; `close()` is how everything ends."""

    def __init__(self, ctx: ServerContext, server: RpcServer,
                 task: asyncio.Task) -> None:
        self._ctx = ctx
        self._server = server
        self._server_task = task

    @property
    def agent_session(self) -> AgentSession | None:
        """OVAT's own session (the fake one, in tests)."""
        return self._ctx.agent

    @property
    def ready(self) -> bool:
        """The agent is built (False while it builds, or when it could not
        be)."""
        return self._ctx.is_ready

    @property
    def build_error(self) -> str | None:
        """Why there is no agent, when the build failed."""
        return self._ctx.build_error

    def subscribe(self, listener: Callable[[object], None]
                  ) -> Callable[[], None]:
        """Hear every report the AgentSession makes (Ready, BuildFailed,
        TurnDone, Notice...), on the event loop, after the server has handled
        it. OVAT's own app uses it to keep its banner and bottom bar current.
        Returns the function that stops listening."""
        self._ctx.listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._ctx.listeners:
                self._ctx.listeners.remove(listener)
        return unsubscribe

    async def switch_engine(self, engine: str) -> None:
        """`/engine`: build the agent again with another engine. Turns wait
        for it; a failure leaves the old agent working."""
        await self._ctx.rebuild(engine=engine)

    async def switch_workflow(self, path: str) -> None:
        """`/workflow`: build the agent from another workflow file."""
        await self._ctx.rebuild(workflow=path)

    @property
    def running(self) -> bool:
        return not self._server_task.done()

    async def close(self) -> None:
        """Stop the server task, close the agent session, leave nothing
        running. Safe to call twice, and after the client closed first."""
        # A server that is already ending (the client sent session/stop) is
        # waited for, not cancelled: cancelling it would cut its shutdown
        # short and leave the agent session half closed.
        if not self._server_task.done() and not self._server.stopping:
            self._server_task.cancel()
        await asyncio.gather(self._server_task, return_exceptions=True)
        await self._ctx.close()


def serve_on_memory(workflow: str | None, *,
                    session_factory: Callable = AgentSession,
                    cwd: str | None = None):
    """Start the server on one end of a memory transport and return the
    client's end with the handle. For callers (and tests) that want to speak
    JSON-RPC themselves; the app uses start_ovat_app_server."""
    loop = asyncio.get_running_loop()
    client_transport, server_transport = memory_transport_pair()
    ctx = ServerContext(workflow, session_factory, cwd, loop)
    server = RpcServer(server_transport, all_routes(), ctx,
                       on_close=ctx.close)
    ctx.rpc = server
    task = loop.create_task(server.serve(), name="ovat-app-server")
    return client_transport, OvatServerHandle(ctx, server, task)


async def start_ovat_app_server(
        workflow: str | None, *,
        session_factory: Callable = AgentSession, cwd: str | None = None,
        continue_session: bool = False, resume_session_id: str | None = None,
        client_info: ClientInfo | None = None,
        capabilities: ClientCapabilities | None = None
) -> tuple[AppServerSession, OvatServerHandle]:
    """Start the server, connect the copied client to it, open a session.

    `session_factory(workflow, sink, cwd=...)` makes the AgentSession; the
    default is the real one, a test passes a fake. Returns the client's
    AppServerSession (what the copied app uses) and the handle to the server.

    Closing: `await session.close()` (it sends `session/stop`), then
    `await handle.close()`.
    """
    client_transport, handle = serve_on_memory(
        workflow, session_factory=session_factory, cwd=cwd)
    client = AppServerClient(client_transport)
    try:
        session = await AppServerSession.start(
            client,
            client_info=client_info or ClientInfo(
                name="ovat-agent", version=_version(), title="OVAT agent",
                entrypoint="cli"),
            capabilities=capabilities or ClientCapabilities(
                callback_kinds=["approval", "user_input"]),
            session_options=SessionOptions(cwd=cwd),
            resume_session_id=resume_session_id,
            continue_session=continue_session)
    except BaseException:
        # The client closed itself on the failure (and with it, its end of
        # the wire); the server task and the agent are still ours to stop.
        await client.close()
        await handle.close()
        raise
    return session, handle


def _version() -> str:
    try:
        return version("ovat")
    except PackageNotFoundError:
        return "unknown"
