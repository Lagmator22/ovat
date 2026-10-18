# ovat/tools/mcp_client.py
"""The MCP client: plug ANY MCP server into a workflow as tools.

This is the other half of the MCP story. search_docs.py and transcribe.py can
already RUN as MCP servers (`python -m ovat.tools.search_docs`); this file is
the piece that lets an OVAT agent actually CONNECT to one, ours or anyone
else's, over stdio, using the official `mcp` SDK:

    tools:
      - name: search_docs
        type: mcp_stdio
        command: ["python", "-m", "ovat.tools.search_docs"]

The factory spawns the server, asks it what tools it offers (list_tools),
converts each to the same {schema, function} shape the agent loop already
consumes, and routes calls over the wire (call_tool). The loop never learns
the difference between a builtin and an MCP tool.

Sync-over-async design note: the mcp SDK is async (anyio). OVAT's loop is
synchronous. Each server gets ONE background thread running ONE event loop,
and (the subtle part) one long-lived "manager" coroutine that connects,
waits, and disconnects. anyio cancel scopes must be entered and exited by the
SAME task, so close() cannot unwind the connection from outside; it asks the
manager to stop (an event once connected, its own cancel scope while still
connecting) and WAITS for it to unwind before stopping the loop.
"""
import asyncio
import atexit
import os
import threading
from contextlib import AsyncExitStack

# How long close() waits for the manager to unwind. The SDK's own shutdown
# is bounded at about 6.5 s (mcp/client/stdio.py: 0.5 s to flush, 2 s for
# the server to exit after stdin closes, 2 s after killing its process
# tree, 2 s to reap it), so this covers its worst case.
_UNWIND_TIMEOUT = 10.0


def _is_transport_error(exc: BaseException) -> bool:
    """True when an MCP call failed because the underlying stream or pipe died."""
    import anyio
    if isinstance(exc, (anyio.ClosedResourceError, anyio.BrokenResourceError,
                        BrokenPipeError, ConnectionError, EOFError)):
        return True
    return "connection closed" in str(exc).lower()


class _MCPClientBase:
    """Shared sync-over-async event-loop thread and session manager for MCP."""

    def __init__(self, label: str, connect_timeout: float = 30.0):
        self._label = label
        self._connect_timeout = connect_timeout
        self._reconnect_lock = threading.Lock()
        self.tools: list = []            # mcp Tool objects, filled on connect
        self._session = None
        self._error: BaseException | None = None
        self._shutdown: asyncio.Event | None = None
        self._scope = None               # the manager's anyio cancel scope
        self._abort = False              # close() came before the manager ran
        self._closed = False

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever,
                                        daemon=True, name=f"mcp:{label}")
        self._thread.start()

        try:
            self._start_session()
        except BaseException:
            self.close()
            raise
        # Belt and braces: if the owner never calls close(), the interpreter
        # shutdown does. Closing twice is safe.
        atexit.register(self.close)

    def _start_session(self) -> None:
        self._error = None
        self._shutdown = None
        self._scope = None
        self._abort = False
        ready = threading.Event()
        self._manager_done = asyncio.run_coroutine_threadsafe(
            self._manager(ready), self._loop)
        if not ready.wait(timeout=self._connect_timeout):
            raise TimeoutError(
                f"MCP server {self._label} did not come up within "
                f"{self._connect_timeout:.0f}s")
        if self._error is not None:
            error = self._error
            raise RuntimeError(
                f"could not connect to MCP server {self._label}: "
                f"{error}") from error

    async def _open_streams(self, stack: AsyncExitStack):
        raise NotImplementedError

    async def _manager(self, ready: threading.Event) -> None:
        """Connect, serve until told to stop, then unwind: all in ONE task.

        Everything sits inside one anyio cancel scope, so close() can also
        stop a server that never finishes its handshake. It used to stop the
        event loop instead, which abandoned this task mid-await: the SDK's
        cleanup (close stdin, then kill the server's process tree) never
        ran, and a server that did not answer stayed alive after OVAT gave
        up on it. Cancelling the SCOPE, not the task, matters: the SDK's
        shielded shutdown survives an anyio cancellation, and its own
        comment says a native task.cancel() can defeat it.
        """
        import anyio
        from mcp import ClientSession

        try:
            with anyio.CancelScope() as scope:
                self._scope = scope
                if self._abort:
                    scope.cancel()
                async with AsyncExitStack() as stack:
                    read, write = await self._open_streams(stack)
                    session = await stack.enter_async_context(
                        ClientSession(read, write))
                    await session.initialize()
                    self.tools = (await session.list_tools()).tools
                    self._session = session
                    self._shutdown = asyncio.Event()
                    ready.set()
                    await self._shutdown.wait()   # serve until close() fires it
        except BaseException as exc:            # noqa: broad on purpose:
            self._error = exc                   # whatever went wrong, report it
            ready.set()
        finally:
            self._session = None

    def _reconnect(self) -> bool:
        """Attempt one fresh connection after a dropped stream or crashed server."""
        with self._reconnect_lock:
            if self._closed or not self._loop.is_running():
                return False
            self._loop.call_soon_threadsafe(self._stop_manager)
            try:
                self._manager_done.result(timeout=_UNWIND_TIMEOUT)
            except Exception:
                pass
            try:
                self._start_session()
                return self._session is not None
            except Exception as exc:
                self._error = exc
                self._session = None
                return False

    def call_tool(self, name: str, args: dict, timeout: float = 120.0) -> str:
        """Call one remote tool and flatten its content blocks into a string.

        The agent loop expects a plain string result, exactly like a builtin
        tool. Errors come back as readable strings for the same reason the
        loop's _execute does it: the model should read the failure and adapt.
        """
        session = self._session
        if session is None:
            if not self._reconnect():
                return f"Error: MCP server {self._label} is not connected."
            session = self._session
        future = asyncio.run_coroutine_threadsafe(
            session.call_tool(name, args), self._loop)
        try:
            result = future.result(timeout=timeout)
        except Exception as exc:
            if _is_transport_error(exc) and self._reconnect():
                session = self._session
                try:
                    retry = asyncio.run_coroutine_threadsafe(
                        session.call_tool(name, args), self._loop)
                    result = retry.result(timeout=timeout)
                except Exception as retry_exc:
                    self._error = retry_exc
                    return f"Error: MCP tool '{name}' failed: {retry_exc}"
            else:
                if _is_transport_error(exc):
                    self._error = exc
                    self._session = None
                return f"Error: MCP tool '{name}' failed: {exc}"
        parts = [block.text for block in result.content
                 if getattr(block, "text", None) is not None]
        text = "\n".join(parts) if parts else str(result.content)
        if getattr(result, "isError", False) and not text.startswith("Error:"):
            return f"Error: {text}"
        return text

    def _stop_manager(self) -> None:
        """On the loop thread: ask the manager to unwind, however far it got."""
        if self._shutdown is not None:
            self._shutdown.set()        # connected: the normal way out
        elif self._scope is not None:
            self._scope.cancel()        # still connecting: give up on it
        else:
            self._abort = True          # not started yet: stop at once

    def close(self) -> None:
        """Stop the server, then the loop thread. Safe to call twice.

        Waits for the manager to finish unwinding, which is when the SDK has
        closed the server's stdin and, if it did not exit, killed its whole
        process tree. Stopping the loop before that (it used to wait a fixed
        0.1 s) can leave the server running.
        """
        self._closed = True
        atexit.unregister(self.close)
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop_manager)
            try:
                self._manager_done.result(timeout=_UNWIND_TIMEOUT)
            except Exception:
                pass                    # timed out or failed: stop anyway
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5)


class MCPStdioServer(_MCPClientBase):
    """One MCP server subprocess + a live session, with a synchronous face."""

    def __init__(self, command: list[str], connect_timeout: float = 30.0,
                 env: dict | None = None):
        if not command:
            raise ValueError("mcp_stdio needs a command to launch, e.g. "
                             "[\"python\", \"-m\", \"ovat.tools.search_docs\"]")
        self.command = list(command)
        # Added ON TOP of the SDK's safe allowlist, never instead of it: the
        # SDK merges these over get_default_environment(). ${VAR} is expanded
        # here so a secret can be named in the YAML without being written there.
        self.env = ({k: os.path.expandvars(str(v)) for k, v in env.items()}
                    if env else None)
        super().__init__(label=str(self.command),
                         connect_timeout=connect_timeout)

    async def _open_streams(self, stack: AsyncExitStack):
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command=self.command[0],
                                       args=self.command[1:],
                                       env=self.env)
        return await stack.enter_async_context(stdio_client(params))

    def status(self) -> dict:
        """A public snapshot for the MCP panel, without environment secrets."""
        return {"connected": self._session is not None,
                "error": ("The MCP server disconnected." if self._error
                          is not None else None),
                "command": list(self.command),
                "tools": [tool.name for tool in self.tools]}


class MCPHttpServer(_MCPClientBase):
    """One remote MCP server over Streamable HTTP or SSE, with a sync face."""

    def __init__(self, url: str, transport: str = "http",
                 connect_timeout: float = 30.0,
                 headers: dict[str, str] | None = None,
                 token_env: str | None = None):
        if not url or not url.startswith(("http://", "https://")):
            raise ValueError(
                f"mcp_{transport} needs an http(s):// URL, got {url!r}")
        if transport not in ("http", "sse"):
            raise ValueError(
                f"Unsupported MCP HTTP transport {transport!r}. "
                f"Supported: http, sse.")
        self.url = url
        self.transport = transport
        self._headers = (
            {k: os.path.expandvars(str(v)) for k, v in headers.items()}
            if headers else {}
        )
        self.token_env = token_env
        super().__init__(label=url, connect_timeout=connect_timeout)

    def _request_headers(self) -> dict[str, str] | None:
        merged = dict(self._headers)
        if self.token_env:
            token = os.environ.get(self.token_env, "").strip()
            if not token:
                raise RuntimeError(
                    f"{self.token_env} is not set, so the MCP server at "
                    f"{self.url} has no bearer token")
            merged.setdefault("Authorization", f"Bearer {token}")
        return merged or None

    async def _open_streams(self, stack: AsyncExitStack):
        hdrs = self._request_headers()
        if self.transport == "sse":
            from mcp.client.sse import sse_client
            read, write = await stack.enter_async_context(
                sse_client(self.url, headers=hdrs,
                           timeout=self._connect_timeout))
            return read, write
        from mcp.client.streamable_http import streamablehttp_client
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(self.url, headers=hdrs,
                                  timeout=self._connect_timeout))
        return read, write

    def status(self) -> dict:
        """A public snapshot for the MCP panel, without header secrets."""
        return {"connected": self._session is not None,
                "error": ("The MCP server disconnected." if self._error
                          is not None else None),
                "url": self.url,
                "tools": [tool.name for tool in self.tools]}


def openai_schema_from_mcp_tool(tool) -> dict:
    """An MCP tool advertisement -> the OpenAI-style schema our loop shows."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema
            or {"type": "object", "properties": {}},
        },
    }
