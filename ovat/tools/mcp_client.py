# ovat/tools/mcp_client.py
"""The MCP client: plug ANY MCP server into a workflow as tools.

This is the other half of the MCP story. search_docs.py and transcribe.py can
already RUN as MCP servers (`python -m ovat.tools.search_docs`); this file is
the piece that lets an OVAT agent actually CONNECT to one — ours or anyone
else's — over stdio, using the official `mcp` SDK:

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
and — the subtle part — one long-lived "manager" coroutine that connects,
waits, and disconnects. anyio cancel scopes must be entered and exited by the
SAME task, so close() cannot unwind the connection from outside; it just sets
an event and the manager unwinds itself.
"""
import asyncio
import atexit
import threading
from contextlib import AsyncExitStack


class MCPStdioServer:
    """One MCP server subprocess + a live session, with a synchronous face."""

    def __init__(self, command: list[str], connect_timeout: float = 30.0):
        if not command:
            raise ValueError("mcp_stdio needs a command to launch, e.g. "
                             "[\"python\", \"-m\", \"ovat.tools.search_docs\"]")
        self.command = list(command)
        self.tools: list = []            # mcp Tool objects, filled on connect
        self._session = None
        self._error: BaseException | None = None
        self._shutdown: asyncio.Event | None = None

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever,
                                        daemon=True, name=f"mcp:{command[0]}")
        self._thread.start()

        ready = threading.Event()
        asyncio.run_coroutine_threadsafe(self._manager(ready), self._loop)
        if not ready.wait(timeout=connect_timeout):
            self.close()
            raise TimeoutError(f"MCP server {command} did not come up within "
                               f"{connect_timeout:.0f}s")
        if self._error is not None:
            error = self._error
            self.close()
            raise RuntimeError(f"could not connect to MCP server {command}: "
                               f"{error}") from error
        # Belt and braces: if the owner never calls close(), the interpreter
        # shutdown does. Closing twice is safe.
        atexit.register(self.close)

    async def _manager(self, ready: threading.Event) -> None:
        """Connect, serve until told to stop, then unwind — all in ONE task."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        try:
            async with AsyncExitStack() as stack:
                params = StdioServerParameters(command=self.command[0],
                                               args=self.command[1:])
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.tools = (await session.list_tools()).tools
                self._session = session
                self._shutdown = asyncio.Event()
                ready.set()
                await self._shutdown.wait()     # serve until close() fires this
        except BaseException as exc:            # noqa: broad on purpose —
            self._error = exc                   # whatever went wrong, report it
            ready.set()
        finally:
            self._session = None

    def call_tool(self, name: str, args: dict, timeout: float = 120.0) -> str:
        """Call one remote tool and flatten its content blocks into a string.

        The agent loop expects a plain string result, exactly like a builtin
        tool. Errors come back as readable strings for the same reason the
        loop's _execute does it: the model should read the failure and adapt.
        """
        session = self._session
        if session is None:
            return f"Error: MCP server {self.command} is not connected."
        future = asyncio.run_coroutine_threadsafe(
            session.call_tool(name, args), self._loop)
        try:
            result = future.result(timeout=timeout)
        except Exception as exc:
            return f"Error: MCP tool '{name}' failed: {exc}"
        parts = [block.text for block in result.content
                 if getattr(block, "text", None) is not None]
        return "\n".join(parts) if parts else str(result.content)

    def close(self) -> None:
        """Stop the server connection and the loop thread. Safe to call twice."""
        if self._shutdown is not None and not self._shutdown.is_set():
            self._loop.call_soon_threadsafe(self._shutdown.set)
        # Give the manager a moment to unwind its exit stack cleanly.
        deadline = threading.Event()
        deadline.wait(0.1)
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5)


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
