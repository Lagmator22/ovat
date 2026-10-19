# tests/test_mcp_client.py
"""Tests for the MCP stdio client: the agent really speaking MCP.

Note to myself: the best possible test server is our OWN search_docs, launched
exactly the way a user's YAML would launch it (`python -m ovat.tools.search_docs`).
That finally exercises the `mcp.run()` path AND the client in one go, over a
real subprocess and a real stdio wire. No retriever is configured in the child
process, so the tool answers in stub mode; perfect: a deterministic reply
that proves the round trip without any model on disk.
"""
import sys
import time
import uuid

import psutil
import pytest

from ovat.agent.factory import build_agent, build_tools
from ovat.agent.loop import AgentLoop
from ovat.config.workflow import WorkflowConfig
from ovat.tools.mcp_client import MCPStdioServer, openai_schema_from_mcp_tool
from tests.conftest import FakeLLMProvider, make_tool_call, reply

SEARCH_DOCS_SERVER = [sys.executable, "-m", "ovat.tools.search_docs"]


@pytest.fixture(scope="module")
def server():
    """One shared connection for the module: spawning is the slow part."""
    server = MCPStdioServer(SEARCH_DOCS_SERVER)
    yield server
    server.close()


def test_client_discovers_the_advertised_tools(server):
    assert [tool.name for tool in server.tools] == ["search_docs"]
    schema = openai_schema_from_mcp_tool(server.tools[0])
    assert schema["function"]["name"] == "search_docs"
    assert "query" in schema["function"]["parameters"]["properties"]


def test_call_tool_round_trips_over_the_wire(server):
    out = server.call_tool("search_docs", {"query": "hello over mcp"})
    # The child has no retriever wired, so stub mode proves the round trip.
    assert "[stub]" in out
    assert "hello over mcp" in out


def test_call_tool_reports_wire_errors_as_readable_strings(server):
    out = server.call_tool("no_such_tool", {})
    assert out.startswith("Error:") or "no_such_tool" in out


def test_close_is_idempotent():
    probe = MCPStdioServer(SEARCH_DOCS_SERVER)
    probe.close()
    probe.close()                    # second close must be a quiet no-op
    assert probe.call_tool("search_docs", {"query": "x"}).startswith("Error:")


# factory wiring: type: mcp_stdio in the YAML

def _mcp_cfg():
    return WorkflowConfig(
        model={"name": "m"},
        tools=[{"name": "search_docs", "type": "mcp_stdio",
                "command": SEARCH_DOCS_SERVER}],
    )


def test_factory_builds_tools_from_an_mcp_server():
    tools = build_tools(_mcp_cfg())
    assert "search_docs" in tools
    assert tools["search_docs"]["schema"]["function"]["name"] == "search_docs"
    out = tools["search_docs"]["function"](query="factory wired")
    assert "[stub]" in out and "factory wired" in out


def test_agent_loop_calls_a_tool_over_mcp_end_to_end():
    # The full promise: a scripted model asks for search_docs, the loop runs
    # it OVER THE WIRE, and the tool result lands back in the history.
    agent = build_agent(_mcp_cfg())
    assert isinstance(agent, AgentLoop)
    agent.llm = FakeLLMProvider([
        reply("tool_calls", tool_calls=[
            make_tool_call("tc_1", "search_docs", {"query": "via mcp"})]),
        reply("stop", content="answered from the docs"),
    ])
    assert agent.run("ask the docs") == "answered from the docs"
    tool_msg = next(m for m in agent.session.messages if m["role"] == "tool")
    assert "[stub]" in tool_msg["content"] and "via mcp" in tool_msg["content"]


def _processes_with(marker: str) -> list:
    """Every live process whose command line carries `marker`."""
    found = []
    for proc in psutil.process_iter(["cmdline"]):
        if marker in (proc.info["cmdline"] or []):
            found.append(proc)
    return found


def test_a_server_that_never_connects_leaves_no_process_behind():
    # A "server" that starts and then never says a word of MCP. The unique
    # marker is an extra argv entry, so this run can find its own child and
    # nothing else. It is an argv list, not py_command(): the client takes a
    # list and passes it to the SDK unchanged, no shell in between.
    marker = f"ovat-mcp-orphan-{uuid.uuid4().hex}"
    command = [sys.executable, "-c", "import time; time.sleep(60)", marker]
    try:
        with pytest.raises(TimeoutError):
            MCPStdioServer(command, connect_timeout=1)
        deadline = time.monotonic() + 3
        while _processes_with(marker) and time.monotonic() < deadline:
            time.sleep(0.1)
        survivors = _processes_with(marker)
        assert survivors == [], (
            f"the server outlived its failed start: pids "
            f"{[proc.pid for proc in survivors]}")
    finally:
        for proc in _processes_with(marker):    # never leak it, pass or fail
            try:
                proc.kill()
            except psutil.Error:
                pass


_HANGS_ON_SHUTDOWN = """
import atexit, time
from fastmcp import FastMCP

mcp = FastMCP("hangs")


@mcp.tool
def ping() -> str:
    return "pong"


atexit.register(time.sleep, 60)    # ignores the end of its stdin
mcp.run()
"""


def test_closing_a_server_that_hangs_on_shutdown_leaves_no_process(tmp_path):
    # close() used to wait a fixed 0.1 s and then stop the event loop, which
    # cut off the SDK's own shutdown before its kill step: a server that did
    # not exit when its stdin closed was left running.
    marker = f"ovat-mcp-hang-{uuid.uuid4().hex}"
    script = tmp_path / "hangs.py"
    script.write_text(_HANGS_ON_SHUTDOWN, encoding="utf-8")
    try:
        server = MCPStdioServer([sys.executable, str(script), marker])
        assert server.call_tool("ping", {}) == "pong"
        server.close()
        deadline = time.monotonic() + 1
        while _processes_with(marker) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert _processes_with(marker) == [], "the server outlived close()"
    finally:
        for proc in _processes_with(marker):
            try:
                proc.kill()
            except psutil.Error:
                pass


def test_mcp_stdio_without_a_command_is_rejected():
    cfg = WorkflowConfig(model={"name": "m"},
                         tools=[{"name": "x", "type": "mcp_stdio"}])
    with pytest.raises(ValueError, match="no.*command"):
        build_tools(cfg)


def test_unknown_tool_type_is_still_rejected():
    cfg = WorkflowConfig(model={"name": "m"},
                         tools=[{"name": "x", "type": "banana"}])
    with pytest.raises(ValueError, match="Unsupported tool type"):
        build_tools(cfg)


# env: what an MCP server is allowed to see
#
# The mcp SDK starts a stdio server with only a short allowlist of the
# parent's variables (HOME, LOGNAME, PATH, SHELL, TERM, USER on POSIX), on
# purpose, so a third-party server does not inherit every secret in the
# shell. OVAT's own servers read OVAT_WHISPER_MODEL, OVAT_VLM_MODEL and
# friends, which therefore never arrived. The spec's answer is an explicit
# per-server env, which is what tools[].env is.

_ENV_SERVER = '''
import os
from fastmcp import FastMCP
mcp = FastMCP("envprobe")

@mcp.tool
def getenv(name: str) -> str:
    """Return one environment variable as this server sees it."""
    return os.environ.get(name, "<unset>")

mcp.run()
'''


def _env_tools(tmp_path, env=None):
    script = tmp_path / "envprobe.py"
    script.write_text(_ENV_SERVER, encoding="utf-8")
    tool = {"name": "envprobe", "type": "mcp_stdio",
            "command": [sys.executable, str(script)]}
    if env is not None:
        tool["env"] = env
    servers = []
    tools = build_tools(WorkflowConfig(model={"name": "m"}, tools=[tool]),
                        servers=servers)
    return tools["getenv"]["function"], servers


def test_a_tool_env_reaches_the_mcp_server(tmp_path, monkeypatch):
    monkeypatch.setenv("OVAT_TEST_SECRET", "from-the-shell")
    getenv, servers = _env_tools(tmp_path, env={
        "OVAT_WHISPER_MODEL": "models/whisper-base",
        "EXPANDED": "${OVAT_TEST_SECRET}"})
    try:
        assert getenv(name="OVAT_WHISPER_MODEL") == "models/whisper-base"
        # ${VAR} is expanded, so a secret can stay in the shell and out of
        # the YAML file.
        assert getenv(name="EXPANDED") == "from-the-shell"
    finally:
        for server in servers:
            server.close()


def test_without_env_the_parent_shell_still_does_not_leak(tmp_path,
                                                         monkeypatch):
    """The isolation is the point of the SDK's default; env opts in, it does
    not switch the default off."""
    monkeypatch.setenv("OVAT_TEST_SECRET", "from-the-shell")
    getenv, servers = _env_tools(tmp_path)
    try:
        assert getenv(name="OVAT_TEST_SECRET") == "<unset>"
    finally:
        for server in servers:
            server.close()


def test_env_on_a_builtin_tool_is_a_config_error():
    """A builtin runs in this process, so an env there would do nothing."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="mcp_stdio"):
        WorkflowConfig(model={"name": "m"},
                       tools=[{"name": "search_docs", "env": {"A": "b"}}])


def test_close_unregisters_atexit_hook(monkeypatch):
    """Closing an MCP server must unregister its atexit callback so closed
    instances do not accumulate in memory across agent rebuilds."""
    import atexit
    registered = []
    unregistered = []
    orig_reg = atexit.register
    orig_unreg = atexit.unregister

    def track_reg(fn, *a, **kw):
        registered.append(fn)
        return orig_reg(fn, *a, **kw)

    def track_unreg(fn):
        unregistered.append(fn)
        return orig_unreg(fn)

    monkeypatch.setattr(atexit, "register", track_reg)
    monkeypatch.setattr(atexit, "unregister", track_unreg)

    srv = MCPStdioServer(SEARCH_DOCS_SERVER)
    assert srv.close in registered
    srv.close()
    assert srv.close in unregistered


def test_call_tool_surfaces_is_error_with_error_prefix(server, monkeypatch):
    """An MCP tool result with isError=True must start with 'Error:' so the
    agent loop and UI treat it as a tool failure instead of normal data."""
    from types import SimpleNamespace

    async def fake_call_tool(name, args):
        return SimpleNamespace(
            isError=True,
            content=[SimpleNamespace(text="rate limit exceeded")],
        )

    monkeypatch.setattr(server._session, "call_tool", fake_call_tool)
    out = server.call_tool("search_docs", {"query": "x"})
    assert out == "Error: rate limit exceeded"


_CRASH_ONCE_SERVER = '''
import os, pathlib, sys
from fastmcp import FastMCP

marker = pathlib.Path(sys.argv[1])
mcp = FastMCP("flaky")

@mcp.tool
def ping() -> str:
    if not marker.exists():
        marker.write_text("crashed", encoding="utf-8")
        os._exit(1)
    return "recovered pong"

mcp.run()
'''


def test_call_tool_reconnects_once_on_transport_drop(tmp_path):
    """When a child MCP server crashes mid-call, call_tool must reconnect
    once and complete the call instead of leaving the session permanently dead."""
    script = tmp_path / "flaky.py"
    script.write_text(_CRASH_ONCE_SERVER, encoding="utf-8")
    marker = tmp_path / "crashed.flag"
    srv = MCPStdioServer([sys.executable, str(script), str(marker)])
    try:
        out = srv.call_tool("ping", {})
        assert out == "recovered pong"
        assert srv.status()["connected"] is True
    finally:
        srv.close()


def test_colliding_mcp_tool_names_are_namespaced_instead_of_overwritten(
        monkeypatch):
    """When two MCP servers advertise a tool with the same name, the second
    must be namespaced as <server>__<tool> rather than silently overwriting
    the first."""
    from types import SimpleNamespace
    from ovat.tools import mcp_client

    class FakeMCP:
        def __init__(self, command, env=None, **kw):
            self.command = command
            self.tools = [SimpleNamespace(
                name="ping", description=f"from {command[0]}",
                inputSchema={"type": "object", "properties": {}})]

        def call_tool(self, name, args):
            return f"{self.command[0]}:{name}"

        def close(self):
            pass

    monkeypatch.setattr(mcp_client, "MCPStdioServer", FakeMCP)
    cfg = WorkflowConfig(
        model={"name": "m"},
        tools=[
            {"name": "alpha", "type": "mcp_stdio", "command": ["srv_a"]},
            {"name": "beta", "type": "mcp_stdio", "command": ["srv_b"]},
        ],
    )
    tools = build_tools(cfg)
    assert "ping" in tools
    assert "beta__ping" in tools
    assert tools["ping"]["function"]() == "srv_a:ping"
    assert tools["beta__ping"]["function"]() == "srv_b:ping"
    assert tools["beta__ping"]["schema"]["function"]["name"] == "beta__ping"


def test_mcp_http_and_sse_transports_wire_headers_and_permissions(monkeypatch):
    """type: mcp_http and type: mcp_sse connect via MCPHttpServer, expand
    ${VAR} headers and token_env, and default to ask on remote hosts."""
    from types import SimpleNamespace
    from ovat.agent.permissions import default_permission
    from ovat.tools import mcp_client

    seen = []

    class FakeHttpServer:
        def __init__(self, url, transport="http", connect_timeout=30.0,
                     headers=None, token_env=None):
            seen.append({
                "url": url,
                "transport": transport,
                "headers": headers,
                "token_env": token_env,
            })
            self.url = url
            self.transport = transport
            self.tools = [SimpleNamespace(
                name=f"remote_{transport}", description="remote tool",
                inputSchema={"type": "object", "properties": {}})]

        def call_tool(self, name, args):
            return f"ok:{self.transport}:{name}"

        def status(self):
            return {"connected": True, "error": None,
                    "url": self.url, "tools": [t.name for t in self.tools]}

        def close(self):
            pass

    monkeypatch.setattr(mcp_client, "MCPHttpServer", FakeHttpServer)
    monkeypatch.setenv("MCP_TOK", "tok-secret")
    cfg = WorkflowConfig(
        model={"name": "m"},
        tools=[
            {
                "name": "linear",
                "type": "mcp_http",
                "url": "https://mcp.linear.app/mcp",
                "token_env": "MCP_TOK",
                "headers": {"X-Workspace": "eng"},
            },
            {
                "name": "local_sse",
                "type": "mcp_sse",
                "url": "http://127.0.0.1:8900/sse",
            },
        ],
    )
    owned = []
    tools = build_tools(cfg, servers=owned)
    assert tools["remote_http"]["permission"] == "ask"
    assert tools["remote_sse"]["permission"] == "allow"
    assert tools["remote_http"]["function"]() == "ok:http:remote_http"
    assert tools["remote_sse"]["function"]() == "ok:sse:remote_sse"
    assert seen[0]["transport"] == "http"
    assert seen[0]["token_env"] == "MCP_TOK"
    assert seen[1]["transport"] == "sse"
    assert default_permission("mcp_http", "https://mcp.linear.app/mcp") == "ask"
    assert default_permission("mcp_sse", "http://localhost:8900/sse") == "allow"

