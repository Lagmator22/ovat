"""The copied MCP browser over a real AgentSession with a fake MCP server."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ovat.agent.factory import build_tools
from ovat.agentic import session as backend
from ovat.config.workflow import WorkflowConfig
from ovat.tools.mcp_client import MCPStdioServer
from ovat.tui.server import start_ovat_app_server
from ovat.tui.vibe.app_server.protocol import AppServerResponseError
from tests.conftest import FakeLLMProvider


class Server:
    closed = []
    started = []

    def __init__(self, command, env=None):
        self.command = command
        self._session = object()
        self._error = None
        self.tools = [SimpleNamespace(name="lookup", description="read docs",
                                     inputSchema={"type": "object", "properties": {}})]
        self.started.append(self)

    status = MCPStdioServer.status

    def call_tool(self, name, arguments):
        return "docs"

    def close(self):
        self._session = None
        self.closed.append(self)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from ovat.agent import factory
    from ovat.tools import mcp_client
    Server.closed, Server.started = [], []
    monkeypatch.setattr(mcp_client, "MCPStdioServer", Server)
    monkeypatch.setattr(factory, "build_llm", lambda c: FakeLLMProvider([]))
    monkeypatch.setattr(backend, "_model_choices", lambda c: [{"id": "m", "label": "m"}])
    config = WorkflowConfig(model={"name": "m"}, tools=[{
        "name": "docs", "type": "mcp_stdio", "command": ["fake"],
        "env": {"SECRET": "hidden"}}])
    def make(workflow, sink, cwd=None):
        return backend.AgentSession(workflow, sink, cwd=str(tmp_path),
                                    load=lambda path: config)
    return config, make


@asynccontextmanager
async def served(rig):
    _, make = rig
    session, handle = await start_ovat_app_server("w.yml", session_factory=make)
    try:
        await session.resources.runtime.wait_until_ready()
        yield session, handle
    finally:
        await session.close()
        await handle.close()


def test_public_status_reports_connection_tools_and_hides_environment(rig):
    config, _ = rig
    owned = []
    tools = build_tools(config, servers=owned)
    assert tools["lookup"]["mcp_source"] == "docs"
    status = owned[0].status()
    assert status == {"connected": True, "error": None,
                      "command": ["fake"], "tools": ["lookup"]}
    owned[0].close()
    assert owned[0].status()["connected"] is False


def test_read_refresh_and_toggle_rebuild_with_memory_and_no_file_write(rig):
    async def go():
        async with served(rig) as (session, handle):
            real = handle.agent_session
            real.agent.session.add_user("remember this")
            before = real.memory()
            state = await session.resources.mcp.read()
            assert state.connector_error is None
            assert state.sources[0].name == "docs"
            assert state.sources[0].status.value == "connected"
            assert state.sources[0].tools[0].enabled
            assert "hidden" not in state.model_dump_json()
            state = await session.resources.mcp.toggle("docs", source="server", disabled=True)
            assert state.sources[0].status.value == "disabled"
            assert "lookup" not in real.agent.tools
            assert len(Server.started) == 1  # a disabled server is not launched
            assert real.memory() == before
            state = await session.resources.mcp.toggle("docs", source="server", disabled=False)
            assert "lookup" in real.agent.tools
            assert state.sources[0].status.value == "connected"
            state = await session.resources.mcp.toggle(
                "docs", source="server", disabled=True, tool_name="lookup")
            assert not state.sources[0].tools[0].enabled
            assert "lookup" not in real.agent.tools
            assert real.memory() == before
            assert (await session.resources.mcp.refresh()).sources[0].tools[0].enabled is False
            assert real.config.tools[0].name == "docs"
    asyncio.run(go())


def test_bad_selection_is_refused_and_a_failed_build_rolls_back(rig, monkeypatch):
    async def go():
        async with served(rig) as (session, handle):
            real = handle.agent_session
            for name, tool in [("missing", None), ("docs", "missing")]:
                with pytest.raises(AppServerResponseError):
                    await session.resources.mcp.toggle(
                        name, source="server", disabled=True, tool_name=tool)
            old = real.agent
            def fail(config):
                raise ValueError("bad server")
            monkeypatch.setattr(real, "_build", fail)
            with pytest.raises(AppServerResponseError):
                await session.resources.mcp.toggle("docs", source="server", disabled=True)
            assert real.agent is old
            assert not real.mcp_selection.disabled
            assert (await session.resources.mcp.read()).sources[0].status.value == "connected"
    asyncio.run(go())


def test_busy_turn_refuses_mcp_changes(rig):
    async def go():
        async with served(rig) as (session, handle):
            real = handle.agent_session
            real.busy = True
            # The wire turn state, rather than the worker's flag, is the gate.
            ctx = handle._ctx
            monkey = pytest.MonkeyPatch()
            monkey.setattr(type(ctx.turns), "busy", property(lambda s: True))
            try:
                with pytest.raises(AppServerResponseError):
                    await session.resources.mcp.toggle("docs", source="server", disabled=True)
                assert not real.mcp_selection.disabled
            finally:
                monkey.undo()
                real.busy = False
    asyncio.run(go())


def test_mcp_browser_opens_toggles_and_closes(rig):
    from tests.test_vibe_agent_tui import running, say, until, screen_text
    from ovat.tui.vibe.cli.textual_ui.widgets.mcp_app import MCPApp
    async def go():
        _, make = rig
        async with running(session_factory=make, ready="native") as (
                app, pilot, real, handle):
            await say(pilot, "/mcp")
            await until(pilot, lambda: len(app.query(MCPApp)))
            await pilot.press("d")
            await until(pilot, lambda: "docs" in real.mcp_selection.disabled)
            await pilot.press("escape")
            await until(pilot, lambda: not app.query(MCPApp))
            await say(pilot, "/mcp login docs")
            await until(pilot, lambda: "ConfigureMCP" in screen_text(app).replace(" ", ""))
    asyncio.run(go())
