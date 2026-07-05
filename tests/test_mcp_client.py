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
