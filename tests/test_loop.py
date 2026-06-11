# tests/test_loop.py
"""Tests for AgentLoop, using the FakeLLMProvider so no real model is needed.

Note to myself: I script the fake model to first ask for a tool, then answer
in words. That exercises the full ASK READ ACT REPORT cycle without a GPU.
Each test below pins down one behaviour I promised would be robust.
"""
from ovat.agent.loop import AgentLoop
from tests.conftest import FakeLLMProvider, make_tool_call, reply


def _weather_tool(calls_log=None):
    """A tiny dummy tool I can hand to the loop in tests."""
    def get_weather(city: str) -> str:
        if calls_log is not None:
            calls_log.append(city)
        return f"It is sunny in {city}."

    schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    return {"get_weather": {"schema": schema, "function": get_weather}}


def test_loop_answers_directly_when_no_tool_needed():
    llm = FakeLLMProvider([reply("stop", content="Hello there.")])
    agent = AgentLoop(llm, tools={})
    assert agent.run("hi") == "Hello there."


def test_loop_runs_a_tool_then_answers():
    called = []
    llm = FakeLLMProvider([
        reply("tool_calls", tool_calls=[make_tool_call("tc_1", "get_weather", {"city": "Tokyo"})]),
        reply("stop", content="It is sunny in Tokyo."),
    ])
    agent = AgentLoop(llm, tools=_weather_tool(called))
    out = agent.run("weather in Tokyo?")

    assert out == "It is sunny in Tokyo."
    assert called == ["Tokyo"]                       # my real function actually ran
    roles = [m["role"] for m in agent.session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_loop_runs_parallel_tool_calls():
    called = []
    llm = FakeLLMProvider([
        reply("tool_calls", tool_calls=[
            make_tool_call("tc_1", "get_weather", {"city": "Tokyo"}),
            make_tool_call("tc_2", "get_weather", {"city": "Paris"}),
        ]),
        reply("stop", content="done"),
    ])
    agent = AgentLoop(llm, tools=_weather_tool(called))
    agent.run("two cities")

    assert called == ["Tokyo", "Paris"]
    tool_ids = {m["tool_call_id"] for m in agent.session.messages if m["role"] == "tool"}
    assert tool_ids == {"tc_1", "tc_2"}


def test_loop_handles_unknown_tool():
    llm = FakeLLMProvider([
        reply("tool_calls", tool_calls=[make_tool_call("tc_1", "no_such_tool", {})]),
        reply("stop", content="recovered"),
    ])
    agent = AgentLoop(llm, tools={})
    assert agent.run("call a missing tool") == "recovered"
    tool_msg = next(m for m in agent.session.messages if m["role"] == "tool")
    assert "not available" in tool_msg["content"]


def test_loop_handles_tool_that_raises():
    def boom() -> str:
        raise ValueError("kaboom")

    tools = {"boom": {
        "schema": {"type": "function", "function": {
            "name": "boom", "description": "always raises",
            "parameters": {"type": "object", "properties": {}}}},
        "function": boom,
    }}
    llm = FakeLLMProvider([
        reply("tool_calls", tool_calls=[make_tool_call("tc_1", "boom", {})]),
        reply("stop", content="handled"),
    ])
    agent = AgentLoop(llm, tools=tools)
    assert agent.run("trigger an error") == "handled"
    tool_msg = next(m for m in agent.session.messages if m["role"] == "tool")
    assert "kaboom" in tool_msg["content"]


def test_loop_stops_at_max_iterations():
    # The fake always asks for a tool and never says stop. The guard must end it.
    endless = [reply("tool_calls",
                     tool_calls=[make_tool_call(f"tc_{i}", "get_weather", {"city": "X"})])
               for i in range(50)]
    agent = AgentLoop(FakeLLMProvider(endless), tools=_weather_tool(), max_iterations=3)
    assert "max of 3 steps" in agent.run("loop forever")
