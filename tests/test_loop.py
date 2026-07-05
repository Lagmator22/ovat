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


# The run trace (Layer 7 observability)

def test_run_trace_records_turns_tokens_and_tool_calls():
    tool_reply = reply("tool_calls", tool_calls=[
        make_tool_call("tc_1", "get_weather", {"city": "Tokyo"})])
    tool_reply["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
    final_reply = reply("stop", content="sunny")
    final_reply["usage"] = {"prompt_tokens": 150, "completion_tokens": 10}

    agent = AgentLoop(FakeLLMProvider([tool_reply, final_reply]),
                      tools=_weather_tool())
    agent.run("weather?")

    trace = agent.last_trace
    assert trace["engine"] == "native"
    assert len(trace["turns"]) == 2
    assert trace["turns"][0]["finish_reason"] == "tool_calls"
    assert trace["turns"][0]["prompt_tokens"] == 100
    (tool_call,) = trace["turns"][0]["tool_calls"]
    assert tool_call["name"] == "get_weather"
    assert tool_call["arguments"] == {"city": "Tokyo"}
    assert tool_call["duration_s"] >= 0
    assert tool_call["result_chars"] > 0
    totals = trace["totals"]
    assert totals == {"turns": 2, "latency_s": totals["latency_s"],
                      "prompt_tokens": 250, "completion_tokens": 30,
                      "tool_calls": 1}


def test_run_trace_survives_replies_without_usage():
    # conftest.reply() carries no usage key; exactly what a provider that
    # doesn't report tokens looks like. The trace must not crash on it.
    agent = AgentLoop(FakeLLMProvider([reply("stop", content="hi")]), tools={})
    agent.run("hello")
    assert agent.last_trace["totals"]["prompt_tokens"] == 0
    assert agent.last_trace["turns"][0]["prompt_tokens"] is None


# The three "model misbehaves" edge cases


def test_none_content_final_answer_becomes_empty_string():
    # A server may finish with content=None; the CLI must never print "None".
    agent = AgentLoop(FakeLLMProvider([reply("stop", content=None)]), tools={})
    assert agent.run("hi") == ""


def test_empty_tool_calls_list_returns_immediately_instead_of_spinning():
    # finish_reason says tool_calls but the list is empty. Re-asking with the
    # same history would replay the same broken reply until max_iterations;
    # the loop must bail after ONE request instead.
    llm = FakeLLMProvider([reply("tool_calls", tool_calls=[])] * 10)
    agent = AgentLoop(llm, tools=_weather_tool(), max_iterations=10)
    out = agent.run("confuse the loop")
    assert "tool_calls but sent none" in out
    assert len(llm.calls) == 1                       # no wasted re-asks


def test_malformed_json_args_are_reported_to_the_model_not_swallowed():
    from types import SimpleNamespace
    called = []
    broken = SimpleNamespace(                        # arguments is NOT valid JSON
        id="tc_1", type="function",
        function=SimpleNamespace(name="get_weather", arguments="{city: Tokyo"),
    )
    llm = FakeLLMProvider([
        reply("tool_calls", tool_calls=[broken]),
        reply("stop", content="recovered"),
    ])
    agent = AgentLoop(llm, tools=_weather_tool(called))
    assert agent.run("broken json") == "recovered"
    assert called == []                              # the tool never ran blind
    tool_msg = next(m for m in agent.session.messages if m["role"] == "tool")
    assert "not valid JSON" in tool_msg["content"]   # the model sees its mistake
