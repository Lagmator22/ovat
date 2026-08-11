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
                      "tool_calls": 1,
                      # False on a healthy run; True when a tool call came back
                      # as text and never ran. Asserted here as part of the
                      # exact shape so a future key cannot be added silently.
                      "undecoded_tool_call": False,
                      "empty_answer": False,
                      # False on a healthy run. `ovat run` reads this to decide
                      # its exit code: the loop returns its failures AS the
                      # answer text, so without a flag a failed run exited 0
                      # and no script could tell it apart from a good one.
                      "failed": False}


def test_run_trace_survives_replies_without_usage():
    # conftest.reply() carries no usage key; exactly what a provider that
    # doesn't report tokens looks like. The trace must not crash on it.
    #
    # The total is None, NOT zero. This assertion used to read == 0, which
    # encoded a real bug: a server that sends no usage block would produce
    # "prompt_tokens": 0 in the trace, and that reads as "this run used no
    # tokens" rather than "nobody told us". A benchmark built on that number
    # is quietly wrong, which is the worst kind.
    agent = AgentLoop(FakeLLMProvider([reply("stop", content="hi")]), tools={})
    agent.run("hello")
    assert agent.last_trace["totals"]["prompt_tokens"] is None
    assert agent.last_trace["turns"][0]["prompt_tokens"] is None
    # Turn count is genuinely known, so it stays a number.
    assert agent.last_trace["totals"]["turns"] == 1


# The three "model misbehaves" edge cases


def test_none_content_is_reported_rather_than_returned_as_silence():
    """A server may finish with content=None. Two things must be true.

    The original contract here was only the first: never print the literal
    string "None". That still holds. But it returned "" instead, and an empty
    answer turns out to be a SYMPTOM, not a benign edge case -- on live
    hardware a mismatched tool_parser swallowed a reply and produced exactly
    this, with bench scoring the row ok. Returning silence made that
    indistinguishable from a model that simply had nothing to add.

    So content=None is now reported as the failure it almost always is. The
    literal "None" still never appears.
    """
    agent = AgentLoop(FakeLLMProvider([reply("stop", content=None)]), tools={})
    out = agent.run("hi")
    assert "None" not in out                  # the original guarantee
    assert "Error" in out and "no answer" in out.lower()
    assert agent.last_trace["totals"]["empty_answer"] is True


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


def test_an_undecoded_tool_call_is_not_returned_as_the_answer():
    """Raw tool-call markup arriving with finish_reason "stop" is a FAILURE.

    Measured on the AI PC (v0.9.3, tool_parser: qwen3coder): on one OVMS
    instance the model emitted malformed markup -- <parameter=search_docs>
    where it should write <function=search_docs>, plus a closing </function>
    with no opener -- so the server's parser legitimately could not decode it
    and handed the whole thing back as content. finish_reason "stop",
    tool_calls 0, exit 0, and that markup became the ANSWER.

    4/4 consecutive runs on that instance; 12/13 fine after a restart. The
    cause is unresolved and may be unfixable from here -- OVAT cannot stop a
    model emitting bad output. What it CAN stop is presenting the result as a
    fluent answer while no tool ran, which is the worst available outcome.

    The loop already guards the mirror case just below this one (finish_reason
    IS tool_calls but the list is empty), so this is the same treatment for the
    same class of defect.
    """
    from ovat.agent.loop import AgentLoop

    class RawMarkupLLM:
        def chat(self, messages, tools=None):
            return {"finish_reason": "stop",
                    "content": ("<tool_call>\n<parameter=search_docs>\n"
                                "<parameter=query>budget</parameter>\n"
                                "</function>\n</tool_call>"),
                    "tool_calls": None, "usage": None, "raw": None}

    agent = AgentLoop(RawMarkupLLM(), tools={}, max_iterations=3)
    answer = agent.run("what is the memory budget?")

    assert "<tool_call>" not in answer, "raw markup became the answer"
    assert "Error" in answer                      # it is reported AS a failure
    assert "tool_parser" in answer                # and names the likely cause
    # The trace must not read as a clean run that simply used no tools.
    assert agent.last_trace["totals"].get("undecoded_tool_call") is True


def test_a_normal_answer_is_untouched_by_that_guard():
    """The guard must not fire on ordinary prose."""
    from ovat.agent.loop import AgentLoop

    class PlainLLM:
        def chat(self, messages, tools=None):
            return {"finish_reason": "stop", "content": "Under 8 GB.",
                    "tool_calls": None, "usage": None, "raw": None}

    agent = AgentLoop(PlainLLM(), tools={}, max_iterations=3)
    assert agent.run("q") == "Under 8 GB."
    assert not agent.last_trace["totals"].get("undecoded_tool_call")


def test_a_reply_that_says_nothing_is_reported_not_returned_as_an_answer():
    """The quietest member of the no-tool-called family.

    Found on the AI PC (v0.9.6) by deliberately forcing tool_parser: hermes3
    onto a Qwen3.5 model. hermes3 does not merely fail to decode that format --
    it SWALLOWS the call. The reply was reasoning, then </think>, then nothing:

        tool_calls 0, undecoded_tool_call False, and `bench` scored the row ok.

    So neither existing guard sees it. There is no markup left for
    looks_like_undecoded_tool_call to find, and finish_reason is a clean
    "stop". An empty answer is the only remaining evidence, which makes it
    worth treating as evidence: a model with genuinely nothing to say is
    vanishingly rare, a pipeline that ate the reply is not.
    """
    from ovat.agent.loop import AgentLoop

    class SwallowedLLM:
        def chat(self, messages, tools=None):
            # Reasoning, terminator, and then nothing at all.
            return {"finish_reason": "stop",
                    "content": "The user wants the memory budget.</think>",
                    "tool_calls": None, "usage": None, "raw": None}

    agent = AgentLoop(SwallowedLLM(), tools={}, max_iterations=3)
    answer = agent.run("what is the memory budget?")

    assert answer, "an empty answer was returned as if it were a reply"
    assert "Error" in answer
    assert "no answer" in answer.lower()
    assert agent.last_trace["totals"].get("empty_answer") is True


def test_a_real_answer_is_never_mistaken_for_an_empty_one():
    """Reasoning followed by actual content must pass through untouched."""
    from ovat.agent.loop import AgentLoop

    class ThinkingLLM:
        def chat(self, messages, tools=None):
            return {"finish_reason": "stop",
                    "content": "Deciding what they meant.</think>Under 8 GB.",
                    "tool_calls": None, "usage": None, "raw": None}

    agent = AgentLoop(ThinkingLLM(), tools={}, max_iterations=3)
    # The raw text is preserved -- stripping reasoning is a DISPLAY concern,
    # and --trace plus the session must keep what the model actually sent.
    assert "Under 8 GB." in agent.run("q")
    assert not agent.last_trace["totals"].get("empty_answer")
