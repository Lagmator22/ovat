# tests/test_events.py
"""The event bus and the one tool wrapper every engine goes through.

What has to hold, because the TUI's live tool cards and the A2A server's
progress stream both stand on it:
- the wrapper changes NOTHING about what a tool returns or raises;
- events arrive in order, and a broken listener cannot hurt the agent;
- `stop` really stops: the native loop before its next model call, every
  engine before its next tool call;
- all four engines fire the same events, because they all call the same
  wrapped function.
"""
import pytest

from ovat.agent.events import (CANCELLED_TEXT, TOOL_ENDED, TOOL_STARTED,
                               EventBus, observed)
from ovat.agent.factory import build_agent, observe_tools
from ovat.agent.loop import AgentLoop
from ovat.config.workflow import WorkflowConfig
from tests.conftest import (ANSWER, FakeLLMProvider, make_tool_call,
                            note_tools, reply)


def _recorder(bus):
    seen = []
    bus.subscribe(seen.append)
    return seen


def test_the_result_object_passes_through_unchanged():
    # The native loop reads search_docs' list of dicts to collect citations.
    # A wrapper that stringified it would silently lose every source.
    chunks = [{"text": "t", "source": "a.md", "distance": 0.1}]
    wrapped = observed("search_docs", lambda query: chunks, EventBus())
    assert wrapped(query="q") is chunks


def test_start_and_end_events_carry_the_same_call_id_in_order():
    bus = EventBus()
    seen = _recorder(bus)
    observed("add", lambda a, b: a + b, bus)(a=1, b=2)
    assert [e.type for e in seen] == [TOOL_STARTED, TOOL_ENDED]
    start, end = seen
    assert start.data["call_id"] == end.data["call_id"]
    assert start.data["arguments"] == {"a": 1, "b": 2}
    assert end.data["status"] == "ok"
    assert end.data["result_chars"] == 1
    assert end.data["duration_s"] >= 0


def test_an_exception_still_reaches_the_engine_and_is_reported():
    bus = EventBus()
    seen = _recorder(bus)

    def boom():
        raise ValueError("bad path")

    with pytest.raises(ValueError, match="bad path"):
        observed("boom", boom, bus)()
    assert seen[-1].type == TOOL_ENDED
    assert seen[-1].data["status"] == "error"
    assert seen[-1].data["result_chars"] is None    # absent, not zero


def test_a_listener_that_raises_does_not_break_the_tool():
    bus = EventBus()

    def broken(event):
        raise RuntimeError("the UI went away")

    bus.subscribe(broken)
    seen = _recorder(bus)            # a second listener still hears it all
    assert observed("one", lambda: 1, bus)() == 1
    assert len(seen) == 2


def test_unsubscribe_stops_the_events():
    bus = EventBus()
    seen = []
    unsubscribe = bus.subscribe(seen.append)
    unsubscribe()
    observed("one", lambda: 1, bus)()
    assert seen == []


def test_stop_skips_the_tool_and_says_so():
    bus = EventBus()
    seen = _recorder(bus)
    ran = []
    bus.stop.set()
    result = observed("t", lambda: ran.append(1), bus)()
    assert result == CANCELLED_TEXT
    assert ran == []                  # never ran
    assert seen == []                 # and nothing claimed it started


def test_event_to_dict_is_flat():
    bus = EventBus()
    seen = _recorder(bus)
    bus.emit("custom", x=1)
    assert seen[0].to_dict() == {"type": "custom", "x": 1}


# ---- wiring: the factory hands every agent a bus and wraps every tool ----

def test_build_agent_gives_the_agent_a_bus_and_wraps_its_tools():
    config = WorkflowConfig(model={"name": "m"},
                            tools=[{"name": "search_docs"}])
    agent = build_agent(config)
    assert isinstance(agent.events, EventBus)
    seen = _recorder(agent.events)
    agent.llm = FakeLLMProvider([
        reply("tool_calls", tool_calls=[
            make_tool_call("c1", "search_docs", {"query": "q"})]),
        reply("stop", content="done"),
    ])
    assert agent.run("hi") == "done"
    assert [e.type for e in seen] == [TOOL_STARTED, TOOL_ENDED]
    assert seen[0].data["name"] == "search_docs"


def test_the_native_trace_is_unchanged_by_the_wrapper():
    # The trace keys are what `ovat run --trace` and the bench read. Pinned
    # so the wrapper cannot quietly move anything.
    tools, _ = note_tools()
    bus = EventBus()
    loop = AgentLoop(FakeLLMProvider([
        reply("tool_calls", tool_calls=[
            make_tool_call("c1", "add_note", {"text": "x"})]),
        reply("stop", content="ok"),
    ]), observe_tools(tools, bus), stop=bus.stop)
    loop.run("q")
    entry = loop.last_trace["turns"][0]["tool_calls"][0]
    assert set(entry) == {"name", "arguments", "duration_s", "result_chars",
                          "sources"}
    assert entry["result_chars"] == len("noted x")


@pytest.mark.parametrize("engine", ["react", "llamaindex", "openai-agents"])
def test_every_framework_engine_fires_the_same_events(engine):
    pytest.importorskip({"react": "langchain_core",
                         "llamaindex": "llama_index.core",
                         "openai-agents": "agents"}[engine])
    from tests.conftest import (_llamaindex_agent, _openai_agents_agent,
                                _react_agent)

    tools, calls = note_tools()
    bus = EventBus()
    seen = _recorder(bus)
    config = WorkflowConfig(model={"name": "m"}, agent={"type": engine})
    build = {"react": _react_agent, "llamaindex": _llamaindex_agent,
             "openai-agents": _openai_agents_agent}[engine]
    agent = build(config, observe_tools(tools, bus), None)
    assert ANSWER in agent.run("q")
    assert calls == ["x"]                         # the real tool still ran
    assert [e.type for e in seen] == [TOOL_STARTED, TOOL_ENDED]
    assert seen[0].data == {"call_id": seen[0].data["call_id"],
                            "name": "add_note", "arguments": {"text": "x"}}


# ---- cancel on the native loop ----

def test_a_stopped_native_loop_makes_no_model_call():
    bus = EventBus()
    llm = FakeLLMProvider([reply("stop", content="never")])
    loop = AgentLoop(llm, {}, stop=bus.stop)
    bus.stop.set()
    assert loop.run("q") == CANCELLED_TEXT
    assert llm.calls == []
    assert loop.last_trace["totals"]["failed"] is True


def test_cancel_between_turns_leaves_a_history_the_next_run_can_use():
    # Stopped after the tool ran, before the next model call. The history
    # must still alternate properly (it ends on an assistant message), or
    # the NEXT question is sent to OVMS as a malformed conversation.
    bus = EventBus()
    tools, _ = note_tools()

    class StopAfterFirst(FakeLLMProvider):
        def chat(self, messages, tools=None):
            out = super().chat(messages, tools)
            bus.stop.set()
            return out

    loop = AgentLoop(StopAfterFirst([
        reply("tool_calls", tool_calls=[
            make_tool_call("c1", "add_note", {"text": "x"})]),
    ]), observe_tools(tools, bus), stop=bus.stop)
    assert loop.run("q") == CANCELLED_TEXT
    roles = [m["role"] for m in loop.session.messages]
    assert roles[-3:] == ["assistant", "tool", "assistant"]


def test_no_stop_flag_means_todays_behaviour():
    loop = AgentLoop(FakeLLMProvider([reply("stop", content="hi")]), {})
    assert loop.run("q") == "hi"


class _StreamingFakeLLM:
    def __init__(self, chunks, finish_reason="stop"):
        self.chunks = list(chunks)
        self.finish_reason = finish_reason

    def chat(self, messages, tools=None, on_token=None):
        parts = []
        for piece in self.chunks:
            parts.append(piece)
            if on_token is not None and on_token(piece):
                break
        return {
            "finish_reason": self.finish_reason,
            "content": "".join(parts),
            "tool_calls": None,
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        }


def test_native_loop_streams_reasoning_and_answer_deltas_when_tags_split():
    from ovat.agent.events import ANSWER_DELTA, REASONING_DELTA
    bus = EventBus()
    seen = _recorder(bus)
    llm = _StreamingFakeLLM([
        "<th", "ink>check ", "policy</th", "ink>New ", "hires get 20 days."
    ])
    loop = AgentLoop(llm, {}, events=bus)
    out = loop.run("how many days?")
    assert "New hires get 20 days." in out
    reasoning = "".join(e.data["text"] for e in seen
                        if e.type == REASONING_DELTA)
    answer = "".join(e.data["text"] for e in seen if e.type == ANSWER_DELTA)
    assert reasoning == "check policy"
    assert answer == "New hires get 20 days."


def test_native_loop_does_not_stream_undecoded_tool_call_markup():
    from ovat.agent.events import ANSWER_DELTA
    bus = EventBus()
    seen = _recorder(bus)
    llm = _StreamingFakeLLM([
        "<tool_", "call><function=search_docs>",
        '{"query": "leave"}', "</function></tool_call>",
    ])
    loop = AgentLoop(llm, {}, events=bus)
    out = loop.run("how many days?")
    assert out.startswith("Error:")
    assert loop.last_trace["totals"]["undecoded_tool_call"] is True
    assert [e for e in seen if e.type == ANSWER_DELTA] == []

