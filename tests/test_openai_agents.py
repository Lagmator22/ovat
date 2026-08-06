# tests/test_openai_agents.py
"""Tests for the OpenAI Agents SDK engine (agent.type: openai-agents).

Pointing this SDK at OVMS rather than api.openai.com is the whole job, and
each of the three steps fails as something else if missed, so each is asserted
here rather than trusted.
"""
import asyncio
import json

import pytest

pytest.importorskip("agents")

from ovat.agent.openai_agents_agent import (OpenAIAgentsAgent,
                                            build_openai_agents_agent,
                                            _wrap_tools)
from ovat.config.workflow import WorkflowConfig


def _config(**agent) -> WorkflowConfig:
    return WorkflowConfig(model={"name": "test-model"}, agent=agent or {})


def _tools():
    calls = []

    def add_note(text: str, tag: str = "general") -> str:
        calls.append((text, tag))
        return f"noted {text!r} under {tag}"

    tools = {"add_note": {
        "function": add_note,
        "schema": {"type": "function", "function": {
            "name": "add_note",
            "description": "Record a note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "the note"},
                    "tag": {"type": "string", "description": "a label",
                            "default": "general"},
                },
                "required": ["text"],
            },
        }},
    }}
    return tools, calls


def test_the_tool_schema_is_the_hand_written_one_not_an_inferred_copy():
    """The SDK's @function_tool decorator infers a schema from a Python
    signature. OVAT's tools already ship a SCHEMA that IS the contract, and
    inferring a second one is how the two drift apart."""
    tools, _ = _tools()
    wrapped = _wrap_tools(tools)
    assert len(wrapped) == 1
    tool = wrapped[0]

    assert tool.name == "add_note"
    assert "Record a note" in tool.description
    schema = tool.params_json_schema
    assert set(schema["properties"]) == {"text", "tag"}
    assert schema["required"] == ["text"]
    assert schema["properties"]["tag"]["default"] == "general"


def test_arguments_arrive_as_a_json_string_and_reach_the_real_callable():
    """The SDK hands arguments over as a JSON STRING, not a dict. Passing that
    straight through as **kwargs would raise on every single tool call."""
    tools, calls = _tools()
    tool = _wrap_tools(tools)[0]

    result = asyncio.run(tool.on_invoke_tool(
        None, json.dumps({"text": "hello", "tag": "demo"})))
    assert calls == [("hello", "demo")]
    assert "noted" in result


def test_a_broken_tool_becomes_a_readable_string_not_a_traceback():
    """The model produced the call and the model has to recover from it, so
    failures go back as text it can read rather than killing the run."""
    def explode(**kwargs):
        raise ValueError("disk on fire")

    tools = {"boom": {"function": explode, "schema": {
        "type": "function", "function": {
            "name": "boom", "description": "d",
            "parameters": {"type": "object", "properties": {}}}}}}
    tool = _wrap_tools(tools)[0]

    out = asyncio.run(tool.on_invoke_tool(None, "{}"))
    assert "Error running boom" in out and "disk on fire" in out


def test_malformed_json_arguments_are_reported_not_raised():
    tools, _ = _tools()
    tool = _wrap_tools(tools)[0]
    out = asyncio.run(tool.on_invoke_tool(None, "{not json"))
    assert "could not parse tool arguments" in out


def test_a_non_string_result_is_json_encoded():
    """search_docs returns a list of hits; the SDK needs a string back."""
    tools = {"hits": {"function": lambda: [{"text": "a", "source": "b.md"}],
                      "schema": {"type": "function", "function": {
                          "name": "hits", "description": "d",
                          "parameters": {"type": "object",
                                         "properties": {}}}}}}
    out = asyncio.run(_wrap_tools(tools)[0].on_invoke_tool(None, "{}"))
    assert json.loads(out) == [{"text": "a", "source": "b.md"}]


def test_the_model_is_bound_to_ovms_and_not_to_openai():
    """Without an explicit client the SDK reads OPENAI_API_KEY and talks to
    OpenAI, so a local workflow would quietly bill a cloud account."""
    from ovat.agent.openai_agents_agent import _build_model

    model = _build_model(_config())
    assert str(model._client.base_url).rstrip("/").endswith("/v3")


def test_it_uses_chat_completions_not_the_responses_api():
    """OVMS speaks the chat-completions dialect. The SDK's default Responses
    model is OpenAI-only and 404s on a path OVMS never claimed to serve."""
    from agents import OpenAIChatCompletionsModel
    from ovat.agent.openai_agents_agent import _build_model

    assert isinstance(_build_model(_config()), OpenAIChatCompletionsModel)


def test_building_the_agent_disables_tracing():
    """Tracing uploads run data to OpenAI's servers and needs a real key, so
    leaving it on turns a fully local private run into a network call, which
    is the opposite of the point of running OVMS on your own hardware."""
    import agents

    tools, _ = _tools()
    agents.set_tracing_disabled(False)          # make the assertion mean something
    build_openai_agents_agent(_config(), tools, model="fake-model")
    from agents import run as agents_run
    assert agents_run.get_default_agent_runner()  # sanity: SDK is live
    from agents.tracing import get_trace_provider
    assert get_trace_provider()._disabled is True


def test_hitting_the_step_cap_reads_like_the_native_loop():
    """Every engine must fail with the same sentence, or the cap looks like a
    different bug depending on which engine you happened to pick."""
    from agents.exceptions import MaxTurnsExceeded

    tools, _ = _tools()

    class Boom:
        pass

    agent = OpenAIAgentsAgent(Boom(), tools, max_iterations=3,
                              system_prompt=None)

    def fake_run(*args, **kwargs):
        raise MaxTurnsExceeded("too many")

    import agents
    original = agents.Runner.run
    agents.Runner.run = fake_run
    try:
        answer = agent.run("hi")
    finally:
        agents.Runner.run = original
    assert answer == ("Error: I reached my max of 3 steps without a final "
                      "answer.")


def test_running_inside_an_event_loop_is_refused():
    tools, _ = _tools()
    agent = OpenAIAgentsAgent(object(), tools, 5, None)

    async def scenario():
        with pytest.raises(RuntimeError, match="event loop"):
            agent.run("hi")

    asyncio.run(scenario())


# Multi-turn memory: the native loop has it via Session, this must too

def test_the_agents_engine_remembers_earlier_turns(monkeypatch):
    """The SDK is stateless per run. to_input_list() returns the full
    transcript, which is exactly what the next run should start from."""
    import agents
    from ovat.agent.openai_agents_agent import OpenAIAgentsAgent

    tools, _ = _tools()
    seen = []

    class Result:
        final_output = "an answer"

        def __init__(self, given):
            self._given = given

        def to_input_list(self):
            return [*self._given, {"role": "assistant", "content": "an answer"}]

    async def fake_run(agent, input_items, max_turns=None):
        seen.append(len(input_items))
        return Result(input_items)

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    agent = OpenAIAgentsAgent(object(), tools, 5, None)
    agent.run("first")
    agent.run("second")
    assert seen == [1, 3], f"second run saw {seen[-1]} input items"


def test_a_failed_agents_run_does_not_poison_the_next(monkeypatch):
    import agents
    from agents.exceptions import MaxTurnsExceeded
    from ovat.agent.openai_agents_agent import OpenAIAgentsAgent

    tools, _ = _tools()

    async def always_too_many(agent, input_items, max_turns=None):
        raise MaxTurnsExceeded("too many")

    monkeypatch.setattr(agents.Runner, "run", always_too_many)
    agent = OpenAIAgentsAgent(object(), tools, 3, None)
    assert "max of 3 steps" in agent.run("first")
    assert agent._input_items == []


def test_markdown_fenced_arguments_parse_on_the_openai_sdk_path_too():
    """The fence stripper reached the native loop but not this engine.

    Small models wrap tool arguments in ```json fences because that is what
    JSON looks like in their training data. loop.py strips them; the OpenAI
    Agents SDK path hands the raw string to json.loads, so the same model
    lost a turn to "could not parse tool arguments" on --openai-agents while
    working fine on --native.

    This is the only OTHER engine that parses arguments itself: LangChain and
    LlamaIndex both derive a pydantic args_schema and let the framework do it.
    """
    import asyncio

    pytest.importorskip("agents")
    from ovat.agent.openai_agents_agent import _wrap_tools

    called = {}

    def echo(city: str) -> str:
        called["city"] = city
        return f"weather in {city}"

    tools = {"get_weather": {
        "schema": {"type": "function", "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}},
        "function": echo,
    }}

    (wrapped,) = _wrap_tools(tools)
    fenced = '```json\n{"city": "Tokyo"}\n```'
    result = asyncio.run(wrapped.on_invoke_tool(None, fenced))

    assert called.get("city") == "Tokyo", f"fenced args were not parsed: {result}"
    assert "could not parse" not in str(result)
