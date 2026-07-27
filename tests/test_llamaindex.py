# tests/test_llamaindex.py
"""Tests for the LlamaIndex engine (agent.type: llamaindex).

The point of these is that the SAME tools dict the native loop runs also
drives a LlamaIndex FunctionAgent, and that the adapter presents the same
.run(text) -> text face as every other engine. A fake LLM stands in for OVMS
so the whole tool-calling loop runs on any machine with no server.
"""
import pytest

pytest.importorskip("llama_index.core")

from ovat.agent.llamaindex_agent import (LlamaIndexAgent,
                                         build_llamaindex_agent, _wrap_tools)
from ovat.config.workflow import WorkflowConfig


def _config(**agent) -> WorkflowConfig:
    return WorkflowConfig(model={"name": "test-model"}, agent=agent or {})


def _tools() -> dict:
    """One tool with the same shape a real OVAT tool has."""
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


def test_tools_are_wrapped_with_schemas_derived_from_the_contract():
    """The rule that must hold across every engine: the argument model comes
    from the tool's own SCHEMA. A hand-kept second registry is how a tool
    ends up working on one engine and crashing on another."""
    tools, _ = _tools()
    wrapped = _wrap_tools(tools)
    assert len(wrapped) == 1

    meta = wrapped[0].metadata
    assert meta.name == "add_note"
    assert "Record a note" in meta.description

    fields = meta.fn_schema.model_fields
    assert set(fields) == {"text", "tag"}
    assert fields["text"].is_required()
    # The default came from the SCHEMA, not from the Python signature.
    assert fields["tag"].default == "general"


def test_the_wrapped_tool_calls_the_very_same_callable():
    """Not a copy, not a re-implementation: the retriever the factory bound
    into search_docs is already inside that callable."""
    tools, calls = _tools()
    wrapped = _wrap_tools(tools)
    result = wrapped[0].call(text="hello", tag="demo")
    assert calls == [("hello", "demo")]
    assert "noted" in str(result)


def test_the_adapter_presents_the_same_face_as_the_native_loop():
    """The factory hands back one of several engines and the CLI calls .run()
    without knowing which. That only works if they all look alike."""
    tools, _ = _tools()

    class FakeAgent:
        async def run(self, message):
            return f"answered: {message}"

    agent = LlamaIndexAgent(FakeAgent(), tools, max_iterations=7,
                            system_prompt="be brief")
    assert agent.run("hi") == "answered: hi"
    assert agent.tools is tools          # for `ovat run --dry-run`
    assert agent.max_iterations == 7
    assert agent.system_prompt == "be brief"


def test_a_response_object_is_reduced_to_its_text():
    """FunctionAgent returns a response OBJECT, and its repr has gained extra
    detail between releases. Reading .response keeps the answer clean."""
    tools, _ = _tools()

    class Response:
        response = "  the answer  "

        def __str__(self):
            return "Response(blocks=[...], tool_calls=[...])"

    class FakeAgent:
        async def run(self, message):
            return Response()

    agent = LlamaIndexAgent(FakeAgent(), tools, 5, None)
    assert agent.run("q") == "the answer"


def test_running_inside_an_event_loop_is_refused_not_worked_around():
    """A running loop means we are being called from async code. Quietly
    spawning a second loop there is how you get a deadlock that only shows up
    on someone else's machine, so this raises a readable error instead."""
    import asyncio

    tools, _ = _tools()

    class FakeAgent:
        async def run(self, message):
            return "never reached"

    agent = LlamaIndexAgent(FakeAgent(), tools, 5, None)

    async def scenario():
        with pytest.raises(RuntimeError, match="event loop"):
            agent.run("hi")

    asyncio.run(scenario())


def test_build_uses_the_injected_llm_and_never_touches_the_network():
    """Construction must make no network call, so a --dry-run and every test
    works on a machine with no OVMS."""
    from llama_index.core.llms import MockLLM

    tools, _ = _tools()
    agent = build_llamaindex_agent(_config(max_iterations=4), tools,
                                   llm=MockLLM())
    assert isinstance(agent, LlamaIndexAgent)
    assert agent.max_iterations == 4
    assert list(agent.tools) == ["add_note"]


def test_the_llm_is_told_that_ovms_can_chat_and_call_functions():
    """Two flags on OpenAILike that are easy to miss and fail as something
    else: without is_chat_model LlamaIndex calls a completions endpoint OVMS
    does not serve, and without is_function_calling_model FunctionAgent
    decides the model cannot call tools and silently degrades to plain chat,
    which looks like the tools were never configured."""
    from ovat.agent.llamaindex_agent import _build_llm

    llm = _build_llm(_config())
    assert llm.is_chat_model is True
    assert llm.is_function_calling_model is True
    assert llm.api_base.endswith("/v3")
