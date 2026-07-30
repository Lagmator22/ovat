# tests/test_langchain.py
"""Tests for the LangChain react engine (agent.type: react).

Note to myself: I do NOT need OVMS for these. I inject a fake chat model that
replays scripted replies, so the real LangChain agent graph runs the full
tool-calling loop (model -> tool -> model -> answer) on any machine. The only
thing this cannot exercise is the live OVMS round trip, which is a live test on
the AI PC. If LangChain is not installed, the whole file skips cleanly.
"""
import pytest

pytest.importorskip("langchain")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ovat.agent.factory import build_agent, build_tools
from ovat.agent.langchain_agent import LangChainAgent, build_react_agent
from ovat.agent.loop import AgentLoop
from ovat.config.workflow import WorkflowConfig


class FakeToolModel(GenericFakeChatModel):
    """A fake chat model that supports tool binding.

    create_agent calls model.bind_tools(...). The stock fake raises
    NotImplementedError, so I override bind_tools to ignore the tools and just
    keep replaying my scripted messages. That is enough to drive the graph.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def _cfg(agent_type="react", system_prompt="be helpful", max_iterations=5):
    return WorkflowConfig(
        model={"name": "Qwen3-8B-int4-ov", "ovms_url": "http://localhost:8000/v3"},
        tools=[{"name": "search_docs"}, {"name": "transcribe"}],
        agent={"type": agent_type, "max_iterations": max_iterations,
               "system_prompt": system_prompt},
    )


def test_react_agent_runs_full_tool_loop():
    # Script: the model first asks for a tool, then gives a final answer.
    scripted = iter([
        AIMessage(content="",
                  tool_calls=[{"name": "search_docs",
                               "args": {"query": "q3"}, "id": "c1"}]),
        AIMessage(content="Based on the docs, here is the answer."),
    ])
    tools = build_tools(_cfg())                      # search_docs in stub mode
    agent = build_react_agent(_cfg(), tools, llm=FakeToolModel(messages=scripted))

    answer = agent.run("what do my docs say about q3?")
    # If the tool node had not executed, the graph would never reach this reply.
    assert answer == "Based on the docs, here is the answer."


def test_react_agent_caps_runaway_with_friendly_message():
    # When the graph blows past its recursion limit, langgraph raises
    # GraphRecursionError. My adapter must catch it and return the same wording
    # the native loop uses, instead of letting the raw error escape. I drive
    # that path directly with a graph that raises, so the test is about MY
    # handling, not langgraph's internal step counting.
    from langgraph.errors import GraphRecursionError

    class _RunawayGraph:
        def invoke(self, *args, **kwargs):
            raise GraphRecursionError("recursion limit reached")

    agent = LangChainAgent(_RunawayGraph(), tools={}, max_iterations=2,
                           system_prompt=None)
    assert "max of 2 steps" in agent.run("spin forever")


def test_build_react_agent_exposes_native_like_interface():
    agent = build_react_agent(_cfg(max_iterations=9), build_tools(_cfg()),
                              llm=FakeToolModel(messages=iter([AIMessage(content="hi")])))
    assert isinstance(agent, LangChainAgent)
    assert agent.max_iterations == 9
    assert set(agent.tools) == {"search_docs", "transcribe"}
    assert hasattr(agent, "run")


# deriving argument models from the co-located SCHEMA (one source of truth)

def test_args_model_derived_from_schema_matches_the_contract():
    from pydantic import ValidationError
    from ovat.agent.arg_models import args_model_from_schema
    from ovat.tools.search_docs import SCHEMA

    Args = args_model_from_schema("search_docs", SCHEMA)
    ok = Args(query="find my notes")
    assert ok.query == "find my notes"
    assert ok.top_k == 5                       # default comes FROM the schema
    with pytest.raises(ValidationError):       # required field enforced
        Args(top_k=3)


def test_a_brand_new_tool_needs_no_second_registry():
    # The old _ARGS_MODELS dict made every new tool crash on react until it
    # was registered twice. Now any tool with a SCHEMA just works.
    from ovat.agent.langchain_agent import _wrap_tools

    schema = {"type": "function", "function": {
        "name": "shout", "description": "upper-case some text",
        "parameters": {"type": "object",
                       "properties": {
                           "text": {"type": "string", "description": "words"},
                           "times": {"type": "integer", "default": 1},
                       },
                       "required": ["text"]}}}
    tools = {"shout": {"schema": schema,
                       "function": lambda text, times=1: text.upper() * times}}
    (wrapped,) = _wrap_tools(tools)
    assert wrapped.name == "shout"
    assert wrapped.invoke({"text": "hi", "times": 2}) == "HIHI"
    assert wrapped.invoke({"text": "hi"}) == "HI"      # schema default applied


# factory dispatch on agent.type

def test_factory_builds_native_loop_for_native_type():
    assert isinstance(build_agent(_cfg(agent_type="native")), AgentLoop)


def test_factory_builds_langchain_agent_for_react_type():
    # No llm injected here, so the factory builds a real ChatOpenAI. Construction
    # makes no network call, so this is safe with no server running.
    agent = build_agent(_cfg(agent_type="react"))
    assert isinstance(agent, LangChainAgent)
    assert set(agent.tools) == {"search_docs", "transcribe"}


def test_factory_rejects_unknown_agent_type():
    with pytest.raises(ValueError, match="Unknown agent type"):
        build_agent(_cfg(agent_type="banana"))


# Multi-turn memory: the native loop has it via Session, these must too

def test_the_react_engine_remembers_earlier_turns():
    """A spy graph counts what each run is given. Without memory the second
    run sees ONE message and the model cannot answer a follow-up, which
    silently broke the multi-turn promise on three of the four engines."""
    from ovat.agent.langchain_agent import LangChainAgent

    seen = []

    class SpyGraph:
        def invoke(self, state, config=None):
            seen.append(len(state["messages"]))
            # A real graph returns the FULL list, input plus what it added.
            return {"messages": [*state["messages"], _Reply("ok")]}

    class _Reply:
        def __init__(self, content):
            self.content = content

    agent = LangChainAgent(SpyGraph(), tools={}, max_iterations=5,
                           system_prompt=None)
    agent.run("first")
    agent.run("second")
    assert seen == [1, 3], (
        f"second run saw {seen[-1]} messages; it must carry the first "
        f"exchange forward")


def test_a_failed_run_does_not_poison_the_next_question():
    """A half-finished exchange must not become permanent history."""
    from langgraph.errors import GraphRecursionError
    from ovat.agent.langchain_agent import LangChainAgent

    seen = []

    class FlakyGraph:
        def __init__(self):
            self.calls = 0

        def invoke(self, state, config=None):
            seen.append(len(state["messages"]))
            self.calls += 1
            if self.calls == 1:
                raise GraphRecursionError("too deep")
            return {"messages": [*state["messages"], _R("ok")]}

    class _R:
        def __init__(self, content):
            self.content = content

    agent = LangChainAgent(FlakyGraph(), tools={}, max_iterations=3,
                           system_prompt=None)
    assert "max of 3 steps" in agent.run("first")
    agent.run("second")
    assert seen == [1, 1], "the failed turn leaked into the next question"
