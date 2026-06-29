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
