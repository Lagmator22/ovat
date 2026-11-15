# tests/test_ovms_live.py
"""Live OVMS integration tests (skipped unless an OVMS server is reachable).

Run against a live OVMS server::

    OVAT_OVMS_URL=http://localhost:8000/v3 \
    OVAT_OVMS_MODEL=Qwen3.5-4B-int4-ov \
    pytest tests/test_ovms_live.py -v
"""
import os
import urllib.request

import pytest

from ovat.agent.loop import AgentLoop
from ovat.providers.llm_ovms import OVMSLLMProvider

# Read server coordinates from the environment so host-specific settings stay unhardcoded.
OVMS_URL = os.environ.get("OVAT_OVMS_URL", "http://localhost:8000/v3")
OVMS_MODEL = os.environ.get("OVAT_OVMS_MODEL", "Qwen3.5-4B-int4-ov")


def _ovms_reachable() -> bool:
    """Try GET /models on the OVMS server; skip when unreachable or not OVMS."""
    try:
        url = f"{OVMS_URL.rstrip('/')}/models"
        with urllib.request.urlopen(url, timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


needs_ovms = pytest.mark.skipif(
    not _ovms_reachable(),
    reason=f"no OVMS server reachable at {OVMS_URL}",
)

# Every test in this file is a live test, so `pytest -m "not live"` skips the
# whole file fast and `pytest -m live` selects exactly these (on the AI PC).
pytestmark = pytest.mark.live


@needs_ovms
def test_ovms_plain_chat_answers():
    """Simplest proof of life: a question with no tools returns some text."""
    llm = OVMSLLMProvider(base_url=OVMS_URL, model=OVMS_MODEL)
    agent = AgentLoop(llm, tools={})
    out = agent.run("Say hello in exactly five words.")
    assert isinstance(out, str) and len(out) > 0


@needs_ovms
def test_ovms_agent_actually_calls_a_tool():
    """Verify the native loop triggers a real tool call against live OVMS."""
    seen_cities = []

    def get_weather(city: str) -> str:
        seen_cities.append(city)
        return f"It is 22 degrees and sunny in {city}."

    tools = {"get_weather": {
        "schema": {"type": "function", "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }},
        "function": get_weather,
    }}

    llm = OVMSLLMProvider(base_url=OVMS_URL, model=OVMS_MODEL)
    agent = AgentLoop(llm, tools=tools)
    out = agent.run("What is the weather in Tokyo right now? Use your tool.")

    assert seen_cities, "the model never called get_weather"
    assert isinstance(out, str) and len(out) > 0


# The LangChain (agent.type: react) path, same OVMS server, different engine.

@needs_ovms
def test_ovms_react_plain_chat_answers():
    """The LangChain engine should answer a plain question against real OVMS."""
    from ovat.agent.factory import build_agent
    from ovat.config.workflow import WorkflowConfig

    cfg = WorkflowConfig(
        model={"name": OVMS_MODEL, "ovms_url": OVMS_URL},
        agent={"type": "react"},
    )
    agent = build_agent(cfg)
    out = agent.run("Say hello in exactly five words.")
    assert isinstance(out, str) and len(out) > 0


@needs_ovms
def test_ovms_react_calls_a_tool_through_langchain():
    """Verify the LangChain react engine executes a tool call against live OVMS."""
    from ovat.agent.factory import build_agent
    from ovat.config.workflow import WorkflowConfig

    cfg = WorkflowConfig(
        model={"name": OVMS_MODEL, "ovms_url": OVMS_URL},
        tools=[{"name": "search_docs"}],
        agent={"type": "react",
               "system_prompt": "When asked about the user's files or notes, "
                                 "call the search_docs tool to answer."},
    )
    agent = build_agent(cfg)
    out = agent.run("Search my documents for the Q3 budget and summarise what you find.")
    assert isinstance(out, str) and len(out) > 0


# The LlamaIndex and openai-agents paths. Same server, same two shapes as the
# native and react tests above.
#
# WHY THESE ASSERT ON AN OBSERVED CALL, not on the answer text. Both engines
# were proven by hand once, in July, and nothing has gated a regression since.
# An answer that merely READS as though it searched is exactly the failure
# these are here to catch -- an engine that silently degrades to plain chat
# still returns fluent prose, and the react test above (which only checks the
# answer is non-empty) would not notice. So the tool's own Python function is
# wrapped and the call is recorded: it either ran in this process or it did
# not.


def _recording_tools(monkeypatch):
    """Wrap every built tool so a real invocation is observable.

    build_tools is looked up through the factory module's globals at call
    time, so replacing it there reaches whichever engine build_agent picks.
    """
    from ovat.agent import factory

    called = []
    real_build_tools = factory.build_tools

    def recording_build_tools(config, **kwargs):
        tools = real_build_tools(config, **kwargs)
        for name, spec in tools.items():
            original = spec["function"]

            def recorder(*args, _original=original, _name=name, **kw):
                called.append(_name)
                return _original(*args, **kw)

            spec["function"] = recorder
        return tools

    monkeypatch.setattr(factory, "build_tools", recording_build_tools)
    return called


def _tool_config(engine):
    from ovat.config.workflow import WorkflowConfig

    return WorkflowConfig(
        model={"name": OVMS_MODEL, "ovms_url": OVMS_URL},
        tools=[{"name": "search_docs"}],
        agent={"type": engine,
               "system_prompt": "When asked about the user's files or notes, "
                                "call the search_docs tool to answer."},
    )


@needs_ovms
def test_ovms_llamaindex_plain_chat_answers():
    """The LlamaIndex engine should answer a plain question against real OVMS."""
    from ovat.agent.factory import build_agent
    from ovat.config.workflow import WorkflowConfig

    cfg = WorkflowConfig(
        model={"name": OVMS_MODEL, "ovms_url": OVMS_URL},
        agent={"type": "llamaindex"},
    )
    out = build_agent(cfg).run("Say hello in exactly five words.")
    assert isinstance(out, str) and len(out) > 0


@needs_ovms
def test_ovms_llamaindex_calls_a_tool(monkeypatch):
    """FunctionAgent must actually execute search_docs, not describe it.

    is_function_calling_model=False would make LlamaIndex decide the model
    cannot call tools and quietly answer from general knowledge instead, which
    looks identical to a working run in the transcript.
    """
    from ovat.agent.factory import build_agent

    called = _recording_tools(monkeypatch)
    agent = build_agent(_tool_config("llamaindex"))
    out = agent.run("Search my documents for the Q3 budget and summarise "
                    "what you find.")

    assert "search_docs" in called, "the model never called search_docs"
    assert isinstance(out, str) and len(out) > 0


@needs_ovms
def test_ovms_openai_agents_plain_chat_answers():
    """The Agents SDK engine should answer a plain question against real OVMS."""
    from ovat.agent.factory import build_agent
    from ovat.config.workflow import WorkflowConfig

    cfg = WorkflowConfig(
        model={"name": OVMS_MODEL, "ovms_url": OVMS_URL},
        agent={"type": "openai-agents"},
    )
    out = build_agent(cfg).run("Say hello in exactly five words.")
    assert isinstance(out, str) and len(out) > 0


@needs_ovms
def test_ovms_openai_agents_calls_a_tool(monkeypatch):
    """The Agents SDK must run the tool through OVMS's chat-completions path.

    This engine needs OpenAIChatCompletionsModel rather than the SDK default
    Responses model; if that ever regresses the request does not reach OVMS in
    a shape it serves, and no tool runs.
    """
    from ovat.agent.factory import build_agent

    called = _recording_tools(monkeypatch)
    agent = build_agent(_tool_config("openai-agents"))
    out = agent.run("Search my documents for the Q3 budget and summarise "
                    "what you find.")

    assert "search_docs" in called, "the model never called search_docs"
    assert isinstance(out, str) and len(out) > 0
