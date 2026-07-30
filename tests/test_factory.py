# tests/test_factory.py
"""Tests for the component factory.

Note to myself: building the agent does NOT need a running OVMS, because
OVMSLLMProvider only constructs a client. So I can fully test that the config
turns into the right wired-up agent here on the Mac, no server, no GPU.
"""
import pytest

from ovat.agent.factory import build_agent, build_tools, build_llm
from ovat.agent.loop import AgentLoop
from ovat.config.workflow import WorkflowConfig
from ovat.providers.llm_ovms import OVMSLLMProvider


def _cfg(**overrides):
    base = {
        "model": {"name": "Qwen3-8B-int4-ov", "ovms_url": "http://localhost:8000/v3"},
        "tools": [{"name": "search_docs"}, {"name": "transcribe"}],
        "agent": {"type": "native", "max_iterations": 7,
                  "system_prompt": "be helpful"},
    }
    base.update(overrides)
    return WorkflowConfig(**base)


def test_build_llm_uses_config_url_and_model():
    llm = build_llm(_cfg())
    assert isinstance(llm, OVMSLLMProvider)
    assert llm.model == "Qwen3-8B-int4-ov"


def test_build_llm_wires_the_request_timeout_into_the_client():
    # The whole point of the config field: it must reach the actual HTTP
    # client, otherwise a hung OVMS still blocks for the SDK default (~600s).
    cfg = _cfg()
    cfg.model.request_timeout = 5.0
    llm = build_llm(cfg)
    assert llm.client.timeout == 5.0


def test_build_tools_wires_both_builtins():
    tools = build_tools(_cfg())
    assert set(tools) == {"search_docs", "transcribe"}
    # each tool must carry the two keys my loop needs
    for spec in tools.values():
        assert "schema" in spec and "function" in spec


def test_built_tool_function_actually_runs():
    tools = build_tools(_cfg())
    # search_docs in stub mode should run and echo the query back
    out = tools["search_docs"]["function"](query="hello")
    assert "[stub]" in out[0]["text"]


def test_build_agent_returns_wired_agentloop():
    agent = build_agent(_cfg())
    assert isinstance(agent, AgentLoop)
    assert agent.max_iterations == 7
    assert set(agent.tools) == {"search_docs", "transcribe"}
    # the system prompt should have landed as the first message
    assert agent.session.messages[0]["content"] == "be helpful"


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="Unknown builtin tool"):
        build_tools(_cfg(tools=[{"name": "does_not_exist"}]))


def test_unsupported_tool_type_is_rejected():
    # mcp_stdio used to be this test's example of "unsupported"; it is a
    # real tool type now (see test_mcp_client.py), so a made-up type stands in.
    with pytest.raises(ValueError, match="Unsupported tool type"):
        build_tools(_cfg(tools=[{"name": "search_docs", "type": "telepathy"}]))


# Engine dispatch: the factory is the one place that knows the engine names

def test_every_supported_engine_name_is_dispatched():
    """AGENT_TYPES feeds the error message, so a name listed there but not
    handled below would advertise an engine that then fails as 'unknown'."""
    from ovat.agent.factory import AGENT_TYPES, build_agent
    from ovat.config.workflow import WorkflowConfig

    for name in AGENT_TYPES:
        cfg = WorkflowConfig(model={"name": "m"}, agent={"type": name})
        try:
            build_agent(cfg, skip_rag=True)
        except RuntimeError as exc:
            # A missing optional framework is fine and must say how to fix it.
            assert "pip install" in str(exc), exc
        except ValueError as exc:                       # pragma: no cover
            raise AssertionError(f"{name} is listed but not dispatched: {exc}")


def test_an_unknown_engine_names_the_ones_that_exist():
    """The user mistyped. Telling them only that it is wrong is useless."""
    import pytest
    from ovat.agent.factory import AGENT_TYPES, build_agent
    from ovat.config.workflow import WorkflowConfig

    cfg = WorkflowConfig(model={"name": "m"}, agent={"type": "langchian"})
    with pytest.raises(ValueError) as excinfo:
        build_agent(cfg, skip_rag=True)
    for name in AGENT_TYPES:
        assert name in str(excinfo.value)


def test_the_new_engines_build_without_touching_the_network():
    """`ovat run --dry-run` has to work on a laptop with no OVMS."""
    import pytest
    from ovat.agent.factory import build_agent
    from ovat.config.workflow import WorkflowConfig

    for name, module in (("llamaindex", "llama_index.core"),
                         ("openai-agents", "agents")):
        pytest.importorskip(module)
        cfg = WorkflowConfig(model={"name": "m"}, agent={"type": name})
        agent = build_agent(cfg, skip_rag=True)
        assert hasattr(agent, "run"), f"{name} has no .run()"
        assert hasattr(agent, "tools"), f"{name} cannot report its tools"


# The factory must hand out the SOCKET, not one particular plug

def test_build_llm_is_annotated_with_the_abc_not_a_concrete_class():
    """The annotation is the tell. This read `-> OVMSLLMProvider` and
    hardcoded that one plug, so GenAILLMProvider honoured the contract and was
    unreachable. A factory whose signature names a concrete class has stopped
    using its own abstraction."""
    import inspect

    from ovat.agent.factory import build_llm
    from ovat.providers.base import LLMProvider

    assert inspect.signature(build_llm).return_annotation is LLMProvider


def test_the_model_provider_selects_the_backend():
    """The same knob EmbeddingsConfig and RetrieverConfig already had, which
    ModelConfig was missing."""
    from ovat.agent.factory import build_llm
    from ovat.config.workflow import WorkflowConfig
    from ovat.providers.llm_ovms import OVMSLLMProvider

    default = build_llm(WorkflowConfig(model={"name": "m"}))
    assert isinstance(default, OVMSLLMProvider), "ovms must remain the default"


def test_an_unknown_provider_names_the_ones_that_exist():
    import pytest
    from ovat.agent.factory import build_llm
    from ovat.config.workflow import WorkflowConfig

    cfg = WorkflowConfig(model={"name": "m", "provider": "ovmss"})
    with pytest.raises(ValueError) as excinfo:
        build_llm(cfg)
    assert "ovms" in str(excinfo.value) and "genai" in str(excinfo.value)


def test_genai_is_refused_on_the_framework_engines():
    """LangChain builds a ChatOpenAI, LlamaIndex an OpenAILike, the Agents SDK
    an AsyncOpenAI. None can consume an in-process object, so the combination
    cannot work and must say so rather than fail somewhere deeper."""
    import pytest
    from ovat.agent.factory import build_agent
    from ovat.config.workflow import WorkflowConfig

    for engine in ("react", "llamaindex", "openai-agents"):
        cfg = WorkflowConfig(model={"name": "m", "provider": "genai"},
                             agent={"type": engine})
        with pytest.raises(ValueError) as excinfo:
            build_agent(cfg, skip_rag=True)
        message = str(excinfo.value)
        assert "genai" in message and engine in message
        assert "native" in message, "the error must name the way out"


# MCP subprocesses need an owner, or they outlive the agent

def test_the_agent_owns_the_mcp_servers_it_spawned():
    """mcp_stdio tools each launch a subprocess with its own event-loop
    thread. Nothing used to own them, so they lived until the interpreter
    exited via an atexit hook. In the TUI, where every /engine switch rebuilds
    the agent, they accumulated."""
    from ovat.agent.factory import build_agent
    from ovat.config.workflow import WorkflowConfig

    agent = build_agent(WorkflowConfig(model={"name": "m"}), skip_rag=True)
    assert hasattr(agent, "mcp_servers")


def test_close_agent_closes_every_server():
    from ovat.agent.factory import close_agent

    closed = []

    class FakeServer:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    class FakeAgent:
        mcp_servers = [FakeServer("a"), FakeServer("b")]

    close_agent(FakeAgent())
    assert closed == ["a", "b"]


def test_one_wedged_server_does_not_stop_the_others_closing():
    """Shutting down must not raise: a single stuck subprocess should not
    leak the rest, nor take down the caller."""
    from ovat.agent.factory import close_agent

    closed = []

    class Wedged:
        def close(self):
            raise OSError("stuck")

    class Fine:
        def close(self):
            closed.append("fine")

    class FakeAgent:
        mcp_servers = [Wedged(), Fine()]

    close_agent(FakeAgent())          # must not raise
    assert closed == ["fine"]


def test_close_agent_on_an_agent_with_no_servers_is_a_no_op():
    from ovat.agent.factory import close_agent

    class Bare:
        pass

    close_agent(Bare())               # must not raise
