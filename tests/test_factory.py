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


def test_args_model_from_schema_handles_null_parameters():
    from ovat.agent.arg_models import args_model_from_schema
    schema = {"function": {"name": "no_arg_tool", "parameters": None}}
    model = args_model_from_schema("no_arg_tool", schema)
    assert model is not None


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


def test_a_missing_embeddings_model_is_a_sentence_not_a_cpp_assertion(tmp_path):
    """`ovat run` tracebacked where `ovat doctor` politely warned.

    Found on the AI PC: with a rag: section pointing at an unexported embedder,
    build_rag handed the path straight to openvino_genai and the failure came
    back as a C++ assertion out of core.cpp --

        RuntimeError: Exception from src\\inference\\src\\cpp\\core.cpp:84:
        Check 'util::directory_exists(path)...'

    -- which the CLI rendered as a traceback. doctor already reported the same
    fact as an "Embeddings model ... not found" warn row, so the information
    existed; only `run` was rude about it. Errors are for users.
    """
    import pytest
    from ovat.agent.factory import build_rag
    from ovat.config.workflow import WorkflowConfig

    missing = str(tmp_path / "no-such-embedder")
    config = WorkflowConfig(
        model={"name": "m"},
        rag={"embeddings": {"provider": "genai", "model": missing, "dim": 384},
             "retriever": {"provider": "sqlite-vec",
                           "db_path": str(tmp_path / "x.db")}})

    with pytest.raises(RuntimeError) as caught:
        build_rag(config)
    message = str(caught.value)
    assert "not found" in message
    assert missing in message                 # names the path it looked at
    assert "optimum-cli" in message           # and how to fix it
    assert "core.cpp" not in message          # not the C++ assertion


# --- declarative tool config (Ravi's asks 3 and 4) --------------------------
#
# The model path used to come from OVAT_WHISPER_MODEL / OVAT_VLM_MODEL, set by
# `export` before the run. That made workflow.yml an incomplete description of
# the agent: hand the file to a colleague, a CI runner or a container and the
# run fails with nothing in the file to say what is missing.

def _cfg_with_tool(name, **fields):
    from ovat.config.workflow import WorkflowConfig
    return WorkflowConfig(**{
        "model": {"name": "m", "provider": "ovms"},
        "tools": [{"name": name, "type": "builtin", **fields}],
        "agent": {"type": "native"},
    })


def test_a_transcribe_model_in_the_yaml_reaches_the_tool(monkeypatch):
    """Back the binding out and the tool loads whatever the env says instead."""
    from ovat.agent.factory import build_tools
    from ovat.tools import transcribe as transcribe_tool

    seen = {}
    monkeypatch.setattr(
        transcribe_tool, "transcribe_impl",
        lambda path, language="en", **kw: seen.update(kw) or "ok")

    cfg = _cfg_with_tool("transcribe", model="models/my-stt", device="NPU")
    build_tools(cfg)["transcribe"]["function"]("a.wav")

    assert seen["model"] == "models/my-stt"
    assert seen["device"] == "NPU"


def test_a_vlm_model_in_the_yaml_reaches_the_tool(monkeypatch):
    from ovat.agent.factory import build_tools
    from ovat.tools import describe_image as describe_tool

    seen = {}
    monkeypatch.setattr(
        describe_tool, "describe_image_impl",
        lambda image_path, prompt="x", **kw: seen.update(kw) or "ok")

    cfg = _cfg_with_tool("describe_image", model="models/my-vlm", device="GPU")
    build_tools(cfg)["describe_image"]["function"]("a.png")

    assert seen["model"] == "models/my-vlm"
    assert seen["device"] == "GPU"


def test_the_config_wins_over_the_environment(monkeypatch):
    """The env var stays as a FALLBACK, not the interface. A workflow that
    names a model must not be overridden by whatever a shell happened to
    export -- that is the configuration drift this change exists to remove."""
    from ovat.tools.transcribe import _configured_model

    monkeypatch.setenv("OVAT_WHISPER_MODEL", "models/from-the-shell")
    assert _configured_model("models/from-the-yaml") == "models/from-the-yaml"
    assert _configured_model(None) == "models/from-the-shell"


def test_an_unconfigured_tool_still_works_exactly_as_before(monkeypatch):
    """Nothing that works today may stop working: no model, no env var, and
    the documented default is still what loads."""
    from ovat.tools.describe_image import DEFAULT_MODEL_DIR, _configured_model
    from ovat.tools.transcribe import DEFAULT_MODEL_DIR as STT_DEFAULT
    from ovat.tools.transcribe import _configured_model as stt_model

    monkeypatch.delenv("OVAT_WHISPER_MODEL", raising=False)
    monkeypatch.delenv("OVAT_VLM_MODEL", raising=False)
    assert stt_model(None) == STT_DEFAULT
    assert _configured_model(None) == DEFAULT_MODEL_DIR


def test_two_tools_with_different_models_do_not_share_a_pipeline(tmp_path,
                                                                 monkeypatch):
    """The cache is keyed by (model, device).

    It used to be one global slot, which was correct only while the model
    could come from nowhere but a single env var. With a per-tool model, the
    bench -- four engines in one process -- would have the second agent
    transcribe with the first agent's model and report nothing wrong.

    The two folders are real and empty: _load_pipeline now checks the path
    exists before handing it to openvino_genai, so that a wrong `model:` reads
    as one sentence instead of a C++ assertion. Only the check is real here --
    the pipeline behind it is still fake.
    """
    from ovat.tools import transcribe as transcribe_tool

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    built = []

    class FakeGenAI:
        @staticmethod
        def WhisperPipeline(model, device):
            built.append((model, device))
            return object()

    monkeypatch.setitem(__import__("sys").modules, "openvino_genai", FakeGenAI)
    monkeypatch.setattr(transcribe_tool, "_pipelines", {})

    first = transcribe_tool._load_pipeline(str(a), "CPU")
    second = transcribe_tool._load_pipeline(str(b), "CPU")
    again = transcribe_tool._load_pipeline(str(a), "CPU")

    assert first is not second, "two models shared one pipeline"
    assert again is first, "the same model rebuilt instead of being reused"
    assert built == [(str(a), "CPU"), (str(b), "CPU")]
