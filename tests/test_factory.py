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
    # mcp_stdio used to be this test's example of "unsupported" — it is a
    # real tool type now (see test_mcp_client.py), so a made-up type stands in.
    with pytest.raises(ValueError, match="Unsupported tool type"):
        build_tools(_cfg(tools=[{"name": "search_docs", "type": "telepathy"}]))
