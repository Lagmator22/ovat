# tests/test_config.py
"""Tests for the workflow config layer.

Note to myself: I test two things. First, that good YAML loads and the
defaults fill in correctly. Second, that BAD config is rejected loudly by
pydantic, because catching a bad config early is the whole point of validation.
"""
import pytest
from pydantic import ValidationError

from ovat.config.workflow import (
    WorkflowConfig,
    ModelConfig,
    load_workflow,
)


def test_minimal_config_fills_defaults():
    # Only the model name is required; everything else should default.
    cfg = WorkflowConfig(model={"name": "Qwen3-8B-int4-ov"})
    assert cfg.model.device == "CPU"
    assert cfg.model.tool_parser == "hermes3"
    assert cfg.agent.type == "native"
    assert cfg.agent.max_iterations == 10
    assert cfg.tools == []
    assert cfg.model.request_timeout == 120.0    # bounded, never the SDK's ~600s


def test_request_timeout_is_configurable():
    cfg = WorkflowConfig(model={"name": "m", "request_timeout": 30})
    assert cfg.model.request_timeout == 30.0


def test_typoed_key_is_rejected_not_ignored():
    # extra="forbid": a misspelled field must be a loud error, never a silent
    # no-op that leaves the default quietly in charge.
    with pytest.raises(ValidationError, match="max_iteration"):
        WorkflowConfig(model={"name": "m"}, agent={"max_iteration": 3})


def test_unknown_top_level_section_is_rejected():
    with pytest.raises(ValidationError, match="modle"):
        WorkflowConfig(model={"name": "m"}, modle={"name": "typo"})


def test_full_config_parses_all_fields():
    cfg = WorkflowConfig(
        model={"name": "m", "device": "GPU", "reasoning_parser": "qwen3"},
        tools=[{"name": "search_docs"}, {"name": "transcribe"}],
        agent={"type": "native", "max_iterations": 5},
    )
    assert cfg.model.device == "GPU"
    assert cfg.model.reasoning_parser == "qwen3"
    assert [t.name for t in cfg.tools] == ["search_docs", "transcribe"]
    assert cfg.agent.max_iterations == 5


def test_serve_model_location_fields():
    # source_model and model_repository_path are for `ovat serve`. They default
    # sensibly and parse correctly when given.
    cfg = WorkflowConfig(model={"name": "m"})
    assert cfg.model.source_model is None
    assert cfg.model.model_repository_path == "models"

    cfg2 = WorkflowConfig(model={
        "name": "Qwen3-8B-int4-ov",
        "source_model": "OpenVINO/Qwen3-8B-int4-ov",
        "model_repository_path": "/abs/models",
    })
    assert cfg2.model.source_model == "OpenVINO/Qwen3-8B-int4-ov"
    assert cfg2.model.model_repository_path == "/abs/models"


def test_missing_required_model_is_rejected():
    # No model at all should fail validation, not silently pass.
    with pytest.raises(ValidationError):
        WorkflowConfig()


def test_wrong_type_is_rejected():
    # max_iterations must be an int; a non-numeric string must fail.
    with pytest.raises(ValidationError):
        WorkflowConfig(model={"name": "m"}, agent={"max_iterations": "lots"})


def test_load_workflow_reads_the_example_file(tmp_path):
    # I write a small YAML to a temp file and load it end to end.
    yml = tmp_path / "wf.yml"
    yml.write_text(
        "model:\n"
        "  name: Qwen3-8B-int4-ov\n"
        "  device: GPU\n"
        "tools:\n"
        "  - name: search_docs\n"
        "agent:\n"
        "  max_iterations: 7\n"
    )
    cfg = load_workflow(str(yml))
    assert isinstance(cfg, WorkflowConfig)
    assert cfg.model.name == "Qwen3-8B-int4-ov"
    assert cfg.model.device == "GPU"
    assert cfg.tools[0].name == "search_docs"
    assert cfg.agent.max_iterations == 7


# The shipped examples must stay loadable, on every engine they advertise

def test_every_shipped_example_parses():
    """An example that no longer matches the schema is worse than no example:
    it is the first thing a new user runs, and strict validation means a
    stale key is a hard error, not a warning."""
    import glob
    from ovat.config.workflow import load_workflow

    examples = sorted(glob.glob("examples/*.yml"))
    assert examples, "no examples found to check"
    for path in examples:
        load_workflow(path)          # raises if the file has drifted


def test_the_document_qa_sample_has_what_it_claims():
    """The sample agent is the proposal's Document Q&A deliverable, so the
    parts that make it that (retrieval, citations, a grounding prompt) are
    the parts worth asserting."""
    from ovat.config.workflow import load_workflow

    cfg = load_workflow("examples/document-qa.yml")
    assert cfg.rag is not None, "a Q&A sample with no retrieval is not one"
    assert [t.name for t in cfg.tools] == ["search_docs", "transcribe"]
    prompt = (cfg.agent.system_prompt or "").lower()
    # Grounding is the whole point: cite, and admit when the docs do not say.
    assert "cite" in prompt
    assert "search_docs" in prompt


def test_the_document_qa_sample_builds_on_every_engine():
    """The claim the sample makes is "change one line and it runs on another
    framework". If that is not true, the file is advertising a lie."""
    import pytest
    from ovat.agent.factory import AGENT_TYPES, build_agent
    from ovat.config.workflow import load_workflow

    for engine in AGENT_TYPES:
        cfg = load_workflow("examples/document-qa.yml")
        cfg.agent.type = engine
        try:
            agent = build_agent(cfg, skip_rag=True)
        except RuntimeError as exc:
            pytest.skip(f"{engine} not installed: {exc}")
        assert list(agent.tools) == ["search_docs", "transcribe"]


# Sampling must be identical across engines, or the benchmark is meaningless

def test_temperature_is_a_config_field_with_a_deterministic_default():
    """It used to be hardcoded 0 in two engines and absent in the other two,
    so react and llamaindex returned byte-identical answers across repeated
    runs while native and openai-agents varied. Their tight run-to-run spread
    measured determinism, not stability, and the latency column was comparing
    the wrong thing."""
    from ovat.config.workflow import WorkflowConfig

    assert WorkflowConfig(model={"name": "m"}).model.temperature == 0.0
    assert WorkflowConfig(
        model={"name": "m", "temperature": 0.7}).model.temperature == 0.7


def test_every_engine_reads_the_configured_temperature():
    """One config value, four engines. If any of them ignores it, a
    cross-engine comparison is not like-for-like."""
    import pytest
    from ovat.agent.factory import build_agent
    from ovat.config.workflow import WorkflowConfig

    cfg = WorkflowConfig(model={"name": "m", "temperature": 0.42})

    native = build_agent(cfg, skip_rag=True)
    assert native.llm.temperature == 0.42

    from ovat.agent.langchain_agent import _build_chat_model
    assert _build_chat_model(cfg).temperature == 0.42

    pytest.importorskip("llama_index.core")
    from ovat.agent.llamaindex_agent import _build_llm
    assert _build_llm(cfg).temperature == 0.42

    pytest.importorskip("agents")
    from ovat.agent.openai_agents_agent import _model_settings
    assert _model_settings(cfg).temperature == 0.42


def test_no_engine_hardcodes_a_temperature():
    """A literal here is how the four drifted apart in the first place."""
    import re
    from pathlib import Path

    for name in ("langchain_agent", "llamaindex_agent",
                 "openai_agents_agent"):
        source = Path(f"ovat/agent/{name}.py").read_text(encoding="utf-8")
        code = re.sub(r"#.*", "", source)
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        for match in re.finditer(r"temperature\s*=\s*([^,)\s]+)", code):
            assert not match.group(1).replace(".", "").isdigit(), (
                f"{name} hardcodes temperature={match.group(1)}")
