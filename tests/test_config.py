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
