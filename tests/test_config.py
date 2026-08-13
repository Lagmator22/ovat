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
    # Bounded, so a hung server is still an error rather than a frozen CLI --
    # but 20 minutes, not 2. Measured on a CPU-only box: a plain completion
    # returns in a second while a cold AGENT request (system prompt + tool
    # schemas + history, verbose small model) took 1056s, so the old 120s cut
    # off a server that was working and reported it as a timeout.
    assert cfg.model.request_timeout == 1200.0


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

    # Recursive: the use-case samples live one folder down (examples/rag/...)
    # so each can carry its own README and sample data. A non-recursive glob
    # silently skipped every one of them, which is the worst kind of test --
    # green while the files it claims to cover drift out of the schema.
    examples = sorted(glob.glob("examples/**/*.yml", recursive=True))
    assert examples, "no examples found to check"
    for path in examples:
        load_workflow(path)          # raises if the file has drifted


def test_every_config_the_docs_hand_to_ovat_chat_actually_has_rag():
    """`ovat chat` exits 1 on a config with no rag: section.

    Caught in review: the README's macOS path said `ovat chat workflow.yml`
    right after `ovat init`, and the starter file ships rag: commented out on
    purpose. So the one command macOS users are pointed at failed instantly
    with "This workflow has no rag: section" -- the same species of defect
    this whole change set exists to remove.

    Scans the real markdown, so the docs and the configs cannot drift apart
    silently again.

    The bare name `workflow.yml` is checked SEPARATELY and by generating the
    starter, never by loading the repo's own workflow.yml. This repo has a
    development workflow.yml at its root that DOES carry a rag: block, so
    resolving the name against the working tree made the first version of
    this test pass while the documented command was still broken.
    """
    import glob
    import re
    from pathlib import Path

    from ovat.cli.main import _STARTER_YAML
    from ovat.config.workflow import load_workflow

    docs = ["README.md"] + glob.glob("examples/**/README.md", recursive=True)
    pattern = re.compile(r"ovat chat\s+(\S+\.yml)")
    checked = 0
    for doc in docs:
        text = Path(doc).read_text(encoding="utf-8")
        for config_path in pattern.findall(text):
            if config_path == "workflow.yml":
                # What the reader HAS at this point is `ovat init` output, not
                # this repo's root file. Compare against the starter itself.
                assert "\nrag:" in _STARTER_YAML, (
                    f"{doc} runs `ovat chat workflow.yml`, which is the file "
                    f"`ovat init` writes, and that starter has rag: commented "
                    f"out -- the command exits 1. Point it at a config that "
                    f"has retrieval, e.g. examples/rag/workflow.yml")
            else:
                assert load_workflow(config_path).rag is not None, (
                    f"{doc} runs `ovat chat {config_path}`, but that config "
                    f"has no rag: section, so the command exits 1")
            checked += 1
    assert checked, "no `ovat chat <config>` commands found to verify"


def test_every_qwen35_config_names_the_parser_that_actually_decodes_it():
    """Qwen3.5 needs tool_parser: qwen3coder. MEASURED, not reasoned.

    These configs shipped `auto` on the theory that OVMS inspects the chat
    template and picks for itself, so naming a parser could only get in the
    way. A clean-room run on the AI PC (2026-08-03) disproved it:

      auto        -> OVMS selected NOTHING and returned the tool call as
                     plain text. finish_reason "stop", tool_calls 0, and the
                     answer was the raw markup:
                         <tool_call><function=search_docs>
                         <parameter=query>OVAT memory budget</parameter>
                         </function></tool_call>
                     OVMS logged nothing at all about parser selection.
      qwen3coder  -> finish_reason "tool_calls", the call decoded, the tool
                     ran, and the answer cited its source.

    So `auto` is not a safe default here, it is a silent failure: the agent
    answers fluently while never once calling a tool, which is the hardest
    kind of broken to notice. Every Qwen3.5 config must name qwen3coder.
    """
    import glob
    from ovat.cli.main import _STARTER_YAML
    from ovat.config.workflow import load_workflow

    checked = 0
    # Only the tracked examples. The repo-root workflow.yml is gitignored, so
    # it is absent from a fresh clone and cannot be part of the contract.
    for path in sorted(glob.glob("examples/**/*.yml", recursive=True)):
        cfg = load_workflow(path)
        if "qwen3.5" in cfg.model.name.lower():
            assert cfg.model.tool_parser == "qwen3coder", (
                f"{path} serves {cfg.model.name} with "
                f"tool_parser={cfg.model.tool_parser!r}. Use 'qwen3coder': "
                f"measured on live OVMS, anything else decodes no tool calls "
                f"at all while still answering, so the failure is silent.")
            checked += 1
    assert checked, "no Qwen3.5 config found to check"

    # The starter `ovat init` writes is the very first config a new user runs,
    # so it has to carry the right parser too -- it is not covered by the glob.
    assert "Qwen3.5" in _STARTER_YAML and "qwen3coder" in _STARTER_YAML, (
        "the starter config serves Qwen3.5 without naming qwen3coder")


def test_every_documented_use_case_has_a_runnable_example():
    """The README sends users to one folder per use case. A dead link there
    is the first thing a new user hits, so the folders are asserted to exist
    with both a config and instructions."""
    import os

    for use_case in ("rag", "react", "audio-multimodal"):
        folder = os.path.join("examples", use_case)
        assert os.path.isfile(os.path.join(folder, "workflow.yml")), \
            f"{folder} has no workflow.yml"
        assert os.path.isfile(os.path.join(folder, "README.md")), \
            f"{folder} has no README.md"


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


# One description of the backend, read by all four engines

def test_every_engine_reads_the_same_backend_description():
    """THE test that would have caught the temperature bug.

    Four engines each built their own connection to OVMS. Two set temperature
    and two did not, for a whole release, so the benchmark compared
    determinism against sampling and reported it as a framework difference.
    Four places that must agree, with nothing making them agree.
    """
    import pytest
    from ovat.config.workflow import WorkflowConfig
    from ovat.providers.backend import LLMBackend

    cfg = WorkflowConfig(model={
        "name": "some-model", "ovms_url": "http://host:9999/v3",
        "temperature": 0.33, "request_timeout": 77.0})
    expected = LLMBackend.from_config(cfg)

    from ovat.agent.factory import build_llm
    native = build_llm(cfg)
    assert (native.model, native.temperature) == (expected.model,
                                                  expected.temperature)

    from ovat.agent.langchain_agent import _build_chat_model
    lc = _build_chat_model(cfg)
    assert (lc.model_name, lc.temperature) == (expected.model,
                                               expected.temperature)

    pytest.importorskip("llama_index.core")
    from ovat.agent.llamaindex_agent import _build_llm
    li = _build_llm(cfg)
    assert (li.model, li.temperature, li.api_base) == (
        expected.model, expected.temperature, expected.url)

    pytest.importorskip("agents")
    from ovat.agent.openai_agents_agent import _build_model, _model_settings
    assert _model_settings(cfg).temperature == expected.temperature
    assert str(_build_model(cfg)._client.base_url).rstrip("/") == \
        expected.url.rstrip("/")


def test_no_engine_reads_config_model_directly_any_more():
    """A site that goes back to config.model is a site that can drift again.
    Every engine must go through LLMBackend."""
    import re
    from pathlib import Path

    for name in ("langchain_agent", "llamaindex_agent", "openai_agents_agent"):
        source = Path(f"ovat/agent/{name}.py").read_text(encoding="utf-8")
        code = re.sub(r"#.*", "", source)
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        assert "config.model." not in code, (
            f"{name} still reads config.model directly instead of LLMBackend")


def test_the_backend_description_cannot_be_mutated():
    """frozen=True: one engine must not be able to change the shared
    description out from under the other three."""
    import dataclasses
    import pytest
    from ovat.config.workflow import WorkflowConfig
    from ovat.providers.backend import LLMBackend

    backend = LLMBackend.from_config(WorkflowConfig(model={"name": "m"}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        backend.temperature = 0.9


def test_an_omitted_tool_parser_is_derived_from_the_model_not_hardcoded():
    """The default was hermes3, which is WRONG for the default model.

    Every shipped config names qwen3coder explicitly, so this was latent by
    convention only: the moment anyone writes their own config and leaves the
    field out, they got hermes3 against Qwen3.5 and tool calling silently
    stopped. Flagged on the AI PC regression watch-list.

    `auto` is not the answer either -- measured on live OVMS, it made OVMS
    select no parser at all. So the default is derived from the model name,
    which is information OVAT already has.
    """
    from ovat.config.workflow import WorkflowConfig

    q35 = WorkflowConfig(model={"name": "Qwen3.5-4B-int4-ov"})
    assert q35.model.tool_parser == "qwen3coder"

    q3 = WorkflowConfig(model={"name": "Qwen3-8B-int4-ov"})
    assert q3.model.tool_parser == "hermes3"

    # An explicit value always wins; deriving must never override the user.
    named = WorkflowConfig(model={"name": "Qwen3.5-4B-int4-ov",
                                  "tool_parser": "hermes3"})
    assert named.model.tool_parser == "hermes3"

    # An unknown family keeps the historical default rather than guessing.
    other = WorkflowConfig(model={"name": "some-new-model"})
    assert other.model.tool_parser == "hermes3"


def test_a_fractional_cache_size_is_rejected_with_a_reason():
    """OVMS's cache_size is uint64; half a gigabyte cannot be expressed.

    Typing this float meant `ovms_cache_size_gb: 1` stringified as "1.0",
    which OVMS's option parser refuses ("Argument '1.0' failed to parse"),
    so the server exited before writing a log line. Failing in config -- at
    the line the user actually wrote -- beats failing as "OVMS exited
    without becoming ready".
    """
    import pytest
    from pydantic import ValidationError

    from ovat.config.workflow import ModelConfig

    assert ModelConfig(name="m", ovms_cache_size_gb=8).ovms_cache_size_gb == 8
    # A whole number written as a YAML float is still a whole number.
    assert ModelConfig(name="m", ovms_cache_size_gb=8.0).ovms_cache_size_gb == 8

    with pytest.raises(ValidationError):
        ModelConfig(name="m", ovms_cache_size_gb=1.5)
    with pytest.raises(ValidationError):
        ModelConfig(name="m", ovms_cache_size_gb=0)


def test_every_engine_caps_the_reply_length():
    """The same drift as the temperature bug, with a worse failure mode.

    Nothing capped a generation on ANY engine: the provider defaulted
    max_tokens to None and omitted the key, and no config field existed to
    set one. A model that never emits a stop token then generates until the
    client gives up -- measured here as a 13-minute llamaindex request and,
    separately, a native-loop request past the bench's 600s cap.

    An unbounded generation is not something four engines should each decide,
    so the ceiling lives in LLMBackend beside temperature, and this asserts
    all four actually carry it.
    """
    import pytest
    from ovat.config.workflow import WorkflowConfig
    from ovat.providers.backend import LLMBackend

    cfg = WorkflowConfig(model={"name": "some-model", "max_tokens": 1234})
    expected = LLMBackend.from_config(cfg)
    assert expected.max_tokens == 1234

    from ovat.agent.factory import build_llm
    assert build_llm(cfg).max_tokens == 1234

    from ovat.agent.langchain_agent import _build_chat_model
    assert _build_chat_model(cfg).max_tokens == 1234

    pytest.importorskip("llama_index.core")
    from ovat.agent.llamaindex_agent import _build_llm
    assert _build_llm(cfg).max_tokens == 1234

    pytest.importorskip("agents")
    from ovat.agent.openai_agents_agent import _model_settings
    assert _model_settings(cfg).max_tokens == 1234


def test_a_reply_is_capped_by_default_not_only_when_asked():
    """The runaway happened on a config that set nothing, so the DEFAULT is
    the thing that has to be safe. None stays available as an opt-out."""
    import pytest
    from pydantic import ValidationError

    from ovat.config.workflow import ModelConfig

    assert ModelConfig(name="m").max_tokens == 4096
    assert ModelConfig(name="m", max_tokens=None).max_tokens is None
    with pytest.raises(ValidationError):
        ModelConfig(name="m", max_tokens=0)
