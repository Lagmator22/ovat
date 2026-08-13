# tests/test_cli.py
"""Tests for the ovat command line.

Note to myself: typer ships a CliRunner that invokes my commands in-process and
captures their output and exit code, so I can test the CLI without a real shell
or a running OVMS server. I test the wiring (init writes a usable config,
dry-run builds the agent) and the error paths (bad config exits non-zero).
"""
import pytest
from typer.testing import CliRunner

from ovat.cli.main import app
from ovat.config.workflow import load_workflow

runner = CliRunner()


def test_init_writes_a_loadable_config(tmp_path):
    target = tmp_path / "workflow.yml"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0
    assert target.exists()
    # the file it wrote must itself be a valid workflow
    cfg = load_workflow(str(target))
    # Not a hardcoded model name: that only pinned whatever was current and
    # broke on a deliberate change. What must hold is that the two model
    # fields AGREE. `name` is what OVMS serves, `source_model` is what it
    # pulls; if they disagree, serve downloads one model and then fails to
    # find the other, which reads as "the starter config is broken".
    assert cfg.model.source_model is not None
    assert cfg.model.source_model.endswith("/" + cfg.model.name)


def test_init_picks_the_device_this_machine_actually_has(tmp_path):
    # DeviceManager decides the starter's LLM device: CPU here on the Mac,
    # GPU on an AI PC. The starter file must run where it was created.
    from ovat.core.device_manager import DeviceManager
    expected = DeviceManager().get_llm_device()
    target = tmp_path / "workflow.yml"
    runner.invoke(app, ["init", str(target)])
    assert load_workflow(str(target)).model.device == expected


def test_init_refuses_to_overwrite(tmp_path):
    target = tmp_path / "workflow.yml"
    target.write_text("model:\n  name: x\n")
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 1


def test_run_dry_run_builds_agent_without_a_server():
    result = runner.invoke(
        app, ["run", "examples/workflow.yml", "--input", "hello", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "search_docs" in result.output
    assert "dry-run" in result.output


def test_run_with_missing_config_fails():
    result = runner.invoke(app, ["run", "/no/such/file.yml", "-i", "hi"])
    assert result.exit_code != 0


def test_run_prints_an_answer_containing_markup(monkeypatch):
    """ANY model must be printable. [/INST] used to crash run with MarkupError.

    Llama- and Mistral-family models emit [/INST] routinely, and [1] shows up
    in any citation-style answer. The answer is data; the console must not
    read it as a format string.
    """
    from ovat.cli import main as cli_main

    class MarkupAgent:
        tools, max_iterations, last_trace = {}, 5, {}

        def run(self, text):
            return "Done.[/INST] Use [bold] for [1]."

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: MarkupAgent())
    result = runner.invoke(app, ["run", "examples/workflow.yml", "-i", "hi"])
    assert result.exception is None          # no MarkupError traceback
    assert result.exit_code == 0
    assert "[/INST]" in result.output        # printed literally...
    assert "[bold]" in result.output         # ...not swallowed as a style tag


def test_chat_prints_an_answer_and_sources_containing_markup(monkeypatch):
    """Same contract on the local (no-OVMS) path, sources included."""
    from ovat.agent import factory, rag_chat as rag_chat_mod
    from ovat.cli import main as cli_main
    from ovat.providers import llm_genai

    class FakeRetriever:
        def retrieve(self, query, top_k=5):
            return []

        def close(self):
            pass

    monkeypatch.setattr(cli_main, "resolve_chat_model", lambda p: "fake-dir")
    monkeypatch.setattr(factory, "build_rag", lambda cfg: FakeRetriever())
    monkeypatch.setattr(llm_genai, "GenAILLMProvider", lambda *a, **k: object())
    monkeypatch.setattr(rag_chat_mod, "rag_chat",
                        lambda *a, **k: ("Answer[/INST]", ["notes[1].md"]))

    result = runner.invoke(app, ["chat", "examples/workflow.yml", "-i", "hi"])
    assert result.exception is None
    assert "Answer[/INST]" in result.output
    assert "notes[1].md" in result.output     # a source path with brackets


def test_doctor_survives_markup_in_a_config(tmp_path):
    """Config values reach doctor's table, so they are data there too."""
    cfg = tmp_path / "w.yml"
    cfg.write_text("model:\n  name: 'Q[/b]8B'\n")
    result = runner.invoke(app, ["doctor", str(cfg)])
    assert result.exception is None          # a bracketed model name used to crash
    assert "Q[/b]8B" in result.output


def test_run_engine_flag_overrides_the_config(monkeypatch):
    """`--engine` swaps the framework without editing YAML.

    Testing one framework used to mean opening the workflow file, changing
    agent.type, running, and changing it back -- painful on a remote AI PC
    over SSH, which is exactly where the framework engines have to be tested
    because they need a live OVMS.
    """
    from ovat.cli import main as cli_main

    seen = {}

    class StubAgent:
        tools, max_iterations, last_trace = {}, 5, {}

        def run(self, text):
            return "ok"

    def fake_build(cfg, skip_rag=False):
        seen["type"] = cfg.agent.type
        return StubAgent()

    monkeypatch.setattr(cli_main, "build_agent", fake_build)
    result = runner.invoke(app, ["run", "examples/react/workflow.yml",
                                 "-i", "hi", "--engine", "llamaindex"])
    assert result.exit_code == 0
    assert seen["type"] == "llamaindex"        # the override reached the factory
    # The banner reads the CONFIG, never a literal, so an override that did not
    # write back to cfg would print the old engine and quietly mislead a demo.
    assert "LlamaIndex" in result.output


def test_run_without_engine_flag_keeps_the_config_value(monkeypatch):
    """Omitting --engine must not disturb what the workflow already says."""
    from ovat.cli import main as cli_main

    seen = {}

    class StubAgent:
        tools, max_iterations, last_trace = {}, 5, {}

        def run(self, text):
            return "ok"

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: seen.update(
                            type=cfg.agent.type) or StubAgent())
    runner.invoke(app, ["run", "examples/react/workflow.yml", "-i", "hi"])
    assert seen["type"] == "react"             # examples/react says react


def _engine_used(monkeypatch, *args):
    """Run `ovat run` with ARGS and report which engine reached the factory."""
    from ovat.cli import main as cli_main

    seen = {}

    class StubAgent:
        tools, max_iterations, last_trace = {}, 5, {}

        def run(self, text):
            return "ok"

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: seen.update(
                            type=cfg.agent.type) or StubAgent())
    result = runner.invoke(app, ["run", "examples/react/workflow.yml",
                                 "-i", "hi", *args])
    return seen.get("type"), result


def test_bare_engine_flags_select_their_engine(monkeypatch):
    """`--llamaindex` reads better than `--engine llamaindex` in a demo, and
    typing it is the natural first guess."""
    for flag, expected in (("--native", "native"),
                           ("--react", "react"),
                           ("--llamaindex", "llamaindex"),
                           ("--openai-agents", "openai-agents")):
        used, result = _engine_used(monkeypatch, flag)
        assert used == expected, f"{flag} selected {used}"
        assert result.exit_code == 0


def test_library_names_work_as_aliases(monkeypatch):
    """The engine is called `react` but the library is LangChain, and
    `openai-agents` is the OpenAI SDK. Both names are what people actually
    reach for, so both select the same engine rather than erroring."""
    assert _engine_used(monkeypatch, "--langchain")[0] == "react"
    assert _engine_used(monkeypatch, "--openai-sdk")[0] == "openai-agents"


def test_two_different_engines_is_an_error_not_a_coin_flip(monkeypatch):
    """Silently honouring one of them would make a benchmark lie about which
    framework produced the numbers."""
    used, result = _engine_used(monkeypatch, "--react", "--llamaindex")
    assert result.exit_code != 0
    assert used is None                       # nothing was built
    assert "react" in result.output and "llamaindex" in result.output


def test_the_same_engine_twice_is_fine(monkeypatch):
    """--react --langchain name ONE engine. Rejecting that would be pedantry."""
    used, result = _engine_used(monkeypatch, "--react", "--langchain")
    assert result.exit_code == 0
    assert used == "react"


def test_engine_flag_and_a_bare_flag_must_agree(monkeypatch):
    """Mixing the two spellings is fine when they mean the same engine."""
    assert _engine_used(monkeypatch, "--engine", "react", "--langchain")[0] == "react"
    assert _engine_used(monkeypatch, "--engine", "react", "--llamaindex")[1].exit_code != 0


def test_run_rejects_an_unknown_engine_by_name():
    """A typo must name the valid options, not fail inside the factory."""
    result = runner.invoke(app, ["run", "examples/react/workflow.yml",
                                 "-i", "hi", "--engine", "langchain"])
    assert result.exit_code != 0
    assert "langchain" in result.output        # says what was wrong
    assert "react" in result.output            # and what is valid


def test_run_does_not_leak_a_dangling_reasoning_tag(monkeypatch):
    """Qwen3 emits a closing </think> with no opening tag.

    Its chat template pre-fills the opening <think> itself, so the model only
    ever produces the terminator. The TUI folds reasoning away and never shows
    it; `ovat run` printed it verbatim, so every CLI answer on the AI PC ended
    up carrying a stray </think>.

    Everything BEFORE that terminator is reasoning, so it goes with it.
    """
    from ovat.cli import main as cli_main

    class ThinkingAgent:
        tools, max_iterations, last_trace = {}, 5, {}

        def run(self, text):
            return ("The user wants the capital of France. It is Paris.\n"
                    "</think>\n\nParis.")

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: ThinkingAgent())
    result = runner.invoke(app, ["run", "examples/react/workflow.yml", "-i", "hi"])
    assert result.exit_code == 0
    assert "</think>" not in result.output, "the reasoning terminator leaked"
    assert "Paris." in result.output              # the answer survives
    assert "The user wants" not in result.output  # the reasoning does not


def test_run_still_prints_an_answer_that_has_no_reasoning(monkeypatch):
    """Stripping must not eat an ordinary answer from a non-reasoning model."""
    from ovat.cli import main as cli_main

    class PlainAgent:
        tools, max_iterations, last_trace = {}, 5, {}

        def run(self, text):
            return "Paris."

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: PlainAgent())
    result = runner.invoke(app, ["run", "examples/react/workflow.yml", "-i", "hi"])
    assert "Paris." in result.output


def test_run_trace_writes_the_json_report(tmp_path, monkeypatch):
    # No OVMS here: swap the whole agent for a stub whose run() succeeds and
    # whose last_trace looks like a real native-loop trace. This tests the
    # CLI's half of the contract: gather, enrich (model, peak RSS), write.
    import json

    from ovat.cli import main as cli_main

    class StubAgent:
        tools = {"search_docs": {}}
        max_iterations = 5
        last_trace = {"engine": "native",
                      "turns": [{"latency_s": 0.1, "finish_reason": "stop",
                                 "prompt_tokens": 10, "completion_tokens": 5,
                                 "tool_calls": []}],
                      "totals": {"turns": 1, "latency_s": 0.1,
                                 "prompt_tokens": 10, "completion_tokens": 5,
                                 "tool_calls": 0}}

        def run(self, text):
            return "stubbed answer"

    monkeypatch.setattr(cli_main, "build_agent", lambda cfg, skip_rag=False: StubAgent())
    trace_path = tmp_path / "trace.json"
    result = runner.invoke(app, ["run", "examples/workflow.yml", "-i", "hi",
                                 "--trace", str(trace_path)])
    assert result.exit_code == 0
    # _write_trace writes UTF-8, so read UTF-8 rather than trusting the
    # platform default (cp1252 on Windows), which would break the moment a
    # model name or a tool name carried a non-ASCII character.
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    # Compared against the config rather than a literal. What this line is
    # really testing is that the trace is ENRICHED from the workflow at all --
    # the model name is not something the agent reports, it is read from the
    # config -- and pinning the current model name only made the test fail
    # whenever the shipped example legitimately changed models.
    assert data["model"] == load_workflow("examples/workflow.yml").model.name
    assert data["totals"]["prompt_tokens"] == 10    # the loop's numbers
    # psutil is a declared dependency, so a None here means the environment is
    # incomplete, not that the trace is wrong. Say so, instead of letting the
    # comparison below fail with a bare TypeError that names nothing.
    assert data["peak_rss_mb"] is not None, \
        "psutil missing from this env; reinstall with: pip install -e '.[dev]'"
    assert data["peak_rss_mb"] > 0                  # psutil measured something


def test_chat_max_tokens_zero_means_no_cap(monkeypatch):
    """0 is the documented way to ask for an uncapped answer."""
    from ovat.agent import factory, rag_chat as rag_chat_mod
    from ovat.cli import main as cli_main
    from ovat.providers import llm_genai

    built = {}

    class FakeRetriever:
        def retrieve(self, query, top_k=5):
            return []

        def close(self):
            pass

    monkeypatch.setattr(cli_main, "resolve_chat_model", lambda p: "fake-dir")
    monkeypatch.setattr(factory, "build_rag", lambda cfg: FakeRetriever())
    monkeypatch.setattr(rag_chat_mod, "rag_chat", lambda *a, **k: ("ok", []))
    monkeypatch.setattr(
        llm_genai, "GenAILLMProvider",
        lambda path, device=None, max_new_tokens=None:
            built.update(max_new_tokens=max_new_tokens) or object())

    runner.invoke(app, ["chat", "examples/workflow.yml", "-i", "hi",
                        "--max-tokens", "0"])
    assert built["max_new_tokens"] is None            # uncapped

    runner.invoke(app, ["chat", "examples/workflow.yml", "-i", "hi",
                        "--max-tokens", "900"])
    assert built["max_new_tokens"] == 900


def test_chat_rejects_a_negative_token_cap():
    result = runner.invoke(app, ["chat", "examples/workflow.yml", "-i", "hi",
                                 "--max-tokens", "-5"])
    assert result.exit_code == 1
    assert "cannot be negative" in result.output


# `ovat index` must show progress, because embedding looks like a hang

def _rag_config(tmp_path) -> str:
    """A workflow with a rag: section, pointing nowhere real.

    build_rag is stubbed in these tests, so the embedding model never has to
    exist: what is under test is the CLI's progress wiring, not OpenVINO.
    """
    config = tmp_path / "w.yml"
    config.write_text(
        "model:\n  name: m\n"
        "rag:\n"
        "  embeddings:\n    provider: genai\n    model: nope\n"
        "    device: CPU\n    dim: 384\n"
        "  retriever:\n    provider: sqlite-vec\n    db_path: ':memory:'\n"
        "  chunk:\n    size: 512\n    overlap: 0\n", encoding="utf-8")
    return str(config)


class _CountingRetriever:
    def __init__(self):
        self.added = 0

    def add(self, texts, sources=None):
        self.added += len(texts)

    def close(self):
        pass


def test_index_reports_progress_for_every_file(monkeypatch, tmp_path):
    """The regression this guards: wrapping the call in a Progress context is
    exactly the sort of change that swallows an exception or forgets to pass
    the callback, leaving the bar frozen at zero while work happens."""
    from ovat.cli import main as cli_main

    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (docs / name).write_text("some text", encoding="utf-8")

    retriever = _CountingRetriever()
    monkeypatch.setattr(cli_main, "build_agent", lambda *a, **k: None)
    monkeypatch.setattr("ovat.agent.factory.build_rag",
                        lambda cfg: retriever)

    ticks = []
    import ovat.rag.indexer as indexer
    real_index_folder = indexer.index_folder

    def watching(folder, retr, size=512, overlap=64, on_progress=None):
        assert on_progress is not None, (
            "the CLI built a progress bar but never passed the callback, so "
            "it would sit at zero for the whole run")
        return real_index_folder(
            folder, retr, size=size, overlap=overlap,
            on_progress=lambda d, t, p: (ticks.append((d, t)),
                                         on_progress(d, t, p))[1])

    monkeypatch.setattr(indexer, "index_folder", watching)

    result = runner.invoke(app, ["index", str(docs), _rag_config(tmp_path)])
    assert result.exit_code == 0, result.output
    assert ticks == [(1, 3), (2, 3), (3, 3)]
    assert "Indexed 3 chunks from 3 files" in result.output


def test_index_still_reports_a_missing_folder_rather_than_a_bar(monkeypatch,
                                                                tmp_path):
    """The file count is taken BEFORE the bar starts, so a missing folder now
    raises inside the try block. It must still be the friendly message."""
    from ovat.cli import main as cli_main

    monkeypatch.setattr(cli_main, "build_agent", lambda *a, **k: None)
    monkeypatch.setattr("ovat.agent.factory.build_rag",
                        lambda cfg: _CountingRetriever())

    result = runner.invoke(
        app, ["index", str(tmp_path / "nope"), _rag_config(tmp_path)])
    assert result.exit_code == 1
    assert "No such folder to index" in result.output


# Telemetry: a measurement that misreports itself is worse than none

def test_the_trace_names_the_engine_that_actually_ran(tmp_path):
    """The engine name used to be the literal "react", which was true while
    react was the only framework engine and became a lie the moment there
    were three: a llamaindex run wrote a trace claiming it was react."""
    import json
    from ovat.cli.main import _write_trace
    from ovat.config.workflow import WorkflowConfig

    class FrameworkAgent:          # no last_trace, like every adapter
        pass

    for engine in ("react", "llamaindex", "openai-agents"):
        cfg = WorkflowConfig(model={"name": "m"}, agent={"type": engine})
        path = tmp_path / f"{engine}.json"
        _write_trace(str(path), cfg, FrameworkAgent(), peak_rss_mb=1.0)
        assert json.loads(path.read_text(encoding="utf-8"))["engine"] == engine


def test_the_native_trace_keeps_its_own_engine_name(tmp_path):
    """The loop fills last_trace itself; the config must not overwrite it."""
    import json
    from ovat.cli.main import _write_trace
    from ovat.config.workflow import WorkflowConfig

    class NativeAgent:
        last_trace = {"engine": "native", "turns": [], "totals": {}}

    cfg = WorkflowConfig(model={"name": "m"}, agent={"type": "native"})
    path = tmp_path / "t.json"
    _write_trace(str(path), cfg, NativeAgent(), peak_rss_mb=2.0)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["engine"] == "native"
    assert data["peak_rss_mb"] == 2.0


def test_the_run_banner_names_the_engine_that_actually_ran(tmp_path, monkeypatch):
    """The banner carried the same two-engine assumption the trace grew out
    of: `"LangChain (react)" if type == "react" else "native loop (loop.py)"`.
    So a llamaindex or openai-agents run announced itself as the native loop,
    to the user's face, in the one line a demo actually points at. The trace
    was fixed and this was not, which is worse: the screen and the JSON
    disagreed about the same run."""
    from ovat.cli import main as cli_main

    class StubAgent:
        tools = {}
        max_iterations = 5

        def run(self, text):
            return "answer"

    monkeypatch.setattr(cli_main, "build_agent", lambda cfg, **k: StubAgent())

    for engine in ("native", "react", "llamaindex", "openai-agents"):
        config = tmp_path / f"{engine}.yml"
        config.write_text(f"model:\n  name: m\nagent:\n  type: {engine}\n",
                          encoding="utf-8")
        result = runner.invoke(app, ["run", str(config), "-i", "hi"])
        assert result.exit_code == 0
        assert engine in result.output, f"{engine} run never named itself"
        if engine != "native":
            assert "native loop" not in result.output, \
                f"{engine} run claimed to be the native loop"


def test_every_supported_engine_has_a_banner_label(tmp_path, monkeypatch):
    """A new engine added to the factory but not to the label map must fall
    back to its own name. A wrong label is a lie; a plain one is merely terse,
    and this is the fallback that keeps the first from ever happening again."""
    from ovat.agent.factory import AGENT_TYPES
    from ovat.cli import main as cli_main

    class StubAgent:
        tools = {}
        max_iterations = 5

        def run(self, text):
            return "answer"

    monkeypatch.setattr(cli_main, "build_agent", lambda cfg, **k: StubAgent())

    for engine in AGENT_TYPES:
        config = tmp_path / f"{engine}.yml"
        config.write_text(f"model:\n  name: m\nagent:\n  type: {engine}\n",
                          encoding="utf-8")
        result = runner.invoke(app, ["run", str(config), "-i", "hi"])
        assert engine in result.output


def test_the_trace_peak_is_measured_during_the_run_not_after_it(monkeypatch,
                                                                 tmp_path):
    """Reading RSS after the run misses the peak: Python has usually freed
    the big allocations by then, and this number exists for the proposal's
    memory criterion, so a wrong one is worse than none."""
    from ovat.cli import main as cli_main

    sampled = []

    class Agent:
        last_trace = {"engine": "native", "turns": [], "totals": {}}

        def run(self, text):
            # If the sampler is live, it is running WHILE this executes.
            sampled.append("running")
            return "answer"

    monkeypatch.setattr(cli_main, "build_agent", lambda cfg, **k: Agent())
    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\n", encoding="utf-8")
    trace = tmp_path / "trace.json"

    result = runner.invoke(app, ["run", str(config), "-i", "hi",
                                 "--trace", str(trace)])
    assert result.exit_code == 0, result.output

    import json
    data = json.loads(trace.read_text(encoding="utf-8"))
    assert data["peak_rss_mb"] is not None and data["peak_rss_mb"] > 0
    assert sampled == ["running"]


def test_unknown_token_counts_stay_unknown_in_the_totals():
    """A server that sends no usage block would otherwise produce
    "prompt_tokens": 0, which reads as "this run used no tokens" rather than
    "nobody told us", and a benchmark built on that is quietly wrong."""
    from ovat.agent.loop import AgentLoop
    from tests.conftest import FakeLLMProvider, reply

    llm = FakeLLMProvider([reply("hi")])
    original = llm.chat
    llm.chat = lambda messages, tools=None: {
        **original(messages, tools=tools), "usage": None}

    loop = AgentLoop(llm=llm, tools={}, system_prompt=None, max_iterations=3)
    loop.run("q")
    totals = loop.last_trace["totals"]
    assert totals["prompt_tokens"] is None
    assert totals["completion_tokens"] is None
    assert totals["turns"] == 1          # turns ARE known, and stay a number


def test_known_token_counts_are_still_summed():
    from ovat.agent.loop import AgentLoop
    from tests.conftest import FakeLLMProvider, reply

    llm = FakeLLMProvider([reply("hi")])
    original = llm.chat
    llm.chat = lambda messages, tools=None: {
        **original(messages, tools=tools),
        "usage": {"prompt_tokens": 10, "completion_tokens": 4}}

    loop = AgentLoop(llm=llm, tools={}, system_prompt=None, max_iterations=3)
    loop.run("q")
    assert loop.last_trace["totals"]["prompt_tokens"] == 10
    assert loop.last_trace["totals"]["completion_tokens"] == 4


# The benchmark table must say what to DO, not name the exception class

def test_a_missing_extra_shows_the_install_command_not_the_class():
    """This actively misled once. A missing extra reads "RuntimeError:
    agent.type 'llamaindex' needs LlamaIndex. Install it with: pip install
    'ovat[llamaindex]'", and showing only the class rendered a bare
    "RuntimeError" next to two engines that worked: it looked like the engine
    was broken rather than simply not installed."""
    from ovat.cli.main import _brief_error

    message = ("RuntimeError: agent.type 'llamaindex' needs LlamaIndex. "
               "Install it with: pip install 'ovat[llamaindex]'")
    brief = _brief_error(message)
    assert brief.startswith("pip install")
    assert "RuntimeError" not in brief


def test_other_errors_keep_their_detail_and_drop_the_class():
    from ovat.cli.main import _brief_error

    assert _brief_error("APIConnectionError: Connection error.") == \
        "Connection error."
    assert _brief_error(None) == "failed"


def test_a_long_error_is_truncated_with_an_ellipsis():
    from ovat.cli.main import _brief_error

    brief = _brief_error("ValueError: " + "x" * 200)
    assert len(brief) <= 34
    assert brief.endswith("…")


# A bad workflow file must be a sentence, not a traceback

def test_a_missing_workflow_names_the_file_and_the_likely_typo(tmp_path):
    """Every command starts by reading a workflow, so every command
    inherited the raw exception. Typing workflow.yaml for workflow.yml put a
    twenty-line rich traceback on screen with the useful part buried."""
    result = runner.invoke(app, ["run", str(tmp_path / "workflow.yaml"),
                                 "-i", "hi"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "No such workflow file" in result.output
    # The commonest typo by a mile, and invisible when you are staring at it.
    assert "workflow.yml" in result.output


def test_a_missing_workflow_without_that_typo_points_at_init(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path / "nope.yml"), "-i", "hi"])
    assert result.exit_code == 1
    assert "ovat init" in result.output


def test_malformed_yaml_gives_the_line_and_column(tmp_path):
    """Without the position the user is hunting a stray bracket by eye."""
    config = tmp_path / "bad.yml"
    config.write_text("model: [unclosed\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(config), "-i", "hi"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    # Collapse whitespace before matching. Rich wraps to the terminal width,
    # so on a narrow console this message breaks as "not \nvalid YAML" and a
    # substring check fails -- on the product being RIGHT. It passed on a wide
    # developer console and failed in a container, which makes it the same
    # environment-sensitive class of bug as everything else this release
    # fixed: a test that measures the window rather than the code.
    flat = " ".join(result.output.split())
    assert "not valid YAML" in flat
    assert "line" in flat


def test_an_unknown_key_is_named_and_explained(tmp_path):
    """Strictness is deliberate, and "Extra inputs are not permitted" alone
    does not say so."""
    config = tmp_path / "strict.yml"
    config.write_text("model:\n  name: m\nnonsense_key: 1\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(config), "-i", "hi"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "nonsense_key" in result.output
    assert "on purpose" in result.output


def test_a_folder_instead_of_a_file_says_so(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path), "-i", "hi"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "folder, not a workflow file" in result.output


def test_every_command_that_reads_a_workflow_uses_the_friendly_loader():
    """Five commands call it. One that slipped back to load_workflow would
    quietly reintroduce the traceback for its own users only, which is the
    kind of regression nobody notices until a demo."""
    import inspect
    from ovat.cli import main as cli_main

    source = inspect.getsource(cli_main)
    # The helper itself is the single place allowed to call load_workflow.
    body = source.split("def _load_config")[1].split("\ndef ")[0]
    assert body.count("load_workflow(path)") == 1
    outside = source.replace(body, "")
    assert "load_workflow(config)" not in outside


def test_the_bad_config_message_is_the_same_for_every_command(tmp_path):
    """A user who mistypes the path should get the same help whichever
    command they were running.

    doctor is deliberately absent from this list: it is a DIAGNOSTIC, so a
    missing config is a failed check reported alongside the others rather
    than a reason to stop. See the test below.
    """
    missing = str(tmp_path / "workflow.yaml")
    for argv in (["run", missing, "-i", "x"],
                 ["index", str(tmp_path), missing],
                 ["bench", missing, "-i", "x"]):
        result = runner.invoke(app, argv)
        assert "Traceback" not in result.output, argv
        assert "No such workflow file" in result.output, argv


def test_doctor_reports_a_missing_config_as_a_failed_check(tmp_path):
    """Not an early exit: the point of doctor is to run EVERY check and show
    the whole picture, so a bad config is one red row among the green ones,
    not a reason to stop before testing anything else."""
    missing = str(tmp_path / "workflow.yaml")
    result = runner.invoke(app, ["doctor", missing])
    assert "Traceback" not in result.output
    assert "no such file" in result.output.lower()
    assert "fail" in result.output
    # The environment checks still ran.
    assert "Python" in result.output


# `ovat models list` must not be able to hang forever

def test_list_models_gives_up_on_a_wedged_binary(monkeypatch):
    """It only reads a directory, so a binary that has not answered in 30s is
    wedged. Unbounded, `ovat models list` froze with no output and no
    explanation, which is indistinguishable from the command being broken."""
    import subprocess

    from ovat.core.model_manager import ModelManager

    def hang(*args, **kwargs):
        assert kwargs.get("timeout"), "list_models ran with no timeout at all"
        raise subprocess.TimeoutExpired(cmd="ovms", timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    manager = ModelManager(ovms_binary="/fake/ovms")
    with pytest.raises(RuntimeError) as excinfo:
        manager.list_models()
    assert "did not answer" in str(excinfo.value)
    assert "/fake/ovms" in str(excinfo.value)      # names the binary to try


def test_pull_is_deliberately_unbounded(monkeypatch):
    """A real 8B download takes half an hour. A timeout here would abort it
    and leave a half-written repository, which is worse than waiting. The
    asymmetry with list_models is a decision, not an oversight."""
    import subprocess

    from ovat.core.model_manager import ModelManager

    seen = {}

    class Ok:
        returncode = 0
        stdout = "done"
        stderr = ""

    def record(*args, **kwargs):
        seen.update(kwargs)
        return Ok()

    monkeypatch.setattr(subprocess, "run", record)
    ModelManager(ovms_binary="ovms").pull("OpenVINO/Qwen3-8B-int4-ov")
    assert "timeout" not in seen, "pull must NOT be bounded"


def test_a_question_in_the_config_slot_names_the_real_mistake(tmp_path):
    """`ovat run --input workflow.yml` makes --input eat the CONFIG, so the
    question lands in the config slot. Saying "run ovat init" there is
    actively misleading; the mistake is argument order, not a missing file."""
    result = runner.invoke(app, ["run", "hi how are you", "-i", "x"])
    assert result.exit_code == 1
    assert "looks like a question" in result.output
    assert "--input" in result.output
    assert "ovat init" not in result.output      # the misleading tip is gone


def test_a_genuinely_missing_yml_still_points_at_init(tmp_path):
    """The question heuristic must not swallow the ordinary case."""
    result = runner.invoke(app, ["run", str(tmp_path / "nope.yml"), "-i", "x"])
    assert result.exit_code == 1
    assert "ovat init" in result.output
    assert "looks like a question" not in result.output


# --dry-run must not demand a question it never asks

def test_dry_run_needs_no_input():
    """Demanding --input in order to NOT ask a question made the commonest
    smoke test fail with "Missing option --input", which reads as the command
    being broken rather than as a misuse."""
    result = runner.invoke(app, ["run", "examples/document-qa.yml",
                                 "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Built agent" in result.output
    assert "dry-run" in result.output


def test_a_real_run_without_input_says_what_to_type(tmp_path):
    """Optional in the signature, still required in fact. The error shows the
    exact command with the user's own config path in it."""
    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\n", encoding="utf-8")
    result = runner.invoke(app, ["run", str(config)])
    assert result.exit_code == 1
    assert "Missing --input" in result.output
    assert str(config) in result.output          # copyable, not generic
    assert "--dry-run" in result.output          # names the other way out


def test_telemetry_writes_json_lines(monkeypatch, tmp_path):
    """--telemetry streams alongside the run. JSON Lines so a killed run
    still leaves a readable file."""
    import json

    from ovat.cli import main as cli_main

    class Agent:
        last_trace = {"engine": "native", "turns": [], "totals": {"turns": 1}}
        tools: dict = {}

        def run(self, text):
            return "an answer"

    monkeypatch.setattr(cli_main, "build_agent", lambda cfg, **k: Agent())
    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\n", encoding="utf-8")
    out = tmp_path / "tel.jsonl"

    result = runner.invoke(app, ["run", str(config), "-i", "hi",
                                 "--telemetry", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    for line in out.read_text(encoding="utf-8").strip().splitlines():
        json.loads(line)                          # every line stands alone


def _unavailable_sources() -> dict:
    """Which telemetry sources cannot run on THIS machine.

    Intel UT is absent on macOS and present on the AI PC, so a test that
    hardcodes either answer is testing the machine rather than the code.
    """
    from ovat.telemetry.sources import (IntelHardwareSource, ProcessMemorySource,
                                        SystemSource)
    return [s.name for s in (SystemSource(), ProcessMemorySource(),
                             IntelHardwareSource()) if s.unavailable]


def test_telemetry_says_which_sources_are_unavailable(monkeypatch, tmp_path):
    """A file of process memory alone, with no note that the hardware source
    was skipped, reads as a machine with an idle NPU rather than one that
    cannot measure it."""
    from ovat.cli import main as cli_main

    class Agent:
        last_trace = {"totals": {}}
        tools: dict = {}

        def run(self, text):
            return "ok"

    monkeypatch.setattr(cli_main, "build_agent", lambda cfg, **k: Agent())
    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(config), "-i", "hi",
                                 "--telemetry", str(tmp_path / "t.jsonl")])
    # Only assert the notice when a source really IS unavailable. On the AI PC
    # every source works, so demanding the word "unavailable" failed there --
    # the test was describing a Mac, not the contract. The contract is: if a
    # source cannot run, SAY SO; if they all run, there is nothing to say.
    if _unavailable_sources():
        assert "unavailable" in result.output


# `ovat telemetry`: the same sources, for people who live in the terminal

def test_telemetry_once_prints_a_snapshot():
    result = runner.invoke(app, ["telemetry", "--once"])
    assert result.exit_code == 0, result.output
    assert "Source" in result.output and "Metric" in result.output
    # cpu_pct, not just "cpu". Every rate metric on this page is a delta
    # between two readings, so a --once that sampled ONCE printed no CPU
    # percentages at all -- while the system source reported itself neither
    # unavailable nor silent. The old assertion passed anyway, because
    # `cpu_mhz` contains "cpu".
    assert "cpu_pct" in result.output
    assert "cpu0_pct" in result.output


def test_telemetry_names_the_sources_that_cannot_run_here():
    """A table of CPU alone, with no note that the hardware source was
    skipped, reads as a machine with an idle NPU rather than one that cannot
    see it."""
    result = runner.invoke(app, ["telemetry", "--once"])
    if _unavailable_sources():
        assert "unavailable" in result.output
    assert result.exit_code == 0        # the table renders either way


def test_telemetry_exits_nonzero_when_nothing_can_be_measured(monkeypatch):
    """Printing an empty table and exiting 0 would look like a machine doing
    nothing."""
    from ovat.telemetry.collector import Collector

    monkeypatch.setattr(Collector, "available", property(lambda self: []))
    result = runner.invoke(app, ["telemetry", "--once"])
    assert result.exit_code == 1
    assert "No telemetry sources" in result.output


def test_telemetry_can_stream_to_a_file(tmp_path):
    import json

    out = tmp_path / "t.jsonl"
    result = runner.invoke(app, ["telemetry", "--seconds", "0.3",
                                 "--interval", "0.1", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    for line in out.read_text(encoding="utf-8").strip().splitlines():
        json.loads(line)


def test_init_does_not_write_an_active_rag_block(tmp_path):
    """A fresh clone must survive the whole quickstart with no downloads.

    An active rag: block named an embedding model that is on nobody else's
    disk, so `ovat index` -- the next quickstart command -- died with a C++
    assertion out of OpenVINO's frontend. Everything before it succeeded, so
    the config looked right until it did not. This is the bug a mentor hit
    trying to run OVAT for the first time.
    """
    out = tmp_path / "workflow.yml"
    result = runner.invoke(app, ["init", str(out)])
    assert result.exit_code == 0, result.output

    text = out.read_text(encoding="utf-8")
    cfg = load_workflow(str(out))
    assert cfg.rag is None, "init must not enable RAG a new machine cannot run"
    # ...but the instructions must still be there, or we have only moved the
    # dead end from a traceback to a silence.
    assert "optimum-cli export openvino" in text
    assert "bge-small-en-v1.5" in text


def test_indexing_without_rag_says_how_to_turn_it_on(tmp_path):
    """The message a new user now lands on, so it must not be a dead end.

    Naming what is absent ("no rag: section") is not the same as telling
    someone who has never exported an OpenVINO model what to do next.
    """
    out = tmp_path / "workflow.yml"
    runner.invoke(app, ["init", str(out)])
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("hello", encoding="utf-8")

    result = runner.invoke(app, ["index", str(folder), str(out)])
    assert result.exit_code == 1
    # Collapse whitespace first. Rich wraps to the terminal width and tmp_path
    # is long, so the phrase breaks across lines and a raw substring check
    # fails on correct output -- the same trap as the malformed-YAML test, and
    # it reappeared here the moment xdist changed the reported width. Any
    # assertion against Rich output needs this.
    flat = " ".join(result.output.split())
    # Off, not broken; still usable; and where to look.
    assert "nothing is broken" in flat
    assert "stub mode" in flat
    assert "optimum-cli" in flat


def test_run_warns_when_a_tool_call_was_never_decoded(monkeypatch):
    """Raw tool-call markup in the answer means no tool ran. Say so.

    Twice now this has failed silently on live hardware: `tool_parser: auto`
    selected no parser at all, and later an intermittent case where the model
    emitted <parameter=search_docs> instead of <function=search_docs> and
    qwen3coder correctly refused it. Both times the agent answered fluently
    having called nothing, and only a trace inspection revealed it.
    """
    from ovat.cli import main as cli_main

    class RawMarkupAgent:
        tools, max_iterations, last_trace = {}, 5, {}

        def run(self, text):
            return ("<tool_call>\n<function=search_docs>\n"
                    "<parameter=query>budget</parameter>\n"
                    "</function>\n</tool_call>")

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: RawMarkupAgent())
    result = runner.invoke(app, ["run", "examples/react/workflow.yml", "-i", "hi"])
    # Collapse whitespace before matching: rich hard-wraps to the terminal
    # width, so "never run" can arrive as "never \nrun" and a literal
    # substring check fails on a warning that printed perfectly.
    flat = " ".join(result.output.split())
    assert "never run" in flat                    # the warning fired
    assert "qwen3coder" in flat                   # and names the likely cause


def test_run_does_not_warn_on_an_ordinary_answer(monkeypatch):
    """The warning must not cry wolf on every normal reply."""
    from ovat.cli import main as cli_main

    class PlainAgent:
        tools, max_iterations, last_trace = {}, 5, {}

        def run(self, text):
            return "OVAT targets under 8 GB. Source: examples/rag/docs/ovat-facts.md"

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: PlainAgent())
    result = runner.invoke(app, ["run", "examples/react/workflow.yml", "-i", "hi"])
    assert "never run" not in " ".join(result.output.split())


def test_ovat_version_reports_the_installed_version():
    """`ovat --version` exited 2 with a usage error.

    It is among the first things a new user types and the first thing a
    reviewer types to establish which build they are looking at; the only way
    to get it was `pip show ovat`. Found in the AI PC subcommand sweep.
    """
    from importlib.metadata import version

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    # Read from installed metadata, never a literal: a hardcoded string here
    # would drift from pyproject.toml the first time either changed alone.
    assert version("ovat") in result.output


def test_run_prints_sources_even_when_the_model_forgets_to_cite(monkeypatch):
    """The citation is the RAG demo's whole point; do not leave it to the model.

    examples/rag/ promises an answer and its source. On the AI PC retrieval
    fired every single time (traces show search_docs returning 1932 chars) but
    the Source: line appeared in most runs and not all -- one gave the correct
    "under 8 GB" with no source at all, another said "under 8 MB" while citing
    correctly. The system prompt already asks for a citation; the model simply
    does not always comply.

    OVAT already KNOWS the sources: they are in the trace, on the tool result.
    `ovat chat` prints a separate `sources:` line for exactly this reason, and
    `ovat run` should behave the same rather than hoping.
    """
    from ovat.cli import main as cli_main

    class ForgetfulAgent:
        tools, max_iterations = {"search_docs": {}}, 5
        last_trace = {
            "engine": "native",
            "turns": [{"latency_s": 0.1, "finish_reason": "tool_calls",
                       "prompt_tokens": 1, "completion_tokens": 1,
                       "tool_calls": [{"name": "search_docs", "arguments": {},
                                       "duration_s": 0.1, "result_chars": 99,
                                       "sources": ["examples/rag/docs/ovat-facts.md"]}]}],
            "totals": {"turns": 1, "tool_calls": 1},
        }

        def run(self, text):
            return "OVAT targets under 8 GB."      # no citation at all

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: ForgetfulAgent())
    result = runner.invoke(app, ["run", "examples/react/workflow.yml", "-i", "hi"])
    flat = " ".join(result.output.split())
    assert "ovat-facts.md" in flat, "the source OVAT already knew was not shown"


def test_run_prints_no_sources_line_when_nothing_was_retrieved(monkeypatch):
    """No retrieval means no sources block -- do not print an empty header."""
    from ovat.cli import main as cli_main

    class PlainAgent:
        tools, max_iterations = {}, 5
        last_trace = {"engine": "native", "turns": [], "totals": {}}

        def run(self, text):
            return "Paris."

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: PlainAgent())
    result = runner.invoke(app, ["run", "examples/react/workflow.yml", "-i", "hi"])
    assert "sources:" not in result.output.lower()


def test_run_reports_a_build_failure_instead_of_tracebacking(tmp_path, monkeypatch):
    """`ovat run` tracebacked where `index` and `chat` explained themselves.

    Found on the AI PC verifying v0.9.8. The friendly message added to
    build_embedder reached two of the three commands that build a retriever:

        ovat index  -> "Could not build the embedder/retriever: ..."   clean
        ovat chat   -> "Could not build the retriever: ..."            clean
        ovat run    -> full rich-rendered Python traceback, THEN the message

    Because main.py's run() called build_agent() unprotected while those two
    wrapped it. Same fact, same wording, one of three commands rude about it --
    which is worse than being consistently rude, because it reads as a crash
    rather than a misconfiguration.
    """
    from ovat.cli import main as cli_main

    def boom(cfg, skip_rag=False):
        raise RuntimeError("embeddings model not found at 'models/nope'")

    monkeypatch.setattr(cli_main, "build_agent", boom)
    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(config), "-i", "hi"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit), \
        "the RuntimeError escaped as a traceback"
    flat = " ".join(result.output.split())
    assert "Could not build the agent" in flat
    assert "models/nope" in flat            # the cause still reaches the user


def test_dry_run_reports_a_build_failure_the_same_way(tmp_path, monkeypatch):
    """--dry-run builds too, so it needs the same courtesy."""
    from ovat.cli import main as cli_main

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: (_ for _ in ()).throw(
                            RuntimeError("nope")))
    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\n", encoding="utf-8")
    result = runner.invoke(app, ["run", str(config), "--dry-run"])
    assert result.exit_code == 1
    assert "Could not build the agent" in " ".join(result.output.split())


def test_a_timeout_names_the_setting_that_fixes_it(tmp_path, monkeypatch):
    """"Request timed out." names neither the knob nor the file.

    Measured on a CPU-only Ubuntu box: the server answers one completion in
    about a second, but an agent turn is max_iterations rounds of them, so a
    cold first run blows the 120 s default and reads as a broken server. The
    fix is a config value, so the message has to say which one.
    """
    cfg = tmp_path / "w.yml"
    cfg.write_text(
        "model:\n  name: X\n  ovms_url: http://localhost:8000/v3\n"
        "agent:\n  type: native\n", encoding="utf-8")

    class TimesOut:
        last_trace = {}

        def run(self, _):
            raise RuntimeError("Request timed out.")

        def close(self):
            pass

    from ovat.cli import main as cli_main
    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: TimesOut())
    result = runner.invoke(app, ["run", str(cfg), "-i", "hi"])

    assert result.exit_code == 1
    assert "request_timeout" in result.output
    assert "timeout, not a crash" in result.output
    # and it must point at THEIR file, not a generic one
    assert "w.yml" in result.output


def test_an_ordinary_failure_does_not_get_the_timeout_advice(tmp_path,
                                                             monkeypatch):
    """Only timeouts. Telling someone to raise request_timeout after a
    connection refused would send them to the wrong place entirely."""
    cfg = tmp_path / "w.yml"
    cfg.write_text(
        "model:\n  name: X\n  ovms_url: http://localhost:8000/v3\n"
        "agent:\n  type: native\n", encoding="utf-8")

    class Refused:
        last_trace = {}

        def run(self, _):
            raise RuntimeError("Connection refused")

        def close(self):
            pass

    from ovat.cli import main as cli_main
    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: Refused())
    result = runner.invoke(app, ["run", str(cfg), "-i", "hi"])

    assert result.exit_code == 1
    assert "request_timeout" not in result.output


# --------------------------------------------------------------------------
# Every command must be reachable THROUGH the CLI, not only through the
# module it wraps. setup, serve and models had thorough library tests and no
# invocation test at all, so a broken option name, a typo'd import inside the
# command body, or a Typer signature the runner rejects would have shipped
# green. These are cheap, and they close the last three gaps.
# --------------------------------------------------------------------------

def test_setup_command_is_reachable_and_reports_what_it_did(monkeypatch):
    from ovat.core import ovms_installer

    # installed_binary too, not just install(): the command checks for an
    # existing install FIRST and reports "already installed", so on any machine
    # that really has OVMS this asserted against the real path. It passed only
    # where OVMS was absent -- the same works-on-my-machine shape this release
    # is about, and I wrote it.
    monkeypatch.setattr(ovms_installer, "installed_binary", lambda root=None: None)
    monkeypatch.setattr(ovms_installer, "install",
                        lambda **kw: ("/somewhere/ovms", "installed"))
    monkeypatch.setattr(ovms_installer, "asset_for_platform",
                        lambda version=None: ("ovms_test.tar.gz", "file:///x"))
    monkeypatch.setattr(ovms_installer, "linux_support_note", lambda: None)

    result = runner.invoke(app, ["setup", "--yes"])
    assert result.exit_code == 0, result.output
    assert "/somewhere/ovms" in result.output
    # The promise the README makes, in the command's own words.
    assert "PATH" in result.output


def test_setup_says_nothing_to_do_on_macos_and_exits_zero(monkeypatch):
    """Not an error. An unsupported platform is a fact to report, and exiting
    1 would fail any script that runs setup unconditionally."""
    from ovat.cli import main as cli_main
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0, result.output
    assert "macOS" in result.output
    assert "ovat chat" in result.output          # says what DOES work

def test_serve_passes_ovms_port_to_modelserver(monkeypatch, tmp_path):
    from ovat.cli import main as cli_main
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    cfg = tmp_path / "w.yml"
    cfg.write_text("model:\n  name: X\n  ovms_port: 8002\nagent:\n  type: native\n", encoding="utf-8")

    captured_kwargs = {}

    class FakeModelServer:
        DEFAULT_STALL_TIMEOUT = 120
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.model_name = kwargs.get("model_name")
            self.port = kwargs.get("port")
            class DummyProc:
                pid = 1234
                def poll(self): return None
            self.process = DummyProc()
            self.log_path = ""
            self.base_url = f"http://localhost:{self.port or 8000}/v3"
        def wait_until_ready(self, *args, **kwargs):
            return True
        def start(self, *args, **kwargs):
            pass
        def stop(self):
            pass
        def already_serving(self):
            return None                    # nothing on the port in this test

    from ovat.core import model_server
    monkeypatch.setattr(model_server, "ModelServer", FakeModelServer)
    from ovat.core import model_scout, ovms_locator
    monkeypatch.setattr(model_scout, "identify_model", lambda *args: "llm")
    monkeypatch.setattr(ovms_locator, "find_ovms", lambda *args: ("/path/to/ovms", "1.0"))

    result = runner.invoke(app, ["serve", str(cfg)])
    assert result.exit_code == 0, result.output
    assert captured_kwargs.get("port") == 8002


def _serve_with_occupied_port(monkeypatch, tmp_path, serving, model="X"):
    """Run `ovat serve` against a port that already has a server on it."""
    from ovat.cli import main as cli_main
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    cfg = tmp_path / "w.yml"
    cfg.write_text(f"model:\n  name: {model}\nagent:\n  type: native\n",
                   encoding="utf-8")
    started = []

    class FakeModelServer:
        DEFAULT_STALL_TIMEOUT = 120

        def __init__(self, **kwargs):
            self.port = kwargs.get("port") or 8000
            self.base_url = f"http://localhost:{self.port}/v3"
            self.log_path = ""
            self.process = None

        def already_serving(self):
            return serving

        def start(self, *args, **kwargs):
            started.append(True)

        def wait_until_ready(self, *args, **kwargs):
            return True

        def stop(self):
            pass

    from ovat.core import model_server, model_scout, ovms_locator
    monkeypatch.setattr(model_server, "ModelServer", FakeModelServer)
    monkeypatch.setattr(model_scout, "identify_model", lambda *args: "llm")
    monkeypatch.setattr(ovms_locator, "find_ovms",
                        lambda *args: ("/path/to/ovms", "1.0"))
    result = runner.invoke(app, ["serve", str(cfg)])
    return result, started


def test_serve_refuses_to_start_a_second_server_for_a_different_model(
        monkeypatch, tmp_path):
    """Starting anyway leaves two servers on one port, on Windows.

    Measured: two ovms.exe both LISTENING on 0.0.0.0:8000, the pidfile naming
    only the newer, so --stop orphans the older one holding the port and 4 GB.
    The orphan here was serving a DIFFERENT model, which is what made a live
    suite fail against a server that was running and wrong.
    """
    result, started = _serve_with_occupied_port(
        monkeypatch, tmp_path, serving="SomeOtherModel", model="X")

    assert not started, "a second server was started on an occupied port"
    assert result.exit_code == 1
    assert "already answering" in result.output
    assert "SomeOtherModel" in result.output
    assert "--stop" in result.output


def test_serve_is_content_when_the_right_model_is_already_served(
        monkeypatch, tmp_path):
    """Asking for a served model and finding it served is not an error."""
    result, started = _serve_with_occupied_port(
        monkeypatch, tmp_path, serving="X", model="X")

    assert not started
    assert result.exit_code == 0
    assert "nothing to start" in result.output


def test_serve_stop_is_reachable_through_the_cli(monkeypatch, tmp_path):
    from ovat.core import model_server

    cfg = tmp_path / "w.yml"
    cfg.write_text("model:\n  name: X\nagent:\n  type: native\n",
                   encoding="utf-8")
    monkeypatch.setattr(model_server, "stop_from_pidfile",
                        lambda *a, **k: "Stopped OVMS (pid 4321).")

    result = runner.invoke(app, ["serve", str(cfg), "--stop"])
    assert result.exit_code == 0, result.output
    assert "4321" in result.output


def test_models_list_is_reachable_and_needs_no_network(monkeypatch):
    from ovat.core import ovms_locator

    monkeypatch.setattr(ovms_locator, "find_ovms",
                        lambda explicit=None: (None, "not found (test)"))
    result = runner.invoke(app, ["models", "list"])
    # Either it lists, or it explains it cannot find ovms. What it must NOT do
    # is raise: `models` is the one command whose whole job is talking to a
    # binary that is usually absent.
    assert "Traceback" not in result.output


def test_a_run_that_could_not_answer_exits_non_zero(monkeypatch):
    """Failures are delivered AS the answer, so the exit code is the only
    signal a script can read.

    The loop returns "Error: ..." as the answer text because the model has to
    be able to read it. `ovat run` then printed it and exited 0, so on an AI PC
    four consecutive failed runs all reported success -- undetectable by CI, a
    benchmark, or any wrapper script.
    """
    from ovat.cli import main as cli_main

    class Failing:
        tools, max_iterations = {}, 5
        last_trace = {"totals": {"failed": True, "undecoded_tool_call": True}}

        def run(self, text):
            return ("Error: the model asked for a tool but the server could "
                    "not decode the request, so no tool ran.")

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: Failing())
    result = runner.invoke(app, ["run", "examples/workflow.yml", "-i", "hi"])

    assert result.exit_code == 1
    # and the human still gets the explanation, not just a bare status
    assert "could not decode" in result.output


def test_a_successful_run_still_exits_zero(monkeypatch):
    """The other half: a good answer must not be reported as a failure."""
    from ovat.cli import main as cli_main

    class Working:
        tools, max_iterations = {}, 5
        last_trace = {"totals": {"failed": False}}

        def run(self, text):
            return "I have search_docs and transcribe."

    monkeypatch.setattr(cli_main, "build_agent",
                        lambda cfg, skip_rag=False: Working())
    result = runner.invoke(app, ["run", "examples/workflow.yml", "-i", "hi"])
    assert result.exit_code == 0


def _telemetry_in(tmp_path, monkeypatch, cache_line):
    """Run `ovat telemetry --once` in a directory with a canned ovms.log."""
    (tmp_path / "ovms.log").write_text(cache_line + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return runner.invoke(app, ["telemetry", "--once"])


def test_telemetry_says_a_dynamic_cache_is_not_full(tmp_path, monkeypatch):
    """kv_cache_pct alone is the most misleading number on this page.

    Unset, OVMS grows the cache on demand, so it sits at or near 100% of the
    current allocation as its ordinary state -- 61.6% of one session's
    readings were >=95% while the allocation went 248.5 MB to 5.6 GB. Shown
    bare, "99.2" reads as a server about to fall over, and that conclusion
    was actually drawn here across three sessions.

    The type cannot go in the table: every sample value is formatted as a
    number, and a string there took --once down outright. So it is printed
    beside it.
    """
    result = _telemetry_in(
        tmp_path, monkeypatch,
        "All requests: 1; Cache type: dynamic, cache usage: 99.2% of 5.6 GB;")
    assert result.exit_code == 0, result.output
    assert "kv_cache_pct" in result.output
    assert "DYNAMIC" in result.output
    assert "not full" in result.output


def test_telemetry_says_a_static_cache_percentage_is_real(tmp_path, monkeypatch):
    result = _telemetry_in(
        tmp_path, monkeypatch,
        "All requests: 1; Cache type: static, cache usage: 99.8% of 1.0 GB;")
    assert result.exit_code == 0, result.output
    assert "STATIC" in result.output
    assert "real utilisation" in result.output


def _serve_with_port_busy(tmp_path, monkeypatch, model_name, serving):
    """Run `ovat serve` against a port that already answers with `serving`."""
    from ovat.core import model_scout, model_server, ovms_locator

    monkeypatch.setattr(model_server.ModelServer, "already_serving",
                        lambda self: serving)
    monkeypatch.setattr(model_scout, "identify_model", lambda *a: "llm")
    monkeypatch.setattr(ovms_locator, "find_ovms",
                        lambda *a: ("/path/to/ovms", "1.0"))
    config = tmp_path / "w.yml"
    config.write_text(f"model:\n  name: {model_name}\n  provider: ovms\n",
                      encoding="utf-8")
    return runner.invoke(app, ["serve", str(config)])


def test_a_config_naming_a_prefix_of_the_running_model_is_not_the_same(
        tmp_path, monkeypatch):
    """`"Qwen3-8B" in "Qwen3-8B-int4-cw-ov"` is True, and must not count.

    This is the wrong half of the port guard to get wrong. Told "that is the
    model this config asks for", the user exits 0 and then talks to a server
    running something else -- which is how a live suite once failed against a
    server that was up and serving the wrong model.
    """
    result = _serve_with_port_busy(tmp_path, monkeypatch, "Qwen3-8B",
                                   "Qwen3-8B-int4-cw-ov")
    assert result.exit_code == 1, result.output
    assert "NOT Qwen3-8B" in result.output


def test_the_exact_model_already_serving_is_not_an_error(tmp_path, monkeypatch):
    """The user asked for a served model and there is one."""
    result = _serve_with_port_busy(tmp_path, monkeypatch,
                                   "Qwen3-8B-int4-cw-ov",
                                   "Qwen3-8B-int4-cw-ov")
    assert result.exit_code == 0, result.output
    assert "nothing to start" in result.output
