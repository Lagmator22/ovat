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
    assert cfg.model.name == "Qwen3-8B-int4-ov"


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
    assert data["model"] == "Qwen3-8B-int4-ov"      # enriched from the config
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
    assert "not valid YAML" in result.output
    assert "line" in result.output


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
    assert "unavailable" in result.output


# `ovat telemetry`: the same sources, for people who live in the terminal

def test_telemetry_once_prints_a_snapshot():
    result = runner.invoke(app, ["telemetry", "--once"])
    assert result.exit_code == 0, result.output
    assert "Source" in result.output and "Metric" in result.output
    assert "cpu" in result.output.lower()


def test_telemetry_names_the_sources_that_cannot_run_here():
    """A table of CPU alone, with no note that the hardware source was
    skipped, reads as a machine with an idle NPU rather than one that cannot
    see it."""
    result = runner.invoke(app, ["telemetry", "--once"])
    assert "unavailable" in result.output


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
