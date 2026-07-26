# tests/test_cli.py
"""Tests for the ovat command line.

Note to myself: typer ships a CliRunner that invokes my commands in-process and
captures their output and exit code, so I can test the CLI without a real shell
or a running OVMS server. I test the wiring (init writes a usable config,
dry-run builds the agent) and the error paths (bad config exits non-zero).
"""
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
