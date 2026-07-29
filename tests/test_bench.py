# tests/test_bench.py
"""Tests for the cross-engine benchmark (the AI PC profiling deliverable).

No engine is really built here: build_agent is injected, so the measurement
logic is tested on its own and the whole file runs in milliseconds on any
machine with no OVMS.
"""
import time

from typer.testing import CliRunner

from ovat.bench import benchmark, benchmark_engine
from ovat.cli.main import app
from ovat.config.workflow import WorkflowConfig

runner = CliRunner()


def _config() -> WorkflowConfig:
    return WorkflowConfig(model={"name": "test-model"})


class _Agent:
    """Stands in for a built agent. `trace` mimics the native loop's."""

    def __init__(self, answer="the answer", delay=0.0, trace=None, boom=None):
        self._answer, self._delay, self._boom = answer, delay, boom
        if trace is not None:
            self.last_trace = trace

    def run(self, question):
        time.sleep(self._delay)
        if self._boom:
            raise self._boom
        return self._answer


def test_a_successful_run_reports_timings_and_the_answer():
    row = benchmark_engine(_config(), "native", "q",
                           build_agent=lambda cfg: _Agent(delay=0.02))
    assert row["ok"] is True
    assert row["error"] is None
    assert row["answer"] == "the answer"
    assert row["latency_s"] >= 0.02
    assert row["build_s"] is not None


def test_a_multi_engine_table_admits_the_peak_is_cumulative(monkeypatch,
                                                            tmp_path):
    """Peak MB is whole-process RSS and every engine runs in ONE process, so
    each engine after the first is inflated by the ones before it.

    Measured on the AI PC against live OVMS: native reads 465.8 MB when it
    runs first and 1155.6 MB when the same run order is reversed and it runs
    last, while its own isolated figure (`--engines native`, and `ovat run
    --trace`) is 466-469 MB. The lightest engine is reported as the heaviest
    purely because of its position in the list.

    The real fix is a process per engine. Until then the table must SAY the
    column is cumulative, because a wrong number under a confident heading is
    exactly the kind of measurement this module exists to avoid.
    """
    from ovat.cli import main as cli_main

    monkeypatch.setattr(
        cli_main, "benchmark_engine_used_by_tests", None, raising=False)
    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\n", encoding="utf-8")

    import ovat.bench as bench_mod
    monkeypatch.setattr(
        bench_mod, "benchmark_engine",
        lambda cfg, engine, q, build_agent=None: {
            "engine": engine, "ok": True, "error": None, "answer": "a",
            "latency_s": 1.0, "build_s": 0.1, "peak_rss_mb": 500.0,
            "prompt_tokens": 1, "completion_tokens": 1, "tool_calls": 0})

    many = runner.invoke(app, ["bench", str(config), "-i", "q",
                               "--engines", "native,react"])
    assert many.exit_code == 0
    assert "cumulative" in many.output, \
        "a multi-engine table must disclose that Peak MB is whole-process"

    one = runner.invoke(app, ["bench", str(config), "-i", "q",
                              "--engines", "native"])
    assert one.exit_code == 0
    assert "cumulative" not in one.output, \
        "a single-engine run IS isolated; do not warn about nothing"


def test_the_engine_under_test_is_the_one_that_gets_built():
    """Each row must run the engine it is labelled with, or the whole table
    is measuring the same engine four times."""
    seen = []

    def build(cfg):
        seen.append(cfg.agent.type)
        return _Agent()

    for engine in ("native", "react", "llamaindex"):
        benchmark_engine(_config(), engine, "q", build_agent=build)
    assert seen == ["native", "react", "llamaindex"]


def test_the_shared_config_is_not_mutated_between_engines():
    """Every row copies the config. Without that, the first engine's write
    leaks into the next one's run and the labels stop matching reality."""
    config = _config()
    benchmark_engine(config, "llamaindex", "q",
                     build_agent=lambda cfg: _Agent())
    assert config.agent.type == "native"


def test_a_missing_framework_is_a_row_not_a_crash():
    """The engine that failed has to be VISIBLE next to the ones that did
    not. That is the entire value of a comparison table."""
    def build(cfg):
        raise RuntimeError("needs LlamaIndex. Install it with: pip install ...")

    row = benchmark_engine(_config(), "llamaindex", "q", build_agent=build)
    assert row["ok"] is False
    assert "needs LlamaIndex" in row["error"]
    assert row["latency_s"] is None       # it never got as far as running


def test_an_unreachable_server_still_reports_how_long_it_waited():
    """A timeout is a measurement too, and on an AI PC it is often the most
    interesting number on the row."""
    row = benchmark_engine(
        _config(), "native", "q",
        build_agent=lambda cfg: _Agent(delay=0.02,
                                       boom=ConnectionError("refused")))
    assert row["ok"] is False
    assert "refused" in row["error"]
    assert row["latency_s"] >= 0.02


def test_tokens_are_absent_rather_than_zero_when_unknown():
    """Only the native loop records OVMS's usage field. A zero would read as
    "this engine used no tokens", which is a false claim, so unknown stays
    None and the table prints a dash."""
    row = benchmark_engine(_config(), "react", "q",
                           build_agent=lambda cfg: _Agent())
    assert row["prompt_tokens"] is None
    assert row["completion_tokens"] is None


def test_tokens_are_reported_when_the_engine_recorded_them():
    trace = {"totals": {"prompt_tokens": 120, "completion_tokens": 45,
                        "tool_calls": 2}}
    row = benchmark_engine(_config(), "native", "q",
                           build_agent=lambda cfg: _Agent(trace=trace))
    assert row["prompt_tokens"] == 120
    assert row["completion_tokens"] == 45
    assert row["tool_calls"] == 2


def test_peak_memory_is_sampled_during_the_run_not_read_after_it():
    """A single reading taken at the end misses the peak, because Python has
    usually freed the big allocations by then, and the number exists for the
    proposal's memory criterion."""
    def build(cfg):
        return _Agent(delay=0.15)

    row = benchmark_engine(_config(), "native", "q", build_agent=build)
    assert row["peak_rss_mb"] is not None
    assert row["peak_rss_mb"] > 0


def test_the_sampler_thread_does_not_outlive_the_run():
    """A daemon thread per engine, left running, would still be sampling
    while the next engine is measured."""
    import threading

    before = threading.active_count()
    for engine in ("native", "react", "llamaindex", "openai-agents"):
        benchmark_engine(_config(), engine, "q",
                         build_agent=lambda cfg: _Agent())
    time.sleep(0.1)
    assert threading.active_count() <= before


def test_the_report_carries_what_was_being_compared():
    """A table of numbers with no record of the model or the server it ran
    against is not evidence of anything."""
    report = benchmark(_config(), "what is OVMS?",
                       ["native", "react"],
                       build_agent=lambda cfg: _Agent())
    assert report["model"] == "test-model"
    assert report["ovms_url"].endswith("/v3")
    assert report["question"] == "what is OVMS?"
    assert [r["engine"] for r in report["results"]] == ["native", "react"]


# The CLI wrapper

def test_bench_writes_a_json_report(monkeypatch, tmp_path):
    from ovat import bench as bench_module

    monkeypatch.setattr(bench_module, "benchmark_engine",
                        lambda cfg, engine, q, build_agent=None: {
                            "engine": engine, "ok": True, "error": None,
                            "answer": "a", "latency_s": 1.0, "build_s": 0.5,
                            "peak_rss_mb": 100.0, "prompt_tokens": 10,
                            "completion_tokens": 5, "tool_calls": 1})

    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\n", encoding="utf-8")
    out = tmp_path / "report.json"

    result = runner.invoke(app, ["bench", str(config), "-i", "hi",
                                 "--engines", "native,react",
                                 "--out", str(out)])
    assert result.exit_code == 0, result.output
    import json
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [r["engine"] for r in report["results"]] == ["native", "react"]
    assert "native" in result.output and "react" in result.output


def test_bench_refuses_an_empty_engine_list(tmp_path):
    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\n", encoding="utf-8")
    result = runner.invoke(app, ["bench", str(config), "-i", "hi",
                                 "--engines", " , "])
    assert result.exit_code == 1
    assert "No engines" in result.output
