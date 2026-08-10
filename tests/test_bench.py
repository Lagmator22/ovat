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
    # Positive, not >= the injected delay. Windows' default timer granularity
    # is ~15.6 ms, so time.sleep(0.02) genuinely returns in 0.016 and the old
    # floor failed on the runner while the code was correct. What this test
    # exists to prove is that a latency is MEASURED and recorded, not that the
    # OS scheduler is precise.
    assert row["latency_s"] > 0
    assert row["build_s"] is not None


def test_each_engine_is_measured_in_its_own_process(monkeypatch, tmp_path):
    """Peak MB is whole-process RSS, so engines sharing a process contaminate
    each other: every row inherited what the rows above it had allocated.

    Measured on the AI PC against live OVMS BEFORE this fix: native read
    465.8 MB running first and 1155.6 MB running last, against 466 MB
    measured alone. Reversing the list reversed the conclusion, which means
    the table was measuring list position rather than the engines.
    """
    import ovat.bench as bench_mod

    spawned = []

    def fake_isolated(config_path, engine, question, timeout=None):
        spawned.append(engine)
        return {"engine": engine, "ok": True, "error": None, "answer": "a",
                "latency_s": 1.0, "build_s": 0.1, "peak_rss_mb": 100.0,
                "prompt_tokens": None, "completion_tokens": None,
                "tool_calls": None}

    monkeypatch.setattr(bench_mod, "_run_isolated", fake_isolated)
    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\n", encoding="utf-8")

    result = runner.invoke(app, ["bench", str(config), "-i", "q",
                                 "--engines", "native,react"])
    assert result.exit_code == 0, result.output
    assert spawned == ["native", "react"], "engines were not isolated"
    assert "its own process" in result.output

    one = runner.invoke(app, ["bench", str(config), "-i", "q",
                              "--engines", "native"])
    assert "its own process" not in one.output, \
        "a single-engine run needs no note about cross-row contamination"


def test_the_report_records_whether_the_rows_are_isolated(tmp_path):
    """A reader of the JSON months later cannot tell from the numbers alone,
    and comparing an isolated run against a shared one is meaningless."""
    shared = benchmark(_config(), "q", ["native"],
                       build_agent=lambda cfg: _Agent())
    assert shared["isolated"] is False


def test_the_worker_prints_one_json_row_and_never_a_traceback(tmp_path):
    """The parent parses stdout, so a traceback there is unparseable AND
    hides the reason. A bad config must come back as a row."""
    import json
    from ovat.bench import _worker_main

    import io
    import contextlib

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = _worker_main(["--worker", "--config", str(tmp_path / "nope.yml"),
                             "--engine", "native", "--question", "q"])
    assert code == 0
    row = json.loads(out.getvalue().strip())
    assert row["ok"] is False
    assert row["engine"] == "native"
    assert "FileNotFoundError" in row["error"]


def test_a_worker_that_dies_becomes_a_row_not_a_crash(monkeypatch):
    """A segfaulting or OOM-killed child prints nothing at all."""
    import subprocess
    from ovat import bench as bench_mod

    class Dead:
        stdout = ""
        stderr = "Fatal Python error: something awful\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Dead())
    row = bench_mod._run_isolated("w.yml", "native", "q")
    assert row["ok"] is False
    assert "something awful" in row["error"]


def test_a_hung_engine_times_out_instead_of_blocking_forever(monkeypatch):
    import subprocess
    from ovat import bench as bench_mod

    def explode(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ovat", timeout=1)

    monkeypatch.setattr(subprocess, "run", explode)
    row = bench_mod._run_isolated("w.yml", "native", "q", timeout=1)
    assert row["ok"] is False
    assert "TimeoutError" in row["error"]


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
    # See above: a failing engine must still report how long it waited, and
    # that is what > 0 asserts. A hard floor measured the clock, not the code.
    assert row["latency_s"] > 0


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

    monkeypatch.setattr(bench_module, "_run_isolated",
                        lambda config_path, engine, q, timeout=None: {
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


def test_bench_does_not_score_an_empty_answer_as_ok():
    """"ok" must mean the engine ANSWERED, not merely that it did not raise.

    Found on the AI PC: with tool_parser: hermes3 forced onto a Qwen3.5 model,
    hermes3 SWALLOWED the tool call and the engine returned cleanly with a
    reply containing only reasoning -- and bench scored the row ok. The point
    of the table is a like-for-like comparison, so a row reading as success
    while the model said nothing and called no tool makes the table lie.
    """
    row = benchmark_engine(
        _config(), "native", "q",
        build_agent=lambda cfg: _Agent(answer="reasoning only</think>"))
    assert row["ok"] is False, "an empty answer was scored as a success"
    assert "no answer" in (row["error"] or "")


def test_bench_still_scores_a_real_answer_ok():
    """The guard must not reject ordinary answers."""
    row = benchmark_engine(_config(), "native", "q",
                           build_agent=lambda cfg: _Agent(answer="Under 8 GB."))
    assert row["ok"] is True
    assert row["error"] is None


def test_bench_reads_the_trace_not_the_answer_text():
    """The regression that shipped in 0.9.7 and was caught on hardware.

    Two fixes in one commit defeated each other. The loop began converting an
    empty reply into an "Error: the model returned no answer" string -- so the
    bench guard, which tested says_nothing(answer), saw non-empty text and
    scored the row ok. Verified on the AI PC: forcing hermes3 onto Qwen3.5 gave
    ok=True, error=None, tool_calls=0, with the diagnosis sitting harmlessly in
    the answer column. A green row for a run that answered nothing.

    The trace flags cannot be fooled that way: they record what happened, not
    what was printed.
    """
    class TracedAgent:
        tools: dict = {}
        # Exactly what the loop produces for a swallowed reply.
        last_trace = {"engine": "native", "turns": [],
                      "totals": {"turns": 1, "tool_calls": 0,
                                 "empty_answer": True,
                                 "undecoded_tool_call": False}}

        def run(self, question):
            return ("Error: the model returned no answer. Its reply contained "
                    "only reasoning, or nothing at all.")

    row = benchmark_engine(_config(), "native", "q",
                           build_agent=lambda cfg: TracedAgent())
    assert row["ok"] is False, "a non-empty error string scored as success"
    assert "no answer" in (row["error"] or "")


def test_bench_flags_an_undecoded_tool_call_too():
    """The sibling flag gets the same treatment."""
    class TracedAgent:
        tools: dict = {}
        last_trace = {"engine": "native", "turns": [],
                      "totals": {"turns": 1, "tool_calls": 0,
                                 "empty_answer": False,
                                 "undecoded_tool_call": True}}

        def run(self, question):
            return "Error: the model asked for a tool but ..."

    row = benchmark_engine(_config(), "native", "q",
                           build_agent=lambda cfg: TracedAgent())
    assert row["ok"] is False
    assert "could not decode" in (row["error"] or "")


def test_a_build_failure_is_a_row_not_a_crash():
    """Checked because the v0.9.8 report suspected bench had `run`'s gap.

    It does not: benchmark_engine already catches any build exception and turns
    it into a row, which is the documented contract -- "a missing framework or
    an unreachable OVMS is a RESULT, not a crash", because the whole point of a
    comparison table is that the engine which failed is visible beside the ones
    that did not. Pinned here so the suspicion does not have to be re-checked.
    """
    def boom(cfg):
        raise RuntimeError("embeddings model not found at 'models/nope'")

    row = benchmark_engine(_config(), "native", "q", build_agent=boom)
    assert row["ok"] is False
    assert "models/nope" in (row["error"] or "")     # the cause survives
    assert row["engine"] == "native"                  # and it IS a row
