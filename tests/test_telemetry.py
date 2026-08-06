# tests/test_telemetry.py
"""Tests for the telemetry contracts, sources, sinks and collector.

None of these need hardware: the point of the two ABCs is that a fake source
and a fake sink exercise the whole path in milliseconds.
"""
import os
import pytest

from ovat.telemetry.base import TelemetrySink, TelemetrySource
from ovat.telemetry.collector import Collector
from ovat.telemetry.sinks import FanOutSink, JSONFileSink, LiveBufferSink
from ovat.telemetry.sources import (AgentTraceSource, IntelHardwareSource,
                                    ProcessMemorySource, find_ut)


class _Fake(TelemetrySource):
    name = "fake"

    def __init__(self, value=1.0, reason=None):
        self.value, self.reason = value, reason
        self.started = self.stopped = False

    def sample(self):
        return {"metric": self.value}

    @property
    def unavailable(self):
        return self.reason

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


# The contracts

def test_a_source_that_cannot_run_says_WHY_rather_than_reporting_zero():
    """An absent source and an idle one look identical in a graph, and only
    one of them is worth acting on. A silent empty sample is the worst
    outcome because it reads as a real measurement of nothing."""
    dead = _Fake(reason="no such hardware here")
    collector = Collector([dead], LiveBufferSink())
    assert collector.available == []
    assert collector.unavailable == {"fake": "no such hardware here"}


def test_a_broken_source_does_not_end_the_collection():
    """A telemetry source that takes down the run it is measuring has
    inverted its own purpose."""
    class Exploding(TelemetrySource):
        name = "boom"

        def sample(self):
            raise RuntimeError("sensor on fire")

    live = LiveBufferSink()
    collector = Collector([Exploding(), _Fake(value=7.0)], live)
    merged = collector.sample_once()            # must not raise
    assert merged == {"fake.metric": 7.0}


def test_samples_are_namespaced_by_source():
    """Two sources both reporting rss_mb must not overwrite each other."""
    a, b = _Fake(1.0), _Fake(2.0)
    b.name = "other"
    merged = Collector([a, b], LiveBufferSink()).sample_once()
    assert merged == {"fake.metric": 1.0, "other.metric": 2.0}


# The live buffer, which is what the TUI page draws

def test_the_live_buffer_is_bounded():
    """A telemetry page left open for an hour must not be the reason the
    process runs out of memory."""
    live = LiveBufferSink(capacity=5)
    for i in range(50):
        live.record({"m": i})
    assert len(live.samples) == 5
    assert live.latest("m") == 49


def test_a_gap_in_a_source_is_skipped_not_zero_filled():
    """A sparkline that dips to the floor whenever a collector stalls is a
    graph that lies. A missing reading is not a reading of zero."""
    live = LiveBufferSink()
    live.record({"m": 5})
    live.record({"other": 1})           # m absent this frame
    live.record({"m": 7})
    assert live.series("m") == [5, 7]


def test_metrics_are_discovered_not_hardcoded():
    """The page cannot know which sources are live, so it asks. A hardcoded
    list goes stale the moment a source is added."""
    live = LiveBufferSink()
    live.record({"a": 1, "b": 2})
    live.record({"c": 3})
    assert set(live.metrics()) == {"a", "b", "c"}
    assert "t" not in live.metrics()    # the timestamp is not a metric


def test_latest_returns_the_default_when_never_seen():
    assert LiveBufferSink().latest("nope", default="---") == "---"


# Sinks

def test_the_json_sink_writes_one_object_per_line(tmp_path):
    """JSON Lines, not one array: a run that is killed half-way still leaves
    a readable file, and an array would need a closing bracket it never
    writes."""
    import json

    path = tmp_path / "t.jsonl"
    sink = JSONFileSink(str(path))
    sink.record({"a": 1})
    sink.record({"a": 2})
    sink.close()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["a"] for line in lines] == [1, 2]


def test_a_failing_sink_does_not_break_the_run():
    """A full disk must not end the agent run. The measurement is lost; the
    work is not."""
    sink = JSONFileSink("/nowhere/at/all/t.jsonl")
    sink.record({"a": 1})               # must not raise
    sink.close()


def test_fanout_keeps_going_when_one_sink_fails():
    live = LiveBufferSink()

    class Bad(TelemetrySink):
        name = "bad"

        def record(self, sample):
            raise OSError("gone")

    FanOutSink(Bad(), live).record({"a": 1})
    assert live.latest("a") == 1


# The real sources

def test_the_process_source_reports_this_process():
    sample = ProcessMemorySource().sample()
    assert sample["rss_mb"] > 0


def test_the_agent_source_says_why_it_has_nothing():
    # Names the CONDITION, not the symptom: see the test further down.
    assert "/engine ovms" in AgentTraceSource().unavailable

    class NoTrace:
        last_trace = None

    assert "native loop" in AgentTraceSource(NoTrace()).unavailable


def test_the_agent_source_drops_unknown_values_rather_than_zeroing_them():
    """Absent stays absent. A zero would read as "used no tokens", which is a
    different claim from "nobody told us"."""
    class Agent:
        last_trace = {"totals": {"turns": 2, "prompt_tokens": None,
                                 "tool_calls": 1}}

    assert AgentTraceSource(Agent()).sample() == {"turns": 2, "tool_calls": 1}


def test_intel_telemetry_is_honest_about_macos():
    """It ships as .exe/.dll with a Linux build. On a Mac, sampling zeros
    would be indistinguishable from an idle NPU."""
    import sys

    source = IntelHardwareSource()
    if sys.platform == "darwin":
        assert "macOS" in source.unavailable
    # Wherever we are, an unavailable source samples nothing rather than lying.
    if source.unavailable:
        assert source.sample() == {}


def test_find_ut_reports_how_it_looked():
    """Same shape as find_ovms: the caller needs to tell the user WHERE it
    looked, not just that it failed."""
    path, how = find_ut("/definitely/not/here")
    assert path is None
    assert "/definitely/not/here" in how


# The page was nearly empty on a machine without Intel UT

def test_the_system_source_reports_many_metrics_not_one():
    """The page had ONE graph on any machine without Intel hardware, which
    reads as broken rather than as limited. psutil works on macOS, Windows
    and Linux alike, so these are available everywhere and the Intel rows
    become the extra rather than the only content."""
    from ovat.telemetry.sources import SystemSource

    sample = SystemSource().sample()
    assert len(sample) > 5, f"only got {sorted(sample)}"
    assert "cpu_pct" in sample
    assert "ram_used_pct" in sample


def test_per_core_cpu_is_reported_individually():
    """An agent pinning one core looks identical to an idle machine in an
    averaged figure, and which of those is happening is the whole question
    when a run feels slow."""
    from ovat.telemetry.sources import SystemSource

    sample = SystemSource().sample()
    assert any(k.startswith("cpu0") for k in sample)


def test_intel_metric_names_are_normalised_for_use_as_ids():
    """UT writes "Render %" and "GPU-Freq(MHz)", and metric names become CSS
    ids on the page, where those characters are invalid. A build that renames
    a field must degrade to a missing metric, not a crash."""
    from ovat.telemetry.sources import IntelHardwareSource

    out = IntelHardwareSource._normalise(
        {"GPU": {"Render %": 42.0, "Freq(MHz)": 1300}, "Power W": 3.1})
    assert out == {"gpu_render": 42.0, "gpu_freq_mhz": 1300, "power_w": 3.1}
    for key in out:
        assert key.replace("_", "").isalnum(), f"{key} is not id-safe"


def test_nested_and_flat_ut_output_both_parse():
    """UT's JSON has varied between beta builds: nested per-collector objects
    in some, flat in others."""
    from ovat.telemetry.sources import IntelHardwareSource

    flat = IntelHardwareSource._normalise({"npu_utilization": 55})
    nested = IntelHardwareSource._normalise({"NPU": {"Utilization": 55}})
    assert flat == {"npu_utilization": 55}
    assert nested == {"npu_utilization": 55}


def test_booleans_are_not_mistaken_for_measurements():
    """bool is an int in Python, so a flag would otherwise be graphed."""
    from ovat.telemetry.sources import IntelHardwareSource

    assert IntelHardwareSource._normalise({"enabled": True, "watts": 2.5}) == \
        {"watts": 2.5}


# Finding a UT release that was unzipped into a versioned folder

def test_find_ut_looks_one_level_into_known_folders(tmp_path, monkeypatch):
    """The zip extracts to ut-tool-ext-v0.2.0-beta1.1, so unzipping into ~/ut
    leaves the binary at ~/ut/ut-tool-ext-v0.2.0-beta1.1/bin/ut.exe. A flat
    check of the known folders misses that entirely, which is exactly what
    happened on the AI PC."""
    import sys

    from ovat.telemetry import sources

    release = tmp_path / "ut-tool-ext-v0.2.0-beta1.1" / "bin"
    release.mkdir(parents=True)
    (release / sources._UT_EXE).write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(sources, "_UT_KNOWN_DIRS", [str(tmp_path)])
    monkeypatch.delenv("OVAT_UT", raising=False)
    monkeypatch.setattr(sources.shutil, "which", lambda name: None)

    path, how = sources.find_ut()
    assert path is not None, how
    assert "ut-tool-ext" in how


def test_the_newest_release_wins_when_several_are_unzipped(tmp_path,
                                                           monkeypatch):
    """Two versions side by side must not pick at random."""
    from ovat.telemetry import sources

    for version in ("ut-tool-ext-v0.1.0", "ut-tool-ext-v0.2.0"):
        binary = tmp_path / version / "bin"
        binary.mkdir(parents=True)
        (binary / sources._UT_EXE).write_text("x", encoding="utf-8")

    monkeypatch.setattr(sources, "_UT_KNOWN_DIRS", [str(tmp_path)])
    monkeypatch.delenv("OVAT_UT", raising=False)
    monkeypatch.setattr(sources.shutil, "which", lambda name: None)

    path, how = sources.find_ut()
    assert "v0.2.0" in how, how


def test_an_explicit_path_still_wins_over_everything(tmp_path, monkeypatch):
    """The user must always be able to override autodetection."""
    from ovat.telemetry import sources

    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / sources._UT_EXE).write_text("x", encoding="utf-8")

    path, how = sources.find_ut(str(tmp_path))
    assert path is not None
    assert how == "config"


def test_the_sources_table_is_redrawn_every_tick_not_once():
    """Source availability CHANGES while the page is open: an agent appears
    the moment /engine ovms finishes building one. Drawn once at mount, the
    table said "n/a" forever even while that agent was answering."""
    import inspect

    from ovat.cli import telemetry_screen

    redraw = inspect.getsource(telemetry_screen.TelemetryScreen._redraw)
    assert "_fill_sources_table" in redraw, (
        "the table is not refreshed on the redraw tick")


def test_live_mode_returns_a_table_rather_than_printing_it():
    """rich.Live needs a renderable it can redraw IN PLACE. Printing a fresh
    table every tick scrolled the terminal unreadable within seconds and
    buried the numbers it was meant to show."""
    from ovat.cli.main import _telemetry_table

    table = _telemetry_table({"system.cpu_pct": 12.5})
    assert table is not None
    assert hasattr(table, "columns")          # a rich Table, not None


def test_an_empty_sample_still_renders_a_table():
    """Live mode draws before the first sample arrives; returning None there
    would crash the display on startup."""
    from ovat.cli.main import _telemetry_table

    assert _telemetry_table({}) is not None


def test_the_agent_source_says_what_to_DO_not_just_that_it_is_empty():
    """"no agent has run yet" reads as a bug when a chat is plainly answering
    two screens away. Only /engine ovms builds an agent at all, and the local
    engine is retrieval rather than an agent loop, so the message names the
    condition instead of the symptom."""
    from ovat.telemetry.sources import AgentTraceSource

    message = AgentTraceSource().unavailable
    assert "/engine ovms" in message
    assert "retrieval" in message


def test_a_ut_text_line_becomes_a_metric():
    """Measured on the AI PC: continuous mode writes binary traces rather
    than streaming JSON, so a text fallback is what actually gets read."""
    from ovat.telemetry.sources import IntelHardwareSource as I

    assert I._parse_text_line("NPU Utilization: 42.5") == {
        "npu_utilization": 42.5}
    assert I._parse_text_line("Power W = 3.1 watts") == {"power_w": 3.1}
    assert I._parse_text_line("collecting...") == {}


def test_intel_ut_is_told_where_to_write_and_cleans_it_up(monkeypatch, tmp_path):
    """ut dumped binary traces into the user's project folder, forever.

    start() created a temp FILE and never passed it on the command line, so ut
    ignored it and wrote its own ut_default_output.*.bin (tens of MB) into the
    current directory. stop() then deleted the untouched temp file and left the
    real output behind. Every `ovat run --telemetry` added to the pile.
    """
    import shutil
    from ovat.telemetry.sources import IntelHardwareSource

    class FakeProc:
        stdout = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("ovat.telemetry.sources.subprocess.Popen", fake_popen)
    # unavailable is a computed property that short-circuits on macOS, so the
    # platform is faked: this test is about the command line and the cleanup,
    # not about discovery or which OS ut supports.
    monkeypatch.setattr("ovat.telemetry.sources.sys.platform", "win32")
    source = IntelHardwareSource()
    source.binary = str(tmp_path / "ut")
    assert source.unavailable is None, "fixture did not make ut look present"
    source.start()

    cmd = captured["cmd"]
    assert "--output" in cmd, "ut was never told where to write"
    out_dir = cmd[cmd.index("--output") + 1]
    assert os.path.isdir(out_dir), "--output must name a real directory"

    # Something ut-shaped lands in there; stop() must take the folder with it.
    with open(os.path.join(out_dir, "ut_default_output.l0_gpu.bin"), "wb") as f:
        f.write(b"\x00" * 32)
    source.stop()
    assert not os.path.exists(out_dir), "the trace directory was left behind"
