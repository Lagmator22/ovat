# tests/test_telemetry.py
"""Tests for the telemetry contracts, sources, sinks and collector.

None of these need hardware: the point of the two ABCs is that a fake source
and a fake sink exercise the whole path in milliseconds.
"""
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
    assert "no agent" in AgentTraceSource().unavailable

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
