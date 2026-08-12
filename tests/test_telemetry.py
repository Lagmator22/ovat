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

    source = SystemSource()
    source.sample()                  # first CPU reading is a priming artefact
    sample = source.sample()
    assert len(sample) > 5, f"only got {sorted(sample)}"
    assert "cpu_pct" in sample
    assert "ram_used_pct" in sample


def test_per_core_cpu_is_reported_individually():
    """An agent pinning one core looks identical to an idle machine in an
    averaged figure, and which of those is happening is the whole question
    when a run feels slow."""
    from ovat.telemetry.sources import SystemSource

    source = SystemSource()
    source.sample()                  # see the priming test below
    sample = source.sample()
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
    # CONSTRUCT FIRST, then fake the platform. `sources.sys` IS the sys module,
    # so setting sys.platform here is global: __init__ calls find_ut(), which
    # calls shutil.which(), which then takes its WINDOWS branch on Linux and
    # dies with
    #   AttributeError: 'NoneType' object has no attribute
    #   'NeedCurrentDirectoryForExePath'
    # because _winapi is None off Windows. The test failed on Linux while the
    # product was fine. Nothing after construction touches shutil, so doing
    # discovery on the real platform and faking only what `unavailable` reads
    # keeps the fake as narrow as the thing it is standing in for.
    source = IntelHardwareSource()
    source.binary = str(tmp_path / "ut")
    # unavailable is a computed property that short-circuits on macOS, so the
    # platform is faked: this test is about the command line and the cleanup,
    # not about discovery or which OS ut supports.
    monkeypatch.setattr("ovat.telemetry.sources.sys.platform", "win32")
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


def test_a_source_that_is_live_but_silent_is_reported():
    """Three states, not two: unavailable, live-and-producing, live-and-silent.

    Intel UT on a build whose continuous mode writes binary traces starts
    fine and emits nothing. Reported from an AI PC: no GPU, NPU or power rows
    and no line saying why, which is indistinguishable from idle hardware.
    """
    from ovat.telemetry.collector import Collector

    class Silent:
        name = "intel"
        unavailable = None                      # it DID start
        note = "writes binary traces; decode with bin2perfetto"

        def sample(self):
            return {}

    class Fine:
        name = "system"
        unavailable = None
        note = None

        def sample(self):
            return {"cpu_pct": 3.0}

    collector = Collector([Silent(), Fine()], sink=None)
    assert collector.notes == {
        "intel": "writes binary traces; decode with bin2perfetto"}
    # a healthy source contributes no note, and an explained-elsewhere one
    # must not be reported twice
    assert "system" not in collector.notes


# --- NPUSource -------------------------------------------------------------
#
# Driven against a FAKE sysfs tree, so these run on macOS and Windows CI as
# well as on hardware. A test that needs a real NPU is a test that never runs,
# and this reader's arithmetic is exactly where a mistake would hide.

#: The real device directory is a PCI address -- "0000:00:0b.0" -- and those
#: colons cannot appear in a Windows path, so a fixture using the true name
#: fails on Windows with WinError 123 while passing everywhere else. The
#: source does not care what the directory is CALLED: it lists whatever is
#: there and keeps the entries carrying npu_busy_time_us. So the fixture uses
#: a portable stand-in and the reader is still exercised exactly as written.
_FAKE_DEVICE = "0000_00_0b.0"


def _fake_npu(tmp_path, busy_us, memory_kb=None):
    """A directory shaped like /sys/bus/pci/drivers/intel_vpu."""
    device = tmp_path / _FAKE_DEVICE
    device.mkdir(parents=True, exist_ok=True)
    (device / "npu_busy_time_us").write_text(str(busy_us))
    if memory_kb is not None:
        (device / "npu_memory_utilization").write_text(str(memory_kb))
    return str(tmp_path)


def test_the_first_npu_sample_reports_no_utilization_rather_than_zero(tmp_path):
    """One reading of a monotonic counter cannot be a percentage.

    Reporting 0.0 here would draw an idle NPU for the first second of every
    run, which is the second an agent is most likely to be using it. Same rule
    as the token counts: absent is not zero.
    """
    from ovat.telemetry.sources import NPUSource

    source = NPUSource(sysfs_root=_fake_npu(tmp_path, busy_us=1_000_000))
    assert "utilization" not in source.sample()


def test_npu_utilization_is_the_duty_cycle_between_two_readings(tmp_path,
                                                                monkeypatch):
    """busy% = delta_busy_us / (delta_wall_s * 10_000).

    Half a second of busy time across one second of wall clock is 50%.
    """
    from ovat.telemetry import sources as telemetry_sources
    from ovat.telemetry.sources import NPUSource

    root = _fake_npu(tmp_path, busy_us=1_000_000)
    clock = [100.0]
    monkeypatch.setattr(telemetry_sources.time, "monotonic", lambda: clock[0])

    source = NPUSource(sysfs_root=root)
    source.sample()                                    # primes the delta

    clock[0] = 101.0                                   # one second later
    (tmp_path / _FAKE_DEVICE / "npu_busy_time_us").write_text("1500000")
    assert source.sample()["utilization"] == 50.0


def test_a_counter_that_went_backwards_is_not_reported_as_negative(tmp_path,
                                                                   monkeypatch):
    """A driver reload restarts the counter. That is a gap, not -400%."""
    from ovat.telemetry import sources as telemetry_sources
    from ovat.telemetry.sources import NPUSource

    root = _fake_npu(tmp_path, busy_us=5_000_000)
    clock = [10.0]
    monkeypatch.setattr(telemetry_sources.time, "monotonic", lambda: clock[0])

    source = NPUSource(sysfs_root=root)
    source.sample()
    clock[0] = 11.0
    (tmp_path / _FAKE_DEVICE / "npu_busy_time_us").write_text("12000")
    assert "utilization" not in source.sample()


def test_npu_memory_is_reported_when_the_driver_publishes_it(tmp_path):
    from ovat.telemetry.sources import NPUSource

    root = _fake_npu(tmp_path, busy_us=1, memory_kb=204800)
    assert NPUSource(sysfs_root=root).sample()["memory_mb"] == 200.0


def test_an_absent_npu_says_where_it_looked(tmp_path):
    """Same contract as every other source: a reason, not an empty table."""
    from ovat.telemetry.sources import NPUSource

    source = NPUSource(sysfs_root=str(tmp_path / "nothing-here"))
    reason = source.unavailable
    assert reason is not None
    # macOS and Windows answer with their own platform reason; Linux names
    # the path and the driver. All three are actionable, which is the point.
    assert ("nothing-here" in reason or "macOS" in reason
            or "DXCore" in reason)
    assert source.sample() == {}


# --- the live NUMBERS table ------------------------------------------------
#
# Driven through Textual's headless Pilot, same as the other TUI tests. The
# page used to draw a sparkline per metric, which shows that something moved
# and cannot answer "is core 6 pinned, and how hard". These pin the numbers.

def test_the_live_tab_shows_a_figure_per_metric_not_a_sparkline():
    """Every metric in the buffer gets a row of numbers: now, min, max, mean.

    Back the change out -- put the Sparkline loop back -- and this fails:
    there is no #tel-live-table to query.
    """
    import asyncio

    pytest.importorskip("textual")
    from textual.app import App
    from textual.widgets import DataTable

    from ovat.cli.telemetry_screen import TelemetryScreen

    async def scenario():
        class Harness(App):
            def on_mount(self):
                self.push_screen(TelemetryScreen())

        app = Harness()
        async with app.run_test() as pilot:
            screen = app.screen
            # Stop the real collector first: it samples this machine twice a
            # second into the same buffer, so without this the assertions
            # measure the developer's CPU instead of the rendering.
            screen.collector.stop()
            screen.live.samples.clear()
            # Two ticks of one fake source: enough for a min that differs
            # from a max, so the columns are visibly doing their job.
            screen.live.record({"system.cpu0_pct": 4.0, "npu.utilization": 90.0})
            screen.live.record({"system.cpu0_pct": 96.0, "npu.utilization": 10.0})
            screen._redraw_numbers()
            await pilot.pause()

            table = screen.query_one("#tel-live-table", DataTable)
            rendered = {
                str(table.get_cell_at((row, 1))): [
                    str(table.get_cell_at((row, column)))
                    for column in range(2, 6)
                ]
                for row in range(table.row_count)
            }
            # per-core CPU is present AS A NUMBER, and so is the NPU
            assert "cpu0_pct" in rendered
            assert "utilization" in rendered
            # now, min, max, mean -- the scale a bare figure does not carry
            assert rendered["cpu0_pct"] == ["96", "4", "96", "50"]
    asyncio.run(scenario())


def test_the_numbers_table_updates_in_place_rather_than_refilling():
    """Refilling twice a second resets cursor and scroll, so a 16-core table
    could not be read -- which is the machine it exists for."""
    import asyncio

    pytest.importorskip("textual")
    from textual.app import App
    from textual.widgets import DataTable

    from ovat.cli.telemetry_screen import TelemetryScreen

    async def scenario():
        class Harness(App):
            def on_mount(self):
                self.push_screen(TelemetryScreen())

        app = Harness()
        async with app.run_test() as pilot:
            screen = app.screen
            screen.collector.stop()
            screen.live.samples.clear()
            screen.live.record({"system.cpu0_pct": 10.0})
            screen._redraw_numbers()
            await pilot.pause()
            table = screen.query_one("#tel-live-table", DataTable)
            before = table.row_count

            screen.live.record({"system.cpu0_pct": 20.0})
            screen._redraw_numbers()
            await pilot.pause()

            assert table.row_count == before          # updated, not appended
            assert str(table.get_cell_at((0, 2))) == "20"
    asyncio.run(scenario())


def test_clearing_the_page_forgets_the_rows_as_well_as_the_buffer():
    """Rows are keyed by metric and updated in place, so clearing the buffer
    without clearing the bookkeeping makes the next tick update cells in a
    table that no longer has them."""
    import asyncio

    pytest.importorskip("textual")
    from textual.app import App
    from textual.widgets import DataTable

    from ovat.cli.telemetry_screen import TelemetryScreen

    async def scenario():
        class Harness(App):
            def on_mount(self):
                self.push_screen(TelemetryScreen())

        app = Harness()
        async with app.run_test() as pilot:
            screen = app.screen
            screen.collector.stop()
            screen.live.samples.clear()
            screen.live.record({"system.cpu0_pct": 10.0})
            screen._redraw_numbers()
            await pilot.pause()

            screen.action_clear()
            await pilot.pause()
            assert screen.query_one("#tel-live-table", DataTable).row_count == 0

            # and the next tick must repopulate rather than raise
            screen.live.record({"system.cpu0_pct": 30.0})
            screen._redraw_numbers()
            await pilot.pause()
            assert screen.query_one("#tel-live-table", DataTable).row_count == 1
    asyncio.run(scenario())


# --- OVMSLogSource: the KV cache figure ------------------------------------
#
# The KV-cache explanation for undecoded tool calls rests on a number that,
# until now, had to be read by hand out of a log at the moment of failure.
# These pin the reader so the figure can be recorded automatically instead.

_CACHE_LINE = "[LLM] type: dynamic, cache usage: {}% of {}"


def _ovms_log(tmp_path, *lines):
    path = tmp_path / "ovms.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_the_kv_cache_figure_is_read_from_the_ovms_log(tmp_path):
    """The format is OVMS's own, from its continuous-batching executor."""
    from ovat.telemetry.sources import OVMSLogSource

    log = _ovms_log(tmp_path, "starting", _CACHE_LINE.format("98.7", "3.60 GB"))
    assert OVMSLogSource(log).sample() == {"kv_cache_pct": 98.7,
                                           "kv_cache_gb": 3.6}


def test_the_newest_cache_reading_wins(tmp_path):
    """OVMS logs one line per scheduler tick. The current state is the last
    one, not the first -- reporting the first would show a cache that filled
    an hour ago as though it were empty."""
    from ovat.telemetry.sources import OVMSLogSource

    log = _ovms_log(tmp_path,
                    _CACHE_LINE.format("2.0", "3.60 GB"),
                    _CACHE_LINE.format("55.0", "3.60 GB"),
                    _CACHE_LINE.format("99.9", "3.60 GB"))
    assert OVMSLogSource(log).sample()["kv_cache_pct"] == 99.9


def test_a_static_cache_line_is_read_the_same_way(tmp_path):
    """OVMS writes "type: static" when the cache is not grown dynamically.
    The usage figure is in the same place and means the same thing."""
    from ovat.telemetry.sources import OVMSLogSource

    log = _ovms_log(tmp_path, "type: static, cache usage: 41.0% of 8.00 GB")
    assert OVMSLogSource(log).sample()["kv_cache_pct"] == 41.0


def test_cache_sizes_are_normalised_to_gigabytes(tmp_path):
    """So "98% of 900 MB" and "98% of 16 GB" are not one situation."""
    from ovat.telemetry.sources import OVMSLogSource

    log = _ovms_log(tmp_path, _CACHE_LINE.format("10.0", "512 MB"))
    assert OVMSLogSource(log).sample()["kv_cache_gb"] == 0.5


def test_only_the_tail_of_a_long_log_is_read(tmp_path):
    """This log grows for as long as the server runs, and the source is polled
    twice a second. Reading it whole would be O(file) forever."""
    from ovat.telemetry.sources import OVMSLogSource

    filler = ["noise " * 20] * 20_000
    log = _ovms_log(tmp_path, *filler, _CACHE_LINE.format("77.7", "3.60 GB"))
    source = OVMSLogSource(log)
    assert os.path.getsize(log) > source.TAIL_BYTES     # genuinely long
    assert source.sample()["kv_cache_pct"] == 77.7


def test_no_server_is_reported_as_unavailable_with_the_path(tmp_path):
    from ovat.telemetry.sources import OVMSLogSource

    source = OVMSLogSource(str(tmp_path / "absent.log"))
    assert "absent.log" in source.unavailable
    assert source.sample() == {}


def test_a_running_server_with_no_cache_line_yet_says_so(tmp_path):
    """Third state, and the one that would otherwise read as a bug: the server
    is up, but nothing has been generated, so no cache line exists yet."""
    from ovat.telemetry.sources import OVMSLogSource

    source = OVMSLogSource(_ovms_log(tmp_path, "Server started on port 8000"))
    assert source.unavailable is None            # it IS running
    assert "no KV cache line yet" in source.note
    assert source.sample() == {}


# --- the pre-run cache warning ---------------------------------------------

def test_a_healthy_cache_says_nothing():
    """A warning that fires routinely is a warning nobody reads."""
    from ovat.telemetry.sources import cache_warning

    assert cache_warning(12.0, 3.6) is None
    assert cache_warning(80.0, 3.6) is None


def test_no_warning_in_the_range_where_there_is_no_evidence():
    """94% is not measured. Warning there would be inventing a number."""
    from ovat.telemetry.sources import cache_warning

    assert cache_warning(94.9, 3.6) is None


def test_a_full_cache_warns_and_names_the_remedy():
    """The measured failure range. It must say what to DO."""
    from ovat.telemetry.sources import cache_warning

    warning = cache_warning(99.0, 3.6)
    assert warning is not None
    assert "99% of 3.6 GB" in warning
    assert "restart OVMS" in warning
    assert "ovms_cache_size_gb" in warning


def test_a_large_full_cache_does_not_suggest_making_it_larger():
    """Advice that scales with the reading. Telling someone with 16 GB already
    full to raise the number is not the useful half of the answer."""
    from ovat.telemetry.sources import cache_warning

    warning = cache_warning(99.0, 16.0)
    assert "restart OVMS" in warning
    assert "ovms_cache_size_gb" not in warning


def test_the_warning_does_not_claim_the_run_will_fail():
    """The link is still a correlation. Overstating it is how a hypothesis
    turns into folklore."""
    from ovat.telemetry.sources import cache_warning

    warning = cache_warning(100.0, 3.6).lower()
    assert "will fail" not in warning


def test_the_first_cpu_reading_is_skipped_rather_than_reported_as_zero():
    """psutil's percentages are deltas since the previous call, so the first
    one covers ~0 seconds and returns 0.0 for every core.

    Reporting it puts a zero in the buffer that pins the `min` column at 0
    forever: a table open for an hour still claims every core hit 0%. Same
    rule as the token counts -- absent is not zero.

    Back the guard out and the first sample carries cpu keys, which is the
    failure.
    """
    from ovat.telemetry.sources import SystemSource

    source = SystemSource()
    first = source.sample()
    assert not [key for key in first if key.startswith("cpu")
                and key.endswith("_pct")], (
        "the first CPU reading is a priming artefact, not a measurement")

    second = source.sample()
    assert "cpu_pct" in second
    assert any(key.startswith("cpu0") for key in second)


def test_memory_is_still_reported_on_the_very_first_sample():
    """Only the CPU deltas need priming. RAM is an absolute reading, so
    skipping it would lose a second of data for no reason."""
    from ovat.telemetry.sources import SystemSource

    assert "ram_used_pct" in SystemSource().sample()


def test_the_fake_npu_device_name_is_legal_on_every_platform():
    """Guard for the mistake this fixture already made once.

    The real directory is a PCI address, "0000:00:0b.0". Using that literally
    passed on macOS and Linux and failed the whole Windows CI job with
    WinError 123, because a colon cannot appear in a Windows path. The suite
    must describe the code, not the filesystem it happened to be written on.

    Restore the colon and this fails before the OSError does, naming why.
    """
    illegal = set('<>:"/\\|?*')
    assert not (set(_FAKE_DEVICE) & illegal), (
        f"{_FAKE_DEVICE!r} cannot be a directory name on Windows")
