# ovat/cli/telemetry_screen.py
"""The telemetry screen: live numbers and graphs, inside the TUI.

Same shape as doctor_screen.py, and for the same reason: the CLI can already
write a trace to a file, but a file is not something you WATCH. This is the
layer above the CLI the TUI exists to be.

Three things are load-bearing here, each a bug avoided:

  * Sampling does NOT happen on the UI thread. Collector runs its own daemon
    thread and this screen only reads the buffer it fills, so a stalled
    hardware collector cannot freeze the interface.
  * Sources that cannot run HERE are shown with the reason, not hidden. An
    absent source and an idle one look identical in a graph, and only one of
    them is worth acting on. On a Mac that row reads "Intel UT does not run
    on macOS", which is the honest answer rather than a flat line at zero.
  * Every colour comes from ovat.cli.ui. This screen owns no palette.
"""
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (DataTable, Digits, Label, ProgressBar,
                             Rule, Static, TabbedContent, TabPane)

from ovat.cli import ui
from ovat.cli.commands import ScreenCommands
from ovat.cli.widgets import Footer
from ovat.telemetry.collector import Collector
from ovat.telemetry.sinks import LiveBufferSink
from ovat.telemetry.sources import (IntelHardwareSource, NPUSource,
                                    OVMSLogSource, ProcessMemorySource,
                                    SystemSource)

# How often the page redraws. 2/second is enough to look live without making
# a terminal over SSH repaint faster than the link can carry, which is how
# the AI PC is actually driven.
REFRESH_HZ = 2.0

# The metrics given their own big-number readout, in order. Anything else a
# source reports still appears in the table below; this is just what gets the
# headline treatment.
# metric key, label, unit, and whether it is a 0-100 percentage (those get a
# bar as well as a number, because a percentage has a meaningful maximum and a
# bare figure does not show how close to it you are).
# Preferred headline metrics, best first. The page shows the first MAX_CARDS
# of these that ACTUALLY HAVE DATA, so a machine without Intel UT gets CPU and
# RAM in those slots instead of three boxes reading "---" forever. A fixed
# list was the reason the page looked broken rather than merely limited.
_PREFERRED = [
    # npu.* comes from the driver's own busy counter and is a real percentage.
    # intel.* comes from UT, whose continuous mode on the AI PC reports power,
    # NPU bandwidth and GPU frequency but no utilisation percentage, so the
    # driver's figure stays first for the NPU card.
    ("npu.utilization", "NPU", "%"),
    # The cache figure earns a headline slot because it is the one number that
    # predicts the undecoded-tool-call failure before it happens.
    ("ovms.kv_cache_pct", "KV CACHE", "%"),
    ("intel.npu_utilization", "NPU", "%"),
    ("intel.gpu_utilization", "GPU", "%"),
    ("intel.pkg_power_w", "POWER", "W"),
    ("system.cpu_pct", "SYS CPU", "%"),
    ("system.ram_used_pct", "SYS RAM", "%"),
    ("process.rss_mb", "OVAT MEM", "MB"),
    ("system.proc_cpu_pct", "OVAT CPU", "%"),
    ("system.threads", "THREADS", ""),
]
MAX_CARDS = 5

#: The figure columns, in order. "now" first because it is the one being read;
#: the rest give it a scale, which is the job the sparkline used to do badly.
_COLUMNS = ("now", "min", "max", "mean")

# ONE short help, on its own scrollable tab. The page used to carry three
# long ones (Sources, Intel, Plano) that did not scroll, so the end of each
# was unreadable without zooming the terminal out, and four tabs left the
# owner unsure which one mattered. The plano walkthrough lives in
# examples/plano/README.md, where it can be followed step by step.
_HELP = """
[b]What the numbers are[/b]
  [cyan]system[/cyan]   this computer: CPU per core, RAM, clock
  [cyan]process[/cyan]  OVAT itself (with OVMS serving, the model is not here)
  [cyan]npu[/cyan]      NPU busy %, from the driver (Windows and Linux)
  [cyan]ovms[/cyan]     the server's KV cache, read from ovms.log
  [cyan]intel[/cyan]    power and NPU bandwidth, from Intel Unified Telemetry

[b]The line under the table[/b] says, for each source, whether it is
[green]live[/green], [yellow]silent[/yellow] (running, nothing read yet) or
[yellow]n/a[/yellow] (cannot run here), and why.

[b]Turning on the intel rows[/b] (Windows and Linux)
  Unzip Intel's ut-tool, then set [cyan]OVAT_UT[/cyan] to that folder or put it
  in [cyan]~/ut[/cyan]. On Windows run [cyan]ut-vars.cmd[/cyan] from an
  Administrator prompt first. The first reading takes a few seconds.

[b]Per-request traces[/b] come from the plano gateway instead:
  see [cyan]examples/plano/README.md[/cyan].
"""


class TelemetryCommands(ScreenCommands):
    """Palette entries for this screen only."""

    def commands(self) -> list:
        screen = self.screen
        return [
            ("Clear telemetry history", "empty the graphs and start again",
             screen.action_clear),
            ("Copy telemetry as JSON", "the buffered samples, for a report",
             screen.action_copy),
        ]


class TelemetryScreen(Screen):
    """Live numbers and graphs from every telemetry source that works here."""

    COMMANDS = {TelemetryCommands}

    DEFAULT_CSS = """
    TelemetryScreen { background: $background; }
    #tel-header { padding: 1 2 0 2; height: auto; }
    #tel-numbers { height: 8; margin: 1 2 0 2; }
    .tel-metric { color: $accent; margin: 1 0 0 0; }
    #tel-rule { margin: 0 2; color: $primary 40%; }
    .tel-card {
        width: 1fr;
        border: round $primary;
        background: $surface;
        padding: 0 1;
        content-align: center middle;
    }
    .tel-card Digits { color: $success; }
    .tel-card.-dead { border: round $secondary; }
    .tel-card.-dead Digits { color: $text-muted; }
    .tel-card ProgressBar { width: 100%; }
    .tel-card Bar > .bar--bar { color: $success; }
    #tel-live-table { height: 1fr; border: round $primary; margin: 1 2 0 2; }
    #tel-sources { height: auto; margin: 0 2 1 2; }
    #tel-help { padding: 1 2; color: $text-muted; }
    #tel-tabs { height: 1fr; }
    Footer { background: $surface; color: $accent; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("f5", "clear", "Clear", show=True),
    ]

    def __init__(self, ut_binary: str | None = None):
        super().__init__()
        self.live = LiveBufferSink()
        # No agent source. It could only ever say "n/a" here: an agent exists
        # solely inside the chat screen's OVMS engine, only the native loop
        # fills last_trace at all, and a row that is permanently unavailable
        # teaches the reader to ignore the whole table. Per-run agent numbers
        # live in `ovat run --trace`, which is where they belong.
        self.collector = Collector(
            [SystemSource(),
             ProcessMemorySource(),
             # The driver's busy counter, which is an actual live percentage.
             # UT stays alongside it for power and per-engine detail; the two
             # answer different questions and neither replaces the other.
             NPUSource(),
             # KV cache usage, read from the log OVMS writes. The only
             # interface that carries it: OVMS documents that text-generation
             # metrics are not on its /metrics endpoint.
             OVMSLogSource(),
             IntelHardwareSource(ut_binary)],
            self.live, interval_s=1.0 / REFRESH_HZ)

    def compose(self) -> ComposeResult:
        header = ui.wordmark("TELEMETRY") if hasattr(ui, "wordmark") else None
        yield Static(header or "OVAT telemetry", id="tel-header")
        # Three tabs, switchable by mouse or arrow keys. One page rather
        # than three screens: these are three VIEWS of the same machine, and
        # making them separate screens would mean losing the live buffer
        # every time you looked at a different one.
        with TabbedContent(id="tel-tabs"):
            with TabPane("Live", id="tab-live"):
                # Cards are mounted once we know which metrics have data; an
                # empty row now beats five boxes reading "---" forever.
                yield Horizontal(id="tel-numbers")
                yield Rule(id="tel-rule")
                # A TABLE OF NUMBERS, not a wall of sparklines. Every core and
                # every device gets a row with its current value beside the
                # min, max and mean over the buffer. A sparkline shows that
                # something moved; it cannot answer "is core 6 pinned right
                # now, and how hard" -- which is the actual question when a run
                # feels slow, and the reason per-core sampling exists at all.
                table = DataTable(id="tel-live-table", cursor_type="row",
                                  zebra_stripes=True)
                table.border_title = "live numbers"
                yield table
                # Where each number comes from, refreshed every tick. It used
                # to be a table on its own tab, filled ONCE before any source
                # had started: on the AI PC it said "intel live, sampling"
                # while UT produced nothing, so the page hid the one fact
                # that explained the missing rows.
                yield Static("", id="tel-sources")
            with TabPane("Help", id="tab-help"):
                with VerticalScroll(id="tel-help-scroll"):
                    yield Static(_HELP, id="tel-help")
        yield Footer()

    def on_mount(self) -> None:
        self._cards: dict = {}
        self._rows: set = set()
        numbers = self.query_one("#tel-live-table", DataTable)
        numbers.add_column("Source", key="source")
        numbers.add_column("Metric", key="metric")
        for column in _COLUMNS:
            numbers.add_column(column, key=column)
        self.collector.start()
        # set_interval, not a worker loop: the sampling already happens on
        # the collector's own thread, so all this does is redraw.
        self.set_interval(1.0 / REFRESH_HZ, self._redraw)

    def on_unmount(self) -> None:
        # Leaving the screen must stop the subprocess a hardware source may
        # have launched, or it outlives the page that started it.
        self.collector.stop()

    # ---- drawing -----------------------------------------------------------

    def _sources_line(self) -> Text:
        """One line: each source, and whether it works here right now."""
        unavailable = self.collector.unavailable
        line = Text()
        for source in self.collector.sources:
            reason = unavailable.get(source.name)
            note = None if reason else getattr(source, "note", None)
            if line:
                line.append("   ")
            line.append(source.name, style=ui.CYAN)
            if reason:
                line.append(f" n/a ({_first_clause(reason)})", style=ui.YELLOW)
            elif note:
                line.append(f" silent ({_first_clause(note)})", style=ui.YELLOW)
            else:
                line.append(" live", style=f"bold {ui.GREEN}")
        return line

    def _redraw(self) -> None:
        self.query_one("#tel-sources", Static).update(self._sources_line())
        self._sync_cards()
        for metric, digits, bar in self._cards.values():
            value = self.live.latest(metric)
            if value is None:
                digits.update("---")
                continue
            digits.update(f"{value:.0f}" if abs(value) >= 10
                          else f"{value:.1f}")
            if bar is not None:
                bar.update(progress=max(0.0, min(100.0, float(value))))
        self._redraw_numbers()

    def _sync_cards(self) -> None:
        """Mount headline cards for the preferred metrics that have data.

        Done here rather than in compose because which metrics exist depends
        on which SOURCES work on this machine, and that is not known until
        the collector has ticked at least once.
        """
        if len(self._cards) >= MAX_CARDS:
            return
        row = self.query_one("#tel-numbers", Horizontal)
        for metric, label, unit in _PREFERRED:
            if len(self._cards) >= MAX_CARDS:
                break
            if metric in self._cards or self.live.latest(metric) is None:
                continue
            # BUILT FIRST, MOUNTED ONCE. The obvious shape -- mount the card,
            # then mount its children into it -- has a window in it: mount()
            # is asynchronous, so `card.parent` may still be unset on the very
            # next line, and Textual then raises
            #
            #   MountError: Unable to find relative location of Vertical(...)
            #   because it has no parent
            #
            # Seen once on the AI PC as card-npu-utilization and never
            # reproduced -- 10/10 isolated, 47/47 with a server up, clean in
            # two full suites -- which is exactly what a timing window looks
            # like from the outside. Passing the children to the constructor
            # removes the window rather than narrowing it, and is the pattern
            # Textual documents for building a widget before it is displayed.
            digits = Digits("---")
            children = [Label(f"{label}  [dim]{unit}[/dim]", markup=True),
                        digits]
            bar = None
            if unit == "%":
                # A percentage has a meaningful maximum; a bare number does
                # not show how close to it you are.
                bar = ProgressBar(total=100, show_eta=False,
                                  show_percentage=False)
                children.append(bar)
            row.mount(Vertical(*children, classes="tel-card",
                               id=_card_id(metric)))
            self._cards[metric] = (metric, digits, bar)

    def _redraw_numbers(self) -> None:
        """One row of figures per metric: now, min, max, mean.

        Built from what the buffer HAS rather than a fixed list, so a source
        added later draws itself with no change here, and a machine missing a
        source simply has fewer rows instead of empty ones.

        Rows are added once and then UPDATED in place. Clearing and refilling
        twice a second would work, but it resets the cursor and the scroll
        position on every tick, so the table could not be read on a machine
        with sixteen cores -- which is precisely the machine it is for.
        """
        from rich.text import Text

        table = self.query_one("#tel-live-table", DataTable)
        for metric in self.live.metrics():
            data = self.live.series(metric)
            if not data:
                continue
            row_key = _row_id(metric)
            cells = (_fmt(data[-1]), _fmt(min(data)), _fmt(max(data)),
                     _fmt(sum(data) / len(data)))
            if row_key in self._rows:
                for column, value in zip(_COLUMNS, cells):
                    table.update_cell(row_key, column, value)
                continue
            source, _, short = metric.partition(".")
            table.add_row(Text(source, style=ui.DIM),
                          Text(short, style=ui.CYAN), *cells, key=row_key)
            self._rows.add(row_key)

    # ---- actions -----------------------------------------------------------

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_clear(self) -> None:
        self.live.samples.clear()
        # Rows are keyed by metric and updated in place, so the row bookkeeping
        # has to be cleared alongside the buffer or the next tick tries to
        # update cells in a table that no longer has them.
        self.query_one("#tel-live-table", DataTable).clear()
        self._rows.clear()

    def action_copy(self) -> None:
        import json

        self.app.copy_to_clipboard(json.dumps(list(self.live.samples),
                                              indent=2))
        self.notify("Telemetry copied as JSON.")


def _first_clause(reason: str) -> str:
    """The head of a reason, for a one-line status. The full sentences ("no
    OVMS log at ovms.log. This source reads the log `ovat serve` writes;
    start a server first") wrapped the line over three rows."""
    return reason.split(". ")[0].split("; ")[0].rstrip(".")


def _card_id(metric: str) -> str:
    return "card-" + metric.replace(".", "-").replace("_", "-")


def _row_id(metric: str) -> str:
    """A stable DataTable row key. Not a widget id, so the metric name is
    usable as-is; keeping the same shape as _card_id anyway avoids a reader
    wondering whether the difference is meaningful."""
    return "row-" + metric.replace(".", "-").replace("_", "-")


def _fmt(value: float) -> str:
    """A figure sized to what it is.

    A CPU percentage wants one decimal; 4300 MB of RSS does not want ".0"
    stapled to it, and a thread count is an integer that should look like one.
    """
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.1f}"
