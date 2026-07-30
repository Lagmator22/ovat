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
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Digits, Footer, Label, Sparkline, Static

from ovat.cli import ui
from ovat.cli.commands import ScreenCommands
from ovat.telemetry.collector import Collector
from ovat.telemetry.sinks import LiveBufferSink
from ovat.telemetry.sources import (AgentTraceSource, IntelHardwareSource,
                                    ProcessMemorySource)

# How often the page redraws. 2/second is enough to look live without making
# a terminal over SSH repaint faster than the link can carry, which is how
# the AI PC is actually driven.
REFRESH_HZ = 2.0

# The metrics given their own big-number readout, in order. Anything else a
# source reports still appears in the table below; this is just what gets the
# headline treatment.
_HEADLINE = [
    ("process.rss_mb", "MEMORY MB", "the toolkit's own footprint, not the "
                                    "model's: that lives in OVMS"),
    ("intel.npu_utilization", "NPU %", "Intel UT, AI PC only"),
    ("intel.power_w", "POWER W", "Intel UT, AI PC only"),
]


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
    #tel-numbers { height: 7; margin: 1 2 0 2; }
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
    #tel-graphs { height: 1fr; margin: 1 2 0 2; }
    #tel-graphs Sparkline { height: 3; margin: 0 0 1 0; }
    #tel-graphs > .sparkline--max-color { color: $success; }
    #tel-graphs > .sparkline--min-color { color: $primary; }
    #tel-table { height: 12; border: round $primary; margin: 0 2 1 2; }
    Footer { background: $surface; color: $accent; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("f5", "clear", "Clear", show=True),
    ]

    def __init__(self, agent=None, ut_binary: str | None = None):
        super().__init__()
        self.live = LiveBufferSink()
        self.collector = Collector(
            [ProcessMemorySource(),
             AgentTraceSource(agent),
             IntelHardwareSource(ut_binary)],
            self.live, interval_s=1.0 / REFRESH_HZ)

    def compose(self) -> ComposeResult:
        header = ui.wordmark("TELEMETRY") if hasattr(ui, "wordmark") else None
        yield Static(header or "OVAT telemetry", id="tel-header")
        with Horizontal(id="tel-numbers"):
            for metric, label, _hint in _HEADLINE:
                card = Vertical(classes="tel-card", id=_card_id(metric))
                yield card
        with Vertical(id="tel-graphs"):
            pass
        table = DataTable(id="tel-table", cursor_type="row", zebra_stripes=True)
        table.border_title = "sources"
        yield table
        yield Footer()

    def on_mount(self) -> None:
        for metric, label, _hint in _HEADLINE:
            card = self.query_one(f"#{_card_id(metric)}", Vertical)
            card.mount(Label(label))
            card.mount(Digits("---"))
        self._fill_sources_table()
        self.collector.start()
        # set_interval, not a worker loop: the sampling already happens on
        # the collector's own thread, so all this does is redraw.
        self.set_interval(1.0 / REFRESH_HZ, self._redraw)

    def on_unmount(self) -> None:
        # Leaving the screen must stop the subprocess a hardware source may
        # have launched, or it outlives the page that started it.
        self.collector.stop()

    # ---- drawing -----------------------------------------------------------

    def _fill_sources_table(self) -> None:
        """One row per source, saying plainly whether it works HERE.

        A source that cannot run and a source that is idle look identical in
        a graph. Only one of them is worth doing something about.
        """
        from rich.text import Text

        table = self.query_one("#tel-table", DataTable)
        table.clear(columns=True)
        table.add_column("Source")
        table.add_column("State")
        table.add_column("Detail")
        unavailable = self.collector.unavailable
        for source in self.collector.sources:
            reason = unavailable.get(source.name)
            state = (Text("live", style=f"bold {ui.GREEN}") if reason is None
                     else Text("n/a", style=ui.YELLOW))
            # Text cells, never str: a DataTable renders str as markup, and a
            # path or an exception with a bracket in it would raise mid-draw.
            table.add_row(Text(source.name, style=ui.CYAN), state,
                          Text(reason or "sampling", style=ui.DIM))

    def _redraw(self) -> None:
        for metric, _label, _hint in _HEADLINE:
            card = self.query_one(f"#{_card_id(metric)}", Vertical)
            digits = card.query_one(Digits)
            value = self.live.latest(metric)
            if value is None:
                digits.update("---")
                card.add_class("-dead")
            else:
                digits.update(f"{value:.0f}" if abs(value) >= 10
                              else f"{value:.1f}")
                card.remove_class("-dead")
        self._redraw_graphs()

    def _redraw_graphs(self) -> None:
        """One sparkline per metric actually being reported.

        Built from what the buffer HAS rather than a hardcoded list, so a
        source added later draws itself with no change here.
        """
        graphs = self.query_one("#tel-graphs", Vertical)
        for metric in self.live.metrics():
            spark_id = _spark_id(metric)
            data = self.live.series(metric)
            if len(data) < 2:
                continue                      # a line needs two points
            existing = graphs.query(f"#{spark_id}")
            if existing:
                existing.first(Sparkline).data = data
            else:
                graphs.mount(Label(metric))
                graphs.mount(Sparkline(data, id=spark_id))

    # ---- actions -----------------------------------------------------------

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_clear(self) -> None:
        self.live.samples.clear()
        for spark in list(self.query_one("#tel-graphs", Vertical).children):
            spark.remove()

    def action_copy(self) -> None:
        import json

        self.app.copy_to_clipboard(json.dumps(list(self.live.samples),
                                              indent=2))
        self.notify("Telemetry copied as JSON.")


def _card_id(metric: str) -> str:
    return "card-" + metric.replace(".", "-").replace("_", "-")


def _spark_id(metric: str) -> str:
    return "spark-" + metric.replace(".", "-").replace("_", "-")
