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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (DataTable, Digits, Footer, Label, ProgressBar,
                             Rule, Sparkline, Static)

from ovat.cli import ui
from ovat.cli.commands import ScreenCommands
from ovat.telemetry.collector import Collector
from ovat.telemetry.sinks import LiveBufferSink
from ovat.telemetry.sources import (AgentTraceSource, IntelHardwareSource,
                                    ProcessMemorySource, SystemSource)

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
    ("intel.npu_utilization", "NPU", "%"),
    ("intel.gpu_utilization", "GPU", "%"),
    ("intel.power_w", "POWER", "W"),
    ("system.cpu_pct", "SYS CPU", "%"),
    ("system.ram_used_pct", "SYS RAM", "%"),
    ("process.rss_mb", "OVAT MEM", "MB"),
    ("system.proc_cpu_pct", "OVAT CPU", "%"),
    ("agent.prompt_tokens", "PROMPT", "tok"),
    ("agent.completion_tokens", "REPLY", "tok"),
    ("system.threads", "THREADS", ""),
]
MAX_CARDS = 5


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
            [SystemSource(),
             ProcessMemorySource(),
             AgentTraceSource(agent),
             IntelHardwareSource(ut_binary)],
            self.live, interval_s=1.0 / REFRESH_HZ)

    def compose(self) -> ComposeResult:
        header = ui.wordmark("TELEMETRY") if hasattr(ui, "wordmark") else None
        yield Static(header or "OVAT telemetry", id="tel-header")
        # Cards are mounted once we know which metrics have data; an empty
        # row now beats five boxes reading "---" forever.
        yield Horizontal(id="tel-numbers")
        yield Rule(id="tel-rule")
        with VerticalScroll(id="tel-graphs"):
            pass
        table = DataTable(id="tel-table", cursor_type="row", zebra_stripes=True)
        table.border_title = "sources"
        yield table
        yield Footer()

    def on_mount(self) -> None:
        self._cards: dict = {}
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
        self._redraw_graphs()

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
            card = Vertical(classes="tel-card", id=_card_id(metric))
            row.mount(card)
            card.mount(Label(f"{label}  [dim]{unit}[/dim]", markup=True))
            digits = Digits("---")
            card.mount(digits)
            bar = None
            if unit == "%":
                # A percentage has a meaningful maximum; a bare number does
                # not show how close to it you are.
                bar = ProgressBar(total=100, show_eta=False,
                                  show_percentage=False)
                card.mount(bar)
            self._cards[metric] = (metric, digits, bar)

    def _redraw_graphs(self) -> None:
        """A sparkline per metric, grouped under its source.

        Built from what the buffer HAS rather than a fixed list, so a source
        added later draws itself with no change here, and a machine missing a
        source simply has fewer rows instead of empty ones.
        """
        graphs = self.query_one("#tel-graphs", VerticalScroll)
        for metric in self.live.metrics():
            data = self.live.series(metric)
            if len(data) < 2:
                continue                      # a line needs two points
            spark_id = _spark_id(metric)
            existing = graphs.query(f"#{spark_id}")
            if existing:
                existing.first(Sparkline).data = data
                continue
            source, _, short = metric.partition(".")
            latest = data[-1]
            graphs.mount(Label(f"[dim]{source}[/dim]  {short}",
                               classes="tel-metric", markup=True))
            graphs.mount(Sparkline(data, id=spark_id))

    # ---- actions -----------------------------------------------------------

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_clear(self) -> None:
        self.live.samples.clear()
        for spark in list(self.query_one("#tel-graphs", VerticalScroll).children):
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
