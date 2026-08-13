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
from textual.widgets import (DataTable, Digits, Footer, Label, ProgressBar,
                             Rule, Static, TabbedContent, TabPane)

from ovat.cli import ui
from ovat.cli.commands import ScreenCommands
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
    # npu.* comes from the driver's own busy counter and is a real percentage;
    # intel.* comes from UT and is richer but currently silent. Preferring the
    # driver means the NPU card shows a number where one exists.
    ("npu.utilization", "NPU", "%"),
    # The cache figure earns a headline slot because it is the one number that
    # predicts the undecoded-tool-call failure before it happens.
    ("ovms.kv_cache_pct", "KV CACHE", "%"),
    ("intel.npu_utilization", "NPU", "%"),
    ("intel.gpu_utilization", "GPU", "%"),
    ("intel.power_w", "POWER", "W"),
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

_SOURCES_HELP = """
[b]What each source is[/b]

  [cyan]system[/cyan]   this machine: CPU per core, RAM, clock speed. psutil,
           so it works on macOS, Windows and Linux alike.
  [cyan]process[/cyan]  OVAT's OWN memory. Not the model's: an 8B model lives
           inside OVMS, in a different process entirely.
  [cyan]intel[/cyan]    GPU/NPU activity and package power, via Intel's
           Unified Telemetry tool. AI PC only.

A source that cannot run HERE says so with a reason. That matters more than
it looks: an absent source and an idle one draw the same flat line, and only
one of them is worth doing anything about.
"""

_INTEL_HELP = """
[b]Where the NPU number on the Live tab comes from[/b]

Two different things can report an NPU, and only one of them currently gives
this page a figure. Read this tab as "which one am I getting, and why".

  [cyan]npu.*[/cyan]     the DRIVER's own counter. A real live percentage.
             Linux only today. This is what the NPU card shows.
  [cyan]intel.*[/cyan]   Intel Unified Telemetry. Much richer -- power,
             per-engine timelines, thermals -- but see the limit below.

[b]1. The driver counter (Linux, no install)[/b]
The intel_vpu kernel module publishes a running total of microseconds the NPU
spent executing jobs:

  [cyan]/sys/bus/pci/drivers/intel_vpu/<dev>/npu_busy_time_us[/cyan]

OVAT reads it twice and divides to get a duty cycle -- the same arithmetic
btop's NPU meter uses. Nothing to install; if the row is missing, check the
module is loaded with [cyan]lsmod | grep intel_vpu[/cyan].

[b]2. On Windows there is no verified equivalent yet[/b]
Task Manager shows NPU usage, but it reads it through DXCore rather than a
documented performance counter, so there is no path OVAT can rely on. Before
assuming, check whether YOUR build exposes one:

  [cyan]Get-Counter -ListSet *NPU*[/cyan]
  [cyan]Get-Counter "\\GPU Engine(*)\\Utilization Percentage" -MaxSamples 1[/cyan]

The second is documented and does work, so a GPU figure is reachable on
Windows even where the NPU one is not. If the first prints a counter set,
that is worth reporting -- it is the missing piece.

[b]3. Intel Unified Telemetry (optional, richer)[/b]
  1. Download the ut-tool release and unzip it; it extracts to a versioned
     folder
  2. Windows: open Command Prompt AS ADMINISTRATOR, cd to that folder, run
     [cyan]ut-vars.cmd[/cyan] to set the environment
  3. Point OVAT at it: set [cyan]OVAT_UT[/cyan] to that folder, or drop it in
     [cyan]~/ut[/cyan] and OVAT finds it
  4. Windows and Linux only; there is no macOS build

[b]Why the intel.* rows are still empty[/b]
MEASURED on the AI PC, not assumed: continuous mode does not stream numbers.
It writes binary traces (ut_default_output.l0_gpu.bin, .l0_npu.bin) that need
[cyan]bin2perfetto[/cyan] to decode, so the Sources tab honestly reports it
running while this page gets nothing from it. Decoding those traces is the
next step. Saying so beats a graph that pretends to be live.
"""

_PLANO_HELP = """
[b]plano gateway[/b]  (katanemo/plano, formerly archgw)

An optional proxy between OVAT and OVMS. OVAT keeps talking to a plain URL;
plano forwards it on and adds a trace to every request.

[b]Why it is here[/b]
[cyan]OpenTelemetry for three lines of config[/cyan], rather than an exporter
written by hand. Every request becomes an OTEL span with per-call latency and
token counts, and OVAT gains no OTEL dependency of its own.

[b]Platform, and read this first[/b]
plano's own docs list its supported platforms as Linux (x86_64, aarch64) and
macOS (Apple Silicon). There is NO Windows build, so on the AI PC
`planoai up` exits with "Unsupported platform windows/amd64" before it reads
any config. Three ways round it, none of them a workaround for a bug:

  [cyan]planoai up <config> --docker[/cyan]   plano in a Linux container
  or run plano under WSL2
  or run it on another machine pointing at the AI PC's OVMS over the network
  (base_url takes any host, not just 127.0.0.1)

[b]Run it, in this order[/b]
  [b]1.[/b] [cyan]uv tool install planoai==0.4.27[/cyan]
     Pinned because that is the version this integration was verified
     against. 0.4.33 is current; treat an upgrade as a retest, since the
     config schema is what the two walls below are about.
  [b]2.[/b] [cyan]ovat serve examples/plano/workflow.yml[/cyan]
     OVMS on :8000. Wait for it to report ready.
  [b]3.[/b] [cyan]python examples/plano/ovms_id_bridge.py[/cyan]
     The bridge on :8001. Leave it running.
  [b]4.[/b] [cyan]planoai up examples/plano/plano-config.yaml[/cyan]
     plano on :12000. Add [cyan]--docker[/cyan] on Windows.
  [b]5.[/b] [cyan]ovat run examples/plano/workflow.yml --input "hello"[/cyan]
     The request now goes OVAT -> plano -> bridge -> OVMS.
  [b]6.[/b] [cyan]planoai obs[/cyan]            live aggregate view
     [cyan]planoai trace listen[/cyan]   stream traces as they arrive
     [cyan]planoai trace --list[/cyan]   what has been captured
     [cyan]planoai trace <id>[/cyan]     one request in depth

[b]The spike question, answered[/b]
plano defaults to calling upstreams at /v1; OVMS serves its OpenAI API under
/v3. There is no separate prefix field: plano parses base_url and lifts the
path out of it itself, and its schema REJECTS base_url_path_prefix outright.
So the path half of the fix is just the URL:

  [cyan]base_url: http://<host>:8001/v3[/cyan]

Two more walls turned up behind it, both found in plano's own source rather
than guessed at:

  [b]1.[/b] The model name needs a [cyan]provider/[/cyan] prefix. plano does
     model_name.split("/") in config_generator.py and raises "Invalid model
     name" without one, so the config says [cyan]ovms/Qwen3-8B-int4-ov[/cyan]
     and plano strips the prefix before forwarding.
  [b]2.[/b] plano's Envoy WASM filter requires a top-level [cyan]"id"[/cyan]
     string in the response. OVMS returns valid OpenAI JSON but omits it, so
     plano rejects every reply. [cyan]examples/plano/ovms_id_bridge.py[/cyan]
     sits on :8001, forwards to OVMS on :8000, injects the field, and also
     handles the Transfer-Encoding: chunked bodies plano sends.

So base_url points at the BRIDGE, not at OVMS directly. Still no fork and no
patch of either project: one small proxy and one URL.

[b]Is this tab live?[/b]
No, and deliberately so. plano runs as its own process with its own dashboard
([cyan]planoai obs[/cyan]), so duplicating it here would be a worse copy of a
tool that already exists. What this page owns is the Live tab: system,
process and Intel. plano owns the request traces.
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
    #tel-table { height: 12; border: round $primary; margin: 0 2 1 2; }
    #tel-sources-help, #tel-intel-help, #tel-plano-help {
        padding: 1 2;
        color: $text-muted;
    }
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
            with TabPane("Sources", id="tab-sources"):
                table = DataTable(id="tel-table", cursor_type="row",
                                  zebra_stripes=True)
                table.border_title = "where the numbers come from"
                yield table
                yield Static(_SOURCES_HELP, id="tel-sources-help")
            with TabPane("Intel", id="tab-intel"):
                yield Static(_INTEL_HELP, id="tel-intel-help")
            with TabPane("Plano", id="tab-plano"):
                yield Static(_PLANO_HELP, id="tel-plano-help")
        yield Footer()

    def on_mount(self) -> None:
        self._cards: dict = {}
        self._rows: set = set()
        numbers = self.query_one("#tel-live-table", DataTable)
        numbers.add_column("Source", key="source")
        numbers.add_column("Metric", key="metric")
        for column in _COLUMNS:
            numbers.add_column(column, key=column)
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
        unavailable = self.collector.unavailable
        state_signature = tuple((s.name, unavailable.get(s.name)) for s in self.collector.sources)
        if getattr(self, "_last_table_state", None) == state_signature:
            return
        self._last_table_state = state_signature
        table.clear(columns=True)
        table.add_column("Source")
        table.add_column("State")
        table.add_column("Detail")
        for source in self.collector.sources:
            reason = unavailable.get(source.name)
            state = (Text("live", style=f"bold {ui.GREEN}") if reason is None
                     else Text("n/a", style=ui.YELLOW))
            # Text cells, never str: a DataTable renders str as markup, and a
            # path or an exception with a bracket in it would raise mid-draw.
            detail = reason or getattr(source, "note", None) or "sampling"
            table.add_row(Text(source.name, style=ui.CYAN), state,
                          Text(detail, style=ui.DIM))

    def _redraw(self) -> None:
        self._fill_sources_table()
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
