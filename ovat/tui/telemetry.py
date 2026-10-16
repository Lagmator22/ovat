# ovat/tui/telemetry.py
"""`/telemetry` in the agent TUI: live numbers from every source that works
here, on one page that fits a normal terminal.

    Telemetry  live, 2 readings a second           Esc back  r reset  c copy  ? help
    ● system live  ● process live  ● npu n/a (no NPU found)  ● ovms silent (...)
    ┌ SYS CPU ──────┐┌ SYS RAM ──────┐┌ OVAT MEM ─────┐
    │ 56 %  █████░░ ││ 13 %  █░░░░░░ ││ 328 MB        │
    ┌ source  metric        now   min   max   mean ────────────────────────┐

The sampling is unchanged: ovat.telemetry's Collector and sources, on the
collector's own thread. What changed from the launcher's page
(ovat/cli/telemetry_screen.py), each because it was seen on screen:

  * The copied stylesheet makes every Screen transparent and every
    Horizontal auto-height. Inside the agent app the old page showed the
    chat through it, and its number cards stretched to the bottom of the
    window, which pushed the table and the source line off screen. This
    page sets its own background and heights with ids, which the app's
    stylesheet cannot override.
  * No five-row wordmark: the numbers start on the third line.
  * The source line sits ABOVE the numbers. It says why a row is missing,
    which is the first question when one is.
  * A whole number prints as one ("4", not "4.0").
  * A percentage near its limit is coloured where it is read, in the tile
    and in the table, so a pinned core stands out among sixteen.
"""
import json

from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Static

from ovat.cli import ui
from ovat.cli.telemetry_screen import _PREFERRED, _first_clause, _fmt
from ovat.telemetry.collector import Collector
from ovat.telemetry.sinks import LiveBufferSink

REFRESH_S = 0.5
MAX_TILES = 5
# A percentage at or above these is shown yellow, then red.
ATTENTION_PCT = 85.0
CRITICAL_PCT = 95.0
# Not coloured: with no --cache_size OVMS grows its KV cache on demand and
# sits near 100% of the CURRENT allocation most of the time (AGENTS.md,
# "Tool calls that stop decoding"), so red there would be a false alarm.
# Nor OVAT's own CPU: a process's CPU percentage is summed over cores, so
# 103 on a four-core machine is a normal burst, not an overload.
_NOT_AN_ALARM = {"ovms.kv_cache_pct", "system.proc_cpu_pct"}
# Table order: the sources most likely to explain a slow answer first.
_SOURCE_ORDER = ("agent", "ovms", "npu", "intel", "process", "system")
_COLUMNS = ("now", "min", "max", "mean")

HELP = """[b]What the numbers are[/b]
  [cyan]system[/cyan]   this computer: CPU per core, RAM, clock
  [cyan]process[/cyan]  OVAT itself (with OVMS serving, the model is not here)
  [cyan]npu[/cyan]      NPU busy %, from the driver (Windows and Linux)
  [cyan]ovms[/cyan]     the server's KV cache, read from ovms.log
  [cyan]intel[/cyan]    power and NPU bandwidth, from Intel Unified Telemetry

[b]The line at the top[/b] says, for each source, whether it is [green]live[/green],
[yellow]silent[/yellow] (running, nothing read yet) or [yellow]n/a[/yellow] (cannot run here), and why.
Yellow and red figures are percentages at 85% and 95% or more.

[b]Turning on the intel rows[/b] (Windows 11, Intel Core Ultra): unzip Intel
Unified Telemetry, set [cyan]OVAT_UT[/cyan] to that folder or put it in [cyan]~/ut[/cyan], and run
[cyan]ut-vars.cmd[/cyan] from an Administrator prompt first."""


def _default_sources() -> list:
    from ovat.telemetry.sources import (IntelHardwareSource, NPUSource,
                                        OVMSLogSource, ProcessMemorySource,
                                        SystemSource)
    return [SystemSource(), ProcessMemorySource(), NPUSource(),
            OVMSLogSource(), IntelHardwareSource()]


def is_percentage(metric: str) -> bool:
    return metric.endswith("_pct") or metric.endswith(".utilization")


def level_style(metric: str, value: float | None) -> str:
    """"" for a normal reading, else the colour that says how close to its
    limit a percentage is."""
    if (value is None or not is_percentage(metric)
            or metric in _NOT_AN_ALARM):
        return ""
    if value >= CRITICAL_PCT:
        return f"bold {ui.RED}"
    if value >= ATTENTION_PCT:
        return f"bold {ui.YELLOW}"
    return ""


def bar(value: float, width: int = 10) -> str:
    filled = round(max(0.0, min(100.0, value)) / 100 * width)
    return "█" * filled + "░" * (width - filled)


def tile_text(label: str, unit: str, metric: str, value: float | None) -> Text:
    out = Text(f"{label}\n", style=ui.DIM)
    if value is None:
        out.append("---", style=ui.DIM)
        return out
    out.append(_fmt(value), style=level_style(metric, value) or "bold")
    if unit:
        out.append(f" {unit}", style=ui.DIM)
    if unit == "%":
        out.append(f"  {bar(value)}",
                   style=level_style(metric, value) or ui.GREEN)
    return out


def sources_line(collector: Collector) -> Text:
    """Each source and whether it works here right now, with the reason."""
    unavailable = collector.unavailable
    line = Text()
    for source in collector.sources:
        reason = unavailable.get(source.name)
        note = None if reason else getattr(source, "note", None)
        if line:
            line.append("  ")
        if reason:
            line.append("● ", style=ui.DIM)
            line.append(source.name, style=ui.CYAN)
            line.append(f" n/a ({_first_clause(reason)})", style=ui.YELLOW)
        elif note:
            line.append("● ", style=ui.YELLOW)
            line.append(source.name, style=ui.CYAN)
            line.append(f" silent ({_first_clause(note)})", style=ui.YELLOW)
        else:
            line.append("● ", style=ui.GREEN)
            line.append(source.name, style=ui.CYAN)
            line.append(" live", style=f"bold {ui.GREEN}")
    return line


def _order(metric: str) -> tuple:
    source, _, name = metric.partition(".")
    rank = (_SOURCE_ORDER.index(source) if source in _SOURCE_ORDER
            else len(_SOURCE_ORDER))
    return rank, name


class TelemetryView(Screen):
    """Live telemetry, opened with /telemetry and closed with Esc."""

    # CSS, not DEFAULT_CSS: the app's copied stylesheet outranks any
    # widget's DEFAULT_CSS, and it sets Screen to transparent and
    # Horizontal to auto height. The ids make these rules win. Not scoped:
    # scoped rules match only DESCENDANTS, so #telemetry-view never reached
    # the screen itself and the chat showed through it.
    SCOPED_CSS = False
    CSS = """
    #telemetry-view { background: $background; layout: vertical; }
    #tel-title { height: 1; width: 100%; padding: 0 1; }
    #tel-title-left { width: 1fr; }
    #tel-title-keys { width: auto; color: $text-muted; }
    #tel-sources { height: auto; padding: 0 1; margin-bottom: 1; }
    #tel-tiles { height: 4; width: 100%; padding: 0 1; }
    .tel-tile { width: 1fr; height: 4; padding: 0 1;
                border: solid $foreground 35%; }
    #tel-table { height: 1fr; margin: 0 1;
                 border: solid $foreground 35%; }
    #tel-help { display: none; height: auto; margin: 0 1; padding: 0 1;
                border: solid $foreground 35%; }
    #tel-help.shown { display: block; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "reset", "Reset"),
        Binding("c", "copy", "Copy"),
        Binding("question_mark", "toggle_help", "Help"),
    ]

    def __init__(self, sources=None, refresh_s: float = REFRESH_S,
                 agent=None) -> None:
        """`agent`: the running agent, or a callable returning it (the
        session swaps its agent on /engine, so pass the callable). Without
        one the page simply has no agent rows."""
        super().__init__(id="telemetry-view")
        self.live = LiveBufferSink()
        sources = list(_default_sources() if sources is None else sources)
        if agent is not None:
            from ovat.telemetry.sources import AgentTraceSource
            sources.append(AgentTraceSource(agent))
        self.collector = Collector(sources, self.live, interval_s=refresh_s)
        self._refresh_s = refresh_s
        self._tiles: list = []
        self._rows: set = set()

    def compose(self):
        with Horizontal(id="tel-title"):
            title = Text("Telemetry", style="bold")
            title.append(f"  live, {1 / self._refresh_s:.0f} readings a "
                         f"second", style=ui.DIM)
            yield Static(title, id="tel-title-left")
            yield Static("Esc back  r reset  c copy  ? help",
                         id="tel-title-keys", markup=False)
        yield Static("", id="tel-sources")
        yield Horizontal(id="tel-tiles")
        table = DataTable(id="tel-table", cursor_type="row",
                          zebra_stripes=True)
        table.border_title = "live numbers"
        yield table
        yield Static(HELP, id="tel-help")

    def on_mount(self) -> None:
        table = self.query_one("#tel-table", DataTable)
        table.add_column("source", key="source")
        table.add_column("metric", key="metric")
        for column in _COLUMNS:
            table.add_column(column, key=column)
        table.focus()
        self.collector.start()
        self.set_interval(self._refresh_s, self.redraw)

    def on_unmount(self) -> None:
        # A hardware source may have started a subprocess (Intel UT); it
        # must not outlive the page.
        self.collector.stop()

    # ---- drawing -------------------------------------------------------------

    def _wanted_tiles(self) -> list:
        return [entry for entry in _PREFERRED
                if self.live.latest(entry[0]) is not None][:MAX_TILES]

    def redraw(self) -> None:
        self.query_one("#tel-sources", Static).update(
            sources_line(self.collector))
        wanted = self._wanted_tiles()
        row = self.query_one("#tel-tiles", Horizontal)
        if [w[0] for w in wanted] != [t[0] for t in self._tiles]:
            # Rebuilt when the best-ranked metrics with data change: Intel
            # UT's first reading lands seconds after the rest. Static tiles
            # without ids, so a rebuild cannot collide with the old ones.
            row.remove_children()
            self._tiles = [(metric, label, unit, Static(classes="tel-tile"))
                           for metric, label, unit in wanted]
            row.mount_all([tile for *_, tile in self._tiles])
        for metric, label, unit, tile in self._tiles:
            tile.update(tile_text(label, unit, metric,
                                  self.live.latest(metric)))
        self._redraw_table()

    def _redraw_table(self) -> None:
        table = self.query_one("#tel-table", DataTable)
        added = False
        for metric in self.live.metrics():
            data = self.live.series(metric)
            if not data:
                continue
            now = data[-1]
            cells = (Text(_fmt(now), style=level_style(metric, now)),
                     _fmt(min(data)),
                     Text(_fmt(max(data)), style=level_style(metric,
                                                             max(data))),
                     _fmt(sum(data) / len(data)))
            if metric in self._rows:
                # Updated in place: clearing twice a second would reset the
                # cursor and the scroll on a sixteen-core machine.
                for column, value in zip(_COLUMNS, cells):
                    table.update_cell(metric, column, value)
                continue
            source, _, short = metric.partition(".")
            table.add_row(Text(source, style=ui.DIM),
                          Text(short, style=ui.CYAN), *cells, key=metric)
            self._rows.add(metric)
            added = True
        if added:
            # Sorted only when a row arrives, so the order is stable while
            # the numbers move.
            table.sort("source", "metric", key=lambda cells: _order(
                f"{cells[0]}.{cells[1]}"))

    # ---- keys ----------------------------------------------------------------

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_reset(self) -> None:
        self.live.samples.clear()
        self.query_one("#tel-table", DataTable).clear()
        self._rows.clear()

    def action_copy(self) -> None:
        self.app.copy_to_clipboard(json.dumps(list(self.live.samples),
                                              indent=2))
        self.notify("Telemetry copied as JSON.", timeout=2)

    def action_toggle_help(self) -> None:
        self.query_one("#tel-help", Static).toggle_class("shown")
