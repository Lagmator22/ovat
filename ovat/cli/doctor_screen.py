# ovat/cli/doctor_screen.py
"""The doctor screen: the same checks as `ovat doctor`, live inside the TUI.

Every other TUI command shells out (`ovat doctor` in a subprocess). This one
runs the checks IN-PROCESS, and that is the whole point: fix a PATH, start
OVMS, press r, and the table redraws without leaving the app.

Three things here are load-bearing, and each is a bug avoided:

  * The checks do NOT run on the UI thread. run_checks() imports openvino,
    asks it for the device list, and opens a socket with a one-second timeout.
    On a cold start that is several seconds of a frozen screen, so it runs on
    a worker and posts its result back.
  * Table cells are rich Text, never str. A Table renders a str cell as
    MARKUP, so a config value containing a bracket would raise MarkupError
    part-way through the render: exactly the bug ui.esc() exists to stop on
    the CLI side, in the one place esc() cannot reach.
  * Every colour and the wordmark come from ovat.cli.ui. This screen owns no
    palette of its own. It also cannot use ui.status_text() or rich theme
    names like "ovat.ok": those resolve against the CLI's themed Console, and
    a Textual RichLog renders against the app's console instead, where the
    name means nothing. So the glyph comes from ui.STATUS_GLYPH and the colour
    from ui's hex aliases; both are still single-sourced.
"""
import os
import platform
import sys

from rich import box
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Input, RichLog, Static

from ovat.cli import ui

# Derived from ui's palette aliases, never redefined here.
_STATUS_COLOUR = {"ok": ui.GREEN, "warn": ui.YELLOW, "fail": ui.RED}
_NAME_COLOUR = {"ok": ui.CYAN, "warn": ui.YELLOW, "fail": ui.RED}


def _status_cell(status: str) -> Text:
    """The coloured glyph + label for one check status."""
    glyph = ui.STATUS_GLYPH.get(status, ("?", None))[0]
    colour = _STATUS_COLOUR.get(status, ui.DIM)
    return Text(f"{glyph} {status}", style=f"bold {colour}")


def _header(config_path: str | None) -> Text:
    """The big sign, matching what `ovat doctor` prints in the terminal."""
    out = ui.wordmark("DOCTOR")
    out.append("\nenvironment & workflow diagnostics\n", style=f"bold {ui.CYAN}")
    os_name = {"darwin": "macOS", "win32": "Windows"}.get(
        sys.platform, platform.system())
    line = (f"{os_name} {platform.machine()}  ·  "
            f"Python {sys.version.split()[0]}")
    if config_path:
        line += f"  ·  {os.path.basename(config_path)}"
    out.append(line, style=ui.PURPLE)
    return out


def _report_text(checks: list) -> str:
    """The plain-text form of a run, for /copy. No colour, no box drawing."""
    width = max((len(c.name) for c in checks), default=0)
    lines = [f"{c.name.ljust(width)}  {c.status:<4}  {c.detail}" for c in checks]
    counts = {"ok": 0, "warn": 0, "fail": 0}
    for c in checks:
        counts[c.status] = counts.get(c.status, 0) + 1
    lines.append(f"\n{counts['ok']} ok · {counts['warn']} warn · "
                 f"{counts['fail']} fail")
    return "\n".join(lines)


class DoctorScreen(Screen):
    """`ovat doctor`, in-app and re-runnable."""

    DEFAULT_CSS = """
    DoctorScreen { background: #0b0e14; }
    #doc-header { padding: 1 2 0 2; height: auto; }
    #doc-log {
        height: 1fr;
        border: round #0068B5;
        background: #0d1117;
        padding: 0 1;
        margin: 1 2 0 2;
    }
    #doc-input { margin: 0 2 1 2; border: round #0068B5; }
    Footer { background: #0d1117; color: #00C7FD; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("r", "refresh_checks", "Re-run", show=True),
        Binding("c", "copy_report", "Copy", show=True),
    ]

    def __init__(self, config_path: str | None = None):
        super().__init__()
        # "" from a bare `/doctor` means no config, same as None.
        self._config_path = config_path or None
        self._report = ""        # the last run, as plain text, for /copy
        self._busy = False       # one run at a time; r is easy to lean on

    def compose(self) -> ComposeResult:
        yield Static(_header(self._config_path), id="doc-header")
        yield RichLog(id="doc-log", highlight=False, markup=False, wrap=True)
        yield Input(placeholder="r re-run  ·  c copy  ·  Esc back  ·  "
                                "/refresh /copy /clear /back", id="doc-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#doc-input", Input).focus()
        self.action_refresh_checks()

    # ---- helpers ----------------------------------------------------------

    def _log(self, renderable) -> None:
        self.query_one("#doc-log", RichLog).write(renderable)

    # ---- actions ----------------------------------------------------------

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_refresh_checks(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.query_one("#doc-log", RichLog).clear()
        self._log(Text("Running checks…", style=ui.DIM))
        self._run_checks()

    def action_copy_report(self) -> None:
        if not self._report:
            self._log(Text("nothing to copy yet; the checks are still running.",
                           style=ui.YELLOW))
            return
        self.app.copy_to_clipboard(self._report)
        self._log(Text("copied the report to the clipboard.", style=ui.GREEN))

    # ---- running the checks ------------------------------------------------

    @work(thread=True)
    def _run_checks(self) -> None:
        """Off the UI thread: this imports openvino and opens a socket."""
        from ovat.cli import diagnostics

        try:
            checks = diagnostics.run_checks(self._config_path)
        except Exception as exc:          # a check should never raise, but a
            self.app.call_from_thread(    # broken import here must not kill
                self._show_error, f"{type(exc).__name__}: {exc}")
            return
        self.app.call_from_thread(self._show_checks, checks)

    def _show_error(self, message: str) -> None:
        self._busy = False
        self._log(Text(f"could not run the checks: {message}", style=ui.RED))

    def _show_checks(self, checks: list) -> None:
        table = Table(header_style=f"bold {ui.BLUE}", border_style=ui.BLUE,
                      box=box.ROUNDED, expand=False, pad_edge=True)
        table.add_column("Check", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Detail", max_width=70)

        counts = {"ok": 0, "warn": 0, "fail": 0}
        for c in checks:
            counts[c.status] = counts.get(c.status, 0) + 1
            # Text cells, not str: a bracket in a config value or an exception
            # message would otherwise be parsed as markup and raise.
            table.add_row(
                Text(c.name, style=_NAME_COLOUR.get(c.status, ui.CYAN)),
                _status_cell(c.status),
                Text(c.detail, style=ui.DIM),
            )

        self.query_one("#doc-log", RichLog).clear()
        self._log(table)
        summary = Text()
        summary.append(f"✓ {counts['ok']} ok", style=f"bold {ui.GREEN}")
        if counts["warn"]:
            summary.append(f"  ·  ! {counts['warn']} warn",
                           style=f"bold {ui.YELLOW}")
        if counts["fail"]:
            summary.append(f"  ·  ✗ {counts['fail']} fail",
                           style=f"bold {ui.RED}")
        self._log(summary)
        self._report = _report_text(checks)
        self._busy = False

    # ---- the slash line ----------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        self.query_one("#doc-input", Input).value = ""
        if not line:
            return
        head = line.partition(" ")[0]
        if head in ("/back", "/exit"):
            self.app.pop_screen()
        elif head == "/refresh":
            self.action_refresh_checks()
        elif head == "/copy":
            self.action_copy_report()
        elif head == "/clear":
            self.query_one("#doc-log", RichLog).clear()
        else:
            self._log(Text(f"unknown doctor command {head}. "
                           f"I know /refresh /copy /clear /back.",
                           style=ui.YELLOW))
