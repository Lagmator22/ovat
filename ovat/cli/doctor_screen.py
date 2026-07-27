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

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Input, OptionList, Static
from textual.widgets.option_list import Option

from ovat.cli import ui
from ovat.cli.commands import ScreenCommands
from ovat.cli.widgets import SelectableRichLog

# The slash menu for this screen. Same shape as shell.TEMPLATES in the
# launcher: name, one-line help. Every one is a UI action, so unlike the
# launcher's menu none of these insert text to be edited.
DOCTOR_COMMANDS = [
    ("/refresh", "re-run the checks"),
    ("/sort", "toggle worst-first / original order"),
    ("/copy", "copy the report to the clipboard"),
    ("/clear", "empty the log"),
    ("/back", "return to the launcher"),
]

# Derived from ui's palette aliases, never redefined here.
_STATUS_COLOUR = {"ok": ui.GREEN, "warn": ui.YELLOW, "fail": ui.RED}
# Worst first. This is the order the table sorts into, because the whole
# point of running doctor is to find what is broken, and a fail sitting below
# nine oks is a fail you scroll past.
_SEVERITY = {"fail": 0, "warn": 1, "ok": 2}
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


class DoctorCommands(ScreenCommands):
    """Palette entries for the doctor screen only."""

    def commands(self) -> list:
        screen = self.screen
        return [
            ("Re-run the checks", "diagnose the environment again",
             screen.action_refresh_checks),
            ("Copy the report", "put the plain-text report on the clipboard",
             screen.action_copy_report),
            ("Sort by severity", "worst first, or back to the original order",
             screen.action_toggle_sort),
        ]


class DoctorScreen(Screen):
    """`ovat doctor`, in-app and re-runnable."""

    COMMANDS = {DoctorCommands}

    DEFAULT_CSS = """
    DoctorScreen { background: $background; }
    #doc-header { padding: 1 2 0 2; height: auto; }
    #doc-table {
        height: 1fr;
        border: round $primary;
        background: $surface;
        margin: 1 2 0 2;
    }
    #doc-table > .datatable--cursor { background: $primary; color: $foreground; }
    #doc-table > .datatable--hover { background: $primary 25%; }
    /* The log is now a message strip, not the main event: confirmations,
       refusals and errors only. Fixed height so the table never fights it
       for room, and hidden until it has something to say. */
    #doc-log {
        display: none;
        height: 5;
        border: round $secondary;
        background: $surface;
        padding: 0 1;
        margin: 0 2;
    }
    #doc-palette {
        display: none;
        height: auto;
        max-height: 6;
        margin: 0 2;
        border: round $secondary;
        background: $surface;
    }
    #doc-palette > .option-list--option-hover { background: $primary 30%; }
    #doc-palette > .option-list--option-highlighted {
        background: $primary;
        color: $foreground;
        text-style: bold;
    }
    #doc-input { margin: 0 2 1 2; border: round $primary; }
    Footer { background: $surface; color: $accent; }
    """

    # Only NON-PRINTABLE keys can be bindings here. The input is focused so the
    # user can type slash commands, and a focused Input consumes every
    # printable character, so "r" and "c" bindings never fired: the letter went
    # into the box and came back as "unknown doctor command r". The footer was
    # advertising keys that did nothing. f5 is the usual "refresh" key and is
    # not printable, so it reaches the screen. Copying stays a slash command.
    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("f5", "refresh_checks", "Re-run", show=True),
        Binding("f6", "toggle_sort", "Sort", show=True),
    ]

    def check_action(self, action: str, parameters: tuple):
        """Grey out footer keys that cannot do anything right now.

        Textual calls this for every binding when it builds the footer, and
        again on refresh_bindings(). Returning None shows the key but greys
        it; returning False hides it entirely.

        Grey, not hidden, for both of these. A key that VANISHES while the
        checks run and reappears afterwards makes the footer twitch, and the
        user is left unsure whether F5 exists at all. Greyed says "that is
        the key, just not yet", which is the true statement.
        """
        if action == "refresh_checks" and self._busy:
            return None
        if action == "toggle_sort" and not self._checks:
            return None
        return True

    def __init__(self, config_path: str | None = None):
        super().__init__()
        # "" from a bare `/doctor` means no config, same as None.
        self._config_path = config_path or None
        self._report = ""        # the last run, as plain text, for /copy
        self._busy = False       # one run at a time; r is easy to lean on
        self._checks: list = []  # the last run, for re-sorting without re-running
        self._worst_first = True  # what doctor is FOR: show the problems first

    def compose(self) -> ComposeResult:
        yield Static(_header(self._config_path), id="doc-header")
        table = DataTable(id="doc-table", cursor_type="row", zebra_stripes=True)
        table.border_title = "diagnostics"
        table.border_subtitle = "F5 re-run  ·  F6 sort  ·  Enter copies a row"
        table.tooltip = ("Select a row and press Enter to copy just that "
                         "check\n/copy takes the whole report")
        yield table
        log = SelectableRichLog(id="doc-log", highlight=False,
                                markup=False, wrap=True)
        yield log
        yield OptionList(id="doc-palette")
        doc_input = Input(
            placeholder="type / for commands  ·  F5 re-run  ·  Esc back",
            id="doc-input")
        doc_input.tooltip = ("Checks run in the background, so the screen\n"
                             "stays usable while they do\n"
                             "/copy puts the whole report on the clipboard")
        yield doc_input
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#doc-input", Input).focus()
        self.action_refresh_checks()

    # ---- helpers ----------------------------------------------------------

    def _log(self, renderable) -> None:
        """Say something to the user, and make the message strip visible.

        It stays hidden until there IS a message, so a clean run is the table
        and nothing else rather than the table plus an empty box.
        """
        log = self.query_one("#doc-log", SelectableRichLog)
        log.display = True
        log.write(renderable)

    # ---- actions ----------------------------------------------------------

    def action_back(self) -> None:
        # Esc closes the menu first if it is open, then leaves the screen.
        # Otherwise opening the menu would be a one-way trip without a mouse.
        palette = self.query_one("#doc-palette", OptionList)
        if palette.display:
            palette.display = False
            self.query_one("#doc-input", Input).focus()
            return
        self.app.pop_screen()

    def action_refresh_checks(self) -> None:
        if self._busy:
            return
        self._busy = True
        # Without this the footer keeps showing F5 as live for the whole run:
        # check_action is only consulted when the bindings are refreshed.
        self.refresh_bindings()
        table = self.query_one("#doc-table", DataTable)
        table.loading = True          # Textual's own spinner, over the table
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
        self.refresh_bindings()
        self.query_one("#doc-table", DataTable).loading = False
        self._log(Text(f"could not run the checks: {message}", style=ui.RED))

    def _show_checks(self, checks: list) -> None:
        self._checks = list(checks)
        self._fill_table()

        counts = {"ok": 0, "warn": 0, "fail": 0}
        for c in checks:
            counts[c.status] = counts.get(c.status, 0) + 1
        summary = Text()
        summary.append(f"\u2713 {counts['ok']} ok", style=f"bold {ui.GREEN}")
        if counts["warn"]:
            summary.append(f"  \u00b7  ! {counts['warn']} warn",
                           style=f"bold {ui.YELLOW}")
        if counts["fail"]:
            summary.append(f"  \u00b7  \u2717 {counts['fail']} fail",
                           style=f"bold {ui.RED}")
        # The counts belong on the border, not in the log: they describe the
        # table, and putting them there keeps the message strip for messages.
        self.query_one("#doc-table", DataTable).border_title = Text.assemble(
            ("diagnostics  ", ui.DIM), summary)
        self._report = _report_text(checks)
        self._busy = False
        self.refresh_bindings()       # F5 and F6 are live again

    def _fill_table(self) -> None:
        """Rebuild the rows in the current order.

        Rebuilt rather than sorted in place. DataTable.sort() would order by
        the CELL values, and the status cell renders as a glyph plus a word,
        so sorting it would be alphabetical: fail, ok, warn. That puts ok
        between the two states you actually care about. Ranking the source
        data by severity and re-adding is both correct and obvious.
        """
        table = self.query_one("#doc-table", DataTable)
        table.loading = False
        table.clear(columns=True)
        table.add_column("Check", key="name")
        table.add_column("Status", key="status")
        table.add_column("Detail", key="detail")

        rows = list(self._checks)
        if self._worst_first:
            # Stable, so checks of equal severity keep the order they ran in.
            rows.sort(key=lambda c: _SEVERITY.get(c.status, 9))
        for index, c in enumerate(rows):
            # Text cells, not str: a DataTable renders a str cell as MARKUP,
            # so a config value or an exception message containing a bracket
            # would raise part-way through the render. This is the same bug
            # ui.esc() exists to stop on the CLI side.
            table.add_row(
                Text(c.name, style=_NAME_COLOUR.get(c.status, ui.CYAN)),
                _status_cell(c.status),
                Text(c.detail, style=ui.DIM),
                key=str(index),
            )
        table.border_subtitle = (
            "F5 re-run  \u00b7  F6 "
            + ("original order" if self._worst_first else "worst first")
            + "  \u00b7  Enter copies a row")

    def action_toggle_sort(self) -> None:
        if not self._checks:
            return
        self._worst_first = not self._worst_first
        self._fill_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a row copies THAT check, not the whole report.

        When one check fails, that one line is what gets pasted into a chat
        or an issue. Copying all fifteen and asking the reader to find the
        relevant one is worse than useless.
        """
        table = self.query_one("#doc-table", DataTable)
        try:
            cells = table.get_row(event.row_key)
        except Exception:
            return
        line = "  ".join(str(cell) for cell in cells)
        self.app.copy_to_clipboard(line)
        self._log(Text(f"copied: {line}", style=ui.GREEN))

    # ---- the slash line ----------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        """Show the menu, filtered, as soon as the line starts with a slash."""
        palette = self.query_one("#doc-palette", OptionList)
        value = event.value
        if value.startswith("/") and " " not in value:
            matches = [(name, help_text) for name, help_text in DOCTOR_COMMANDS
                       if name.startswith(value.lower())]
            palette.clear_options()
            if matches:
                palette.add_options([
                    Option(ui.slash_label(name, help_text), id=name.lstrip("/"))
                    for name, help_text in matches
                ])
                palette.display = True
                return
        palette.display = False

    def on_key(self, event) -> None:
        """Down steps from the input into the menu, like the launcher."""
        if event.key != "down":
            return
        palette = self.query_one("#doc-palette", OptionList)
        inp = self.query_one("#doc-input", Input)
        if palette.display and self.focused is inp:
            palette.focus()
            if palette.option_count:
                palette.highlighted = 0
            event.stop()

    def on_option_list_option_selected(self,
                                       event: OptionList.OptionSelected) -> None:
        self._close_menu()
        self._dispatch("/" + (event.option.id or ""))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        self._close_menu()
        if line:
            self._dispatch(line.partition(" ")[0])

    def _close_menu(self) -> None:
        self.query_one("#doc-palette", OptionList).display = False
        inp = self.query_one("#doc-input", Input)
        inp.value = ""
        inp.focus()

    def _dispatch(self, name: str) -> None:
        """Run one doctor command. The menu and the typed line share this."""
        if name in ("/back", "/exit"):
            self.app.pop_screen()
        elif name == "/refresh":
            self.action_refresh_checks()
        elif name == "/copy":
            self.action_copy_report()
        elif name == "/sort":
            self.action_toggle_sort()
        elif name == "/clear":
            log = self.query_one("#doc-log", SelectableRichLog)
            log.clear()
            log.display = False
        else:
            known = " ".join(cmd for cmd, _ in DOCTOR_COMMANDS)
            self._log(Text(f"unknown doctor command {name}. I know {known}.",
                           style=ui.YELLOW))
