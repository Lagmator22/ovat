# ovat/tui/widgets/doctor.py
"""`/doctor` in the agent TUI: the same checks as `ovat doctor`, as one block
in the conversation.

    ✕ doctor  1 fail · 1 warn · 10 ok      Linux x86_64 · Python 3.13 · w.yml
      ✕ Workflow config   no such file: w.yml
      ! OVMS serving      ovms not found (...). Run 'ovat setup' to
                          install it.
      ⏵ 10 checks ok

The problems come first and are never cut off: a doctor is run to find what
is broken, and the old screen put a giant banner above the table and cut
long details at the right edge, which is where the fix ("Run 'ovat setup'")
was. The checks that passed fold into one row that opens on a click or
Ctrl+O, the same as a tool group.

The checks themselves are ovat.cli.diagnostics.run_checks, unchanged. They
import openvino and open a socket, so they run on a worker thread and the
block shows a spinner meanwhile.
"""
import os
import platform
import sys

from rich.table import Table
from rich.text import Text
from textual.containers import Vertical

from ovat.cli import diagnostics, ui
from ovat.tui.vibe.cli.textual_ui.widgets.collapsible import \
    HeaderCollapsibleSection
from ovat.tui.vibe.cli.textual_ui.widgets.no_markup_static import \
    NoMarkupStatic
from ovat.tui.vibe.cli.textual_ui.widgets.status_message import (
    IndicatorState, StatusMessage)

_GLYPH = {"fail": ("✕", ui.RED), "warn": ("!", ui.YELLOW),
          "ok": ("✓", ui.GREEN)}


def machine_line(config_path: str | None) -> str:
    """"Linux x86_64 · Python 3.13.16 · w.yml": what the checks ran on."""
    os_name = {"darwin": "macOS", "win32": "Windows"}.get(
        sys.platform, platform.system())
    parts = [f"{os_name} {platform.machine()}",
             f"Python {sys.version.split()[0]}"]
    if config_path:
        parts.append(os.path.basename(config_path))
    return " · ".join(parts)


def counts(checks: list) -> dict:
    found = {"fail": 0, "warn": 0, "ok": 0}
    for c in checks:
        found[c.status] = found.get(c.status, 0) + 1
    return found


def verdict(checks: list) -> str:
    """"1 fail · 1 warn · 10 ok", worst first, zeros left out."""
    n = counts(checks)
    return " · ".join(f"{n[k]} {k}" for k in ("fail", "warn", "ok")
                      if n.get(k)) or "no checks ran"


def report_text(checks: list, config_path: str | None = None) -> str:
    """The plain-text report `/doctor copy` puts on the clipboard: worst
    first, no colour, so it pastes cleanly into an issue or a chat."""
    rows = diagnostics.worst_first(checks)
    width = max((len(c.name) for c in rows), default=0)
    lines = [f"ovat doctor: {verdict(checks)}  ({machine_line(config_path)})",
             ""]
    lines += [f"{c.status:<4}  {c.name.ljust(width)}  {c.detail}"
              for c in rows]
    return "\n".join(lines)


def _rows(checks: list, name_width: int = 0) -> Table:
    """One row per check: glyph, name, detail. The detail WRAPS rather than
    being cut, and every cell is Text, so a bracket in a path or an error
    message is shown as a bracket, never read as markup. name_width lines the
    folded ok rows up with the problem rows above them."""
    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True, min_width=name_width)
    table.add_column(ratio=1, overflow="fold")
    for c in checks:
        glyph, colour = _GLYPH.get(c.status, ("?", ui.DIM))
        table.add_row(Text(glyph, style=f"bold {colour}"),
                      Text(c.name, style="bold" if c.status != "ok" else ""),
                      Text(c.detail, style=ui.DIM if c.status == "ok" else ""))
    return table


class _Status(StatusMessage):
    def __init__(self) -> None:
        self.text = "doctor  running the checks"
        super().__init__(classes="doctor-status")

    def get_content(self) -> str:
        return self.text


class DoctorReport(Vertical):
    """One run of the checks, drawn into the transcript."""

    DEFAULT_CSS = """
    DoctorReport { height: auto; width: 100%; margin-top: 1; }
    DoctorReport .doctor-problems { height: auto; padding-left: 2; }
    DoctorReport .doctor-machine { color: $text-muted; padding-left: 2; }
    DoctorReport .doctor-ok { padding-left: 2; height: auto; }
    """

    def __init__(self, config_path: str | None = None, *,
                 run_checks=None) -> None:
        super().__init__()
        self.config_path = config_path or None
        self._run_checks = run_checks
        self.checks: list | None = None
        self.error: str | None = None
        self._status = _Status()

    def compose(self):
        yield self._status

    def on_mount(self) -> None:
        self.run_worker(self._work, thread=True, exit_on_error=False)

    def _work(self) -> None:
        run_checks = self._run_checks
        if run_checks is None:
            run_checks = diagnostics.run_checks
        try:
            checks = list(run_checks(self.config_path))
        except Exception as exc:
            # A check never raises; a broken import underneath can, and must
            # not take the app down with it.
            self.app.call_from_thread(self._show_error,
                                      f"{type(exc).__name__}: {exc}")
            return
        self.app.call_from_thread(self._show, checks)

    def _show_error(self, message: str) -> None:
        if not self.is_mounted:
            return
        self.error = message
        self._status.text = f"doctor  could not run the checks: {message}"
        self._status.settle(IndicatorState.ERROR)

    async def _show(self, checks: list) -> None:
        if not self.is_mounted:
            return          # /new cleared the transcript while it ran
        self.checks = checks
        n = counts(checks)
        self._status.text = f"doctor  {verdict(checks)}"
        self._status.settle(IndicatorState.ERROR if n["fail"] else
                            IndicatorState.MUTED if n["warn"] else
                            IndicatorState.SUCCESS)
        width = max((len(c.name) for c in checks), default=0)
        widgets = [NoMarkupStatic(machine_line(self.config_path),
                                  classes="doctor-machine")]
        problems = diagnostics.problems(checks)
        if problems:
            widgets.append(NoMarkupStatic(_rows(problems, width),
                                          classes="doctor-problems"))
        passed = [c for c in checks if c.status == "ok"]
        if passed:
            section = HeaderCollapsibleSection(
                NoMarkupStatic(_rows(passed, width)),
                header_text=f"{len(passed)} checks ok")
            section.add_class("doctor-ok")
            widgets.append(section)
        await self.mount_all(widgets)

    @property
    def report(self) -> str:
        return report_text(self.checks or [], self.config_path)
