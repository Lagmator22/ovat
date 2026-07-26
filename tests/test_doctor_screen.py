# tests/test_doctor_screen.py
"""Tests for the in-app doctor screen, headless via Textual's Pilot.

The real diagnostics are the seam: run_checks is monkeypatched so these run in
milliseconds and can script exactly the states that used to break rendering.
"""
import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, RichLog

from ovat.cli import diagnostics
from ovat.cli.diagnostics import Check
from ovat.cli.doctor_screen import DoctorScreen, _report_text
from ovat.cli.tui import OvatTUI


def _run(coro):
    asyncio.run(coro)


FAKE_CHECKS = [
    Check("Python", "ok", "3.11.9 (3.10+ required)"),
    Check("OVMS serving", "warn", "ovms not found on PATH"),
    Check("Workflow config", "fail", "invalid: model=Q[/b]8B agent=[native]"),
]


async def _open_doctor(app, pilot, config_path=None):
    screen = DoctorScreen(config_path=config_path)
    app.push_screen(screen)
    await app.workers.wait_for_complete()      # the checks worker
    await pilot.pause()
    return screen


def _log_text(screen) -> str:
    log = screen.query_one("#doc-log", RichLog)
    return "\n".join(str(line) for line in log.lines)


def test_a_bracket_in_a_check_detail_does_not_break_the_render(monkeypatch):
    """Table cells must be Text, not str.

    A Table renders a str cell as markup, so "model=Q[/b]8B" would raise
    MarkupError mid-render. This is the one place ui.esc() cannot reach,
    because the value never passes through the CLI console.
    """
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            text = _log_text(screen)
            assert "Q[/b]8B" in text            # rendered literally
            assert "Python" in text
            assert "1 fail" in text             # and the summary counted it
    _run(scenario())


def test_refresh_re_runs_the_checks(monkeypatch):
    runs = []

    def counting(cfg=None):
        runs.append(cfg)
        return FAKE_CHECKS
    monkeypatch.setattr(diagnostics, "run_checks", counting)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            assert len(runs) == 1
            screen.query_one("#doc-input", Input).value = "/refresh"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(runs) == 2               # re-ran in place, no subprocess
    _run(scenario())


def test_checks_that_blow_up_are_reported_not_crashed(monkeypatch):
    def exploding(cfg=None):
        raise RuntimeError("openvino import died")
    monkeypatch.setattr(diagnostics, "run_checks", exploding)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            text = _log_text(screen)
            assert "could not run the checks" in text
            assert "openvino import died" in text
            assert app.screen is screen         # still usable, not torn down
    _run(scenario())


def test_copy_puts_a_plain_text_report_on_the_clipboard(monkeypatch):
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)
    copied = []

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            screen.query_one("#doc-input", Input).value = "/copy"
            await pilot.press("enter")
            await pilot.pause()
            assert len(copied) == 1
            assert "OVMS serving" in copied[0]
            assert "\x1b" not in copied[0]      # plain text, no escape codes
    _run(scenario())


def test_unknown_slash_command_gets_a_hint(monkeypatch):
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            screen.query_one("#doc-input", Input).value = "/refesh"
            await pilot.press("enter")
            await pilot.pause()
            assert "unknown doctor command /refesh" in _log_text(screen)
    _run(scenario())


def test_escape_returns_to_the_launcher(monkeypatch):
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is not screen
    _run(scenario())


def test_slash_doctor_opens_the_screen_and_passes_the_config(monkeypatch, tmp_path):
    """The whole feature hung on this: /doctor used to insert `ovat doctor `
    as text, so the action branch that opens this screen was unreachable."""
    seen = []

    def capture(cfg=None):
        seen.append(cfg)
        return FAKE_CHECKS
    monkeypatch.setattr(diagnostics, "run_checks", capture)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            app._cwd = str(tmp_path)
            app.query_one("#prompt", Input).value = "/doctor w.yml"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, DoctorScreen)
            # Relative paths resolve against the TUI's cd-tracked directory.
            assert seen == [str(tmp_path / "w.yml")]
    _run(scenario())


def test_report_text_is_readable_without_a_terminal():
    report = _report_text(FAKE_CHECKS)
    assert "Python" in report and "ok" in report
    assert "1 ok · 1 warn · 1 fail" in report
