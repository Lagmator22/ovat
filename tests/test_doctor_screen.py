# tests/test_doctor_screen.py
"""Tests for the in-app doctor screen, headless via Textual's Pilot.

The real diagnostics are the seam: run_checks is monkeypatched so these run in
milliseconds and can script exactly the states that used to break rendering.
"""
import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, OptionList, RichLog

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


def test_the_advertised_key_actually_re_runs_the_checks(monkeypatch):
    """Press the key, do not call the action. That is the whole regression.

    The screen first advertised "r re-run" and "c copy", but the input is
    focused so the user can type slash commands, and a focused Input eats
    every printable character. Pressing r put an "r" in the box and the log
    answered "unknown doctor command r". Only a non-printable key reaches the
    screen, so the binding is f5.
    """
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
            await pilot.press("f5")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(runs) == 2                    # the key really fired
            assert screen.query_one("#doc-input", Input).value == ""
    _run(scenario())


def test_a_printable_key_types_into_the_box_and_does_not_act(monkeypatch):
    """The other half: r must NOT be a shortcut, or it could not be typed."""
    runs = []

    def counting(cfg=None):
        runs.append(cfg)
        return FAKE_CHECKS
    monkeypatch.setattr(diagnostics, "run_checks", counting)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            await pilot.press("r")
            await pilot.pause()
            assert len(runs) == 1                    # nothing re-ran
            assert screen.query_one("#doc-input", Input).value == "r"
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


# The slash menu

def test_typing_slash_opens_a_filtered_menu(monkeypatch):
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            palette = screen.query_one("#doc-palette", OptionList)
            assert palette.display is False

            screen.query_one("#doc-input", Input).value = "/"
            await pilot.pause()
            assert palette.display is True
            assert palette.option_count == 4          # every command

            screen.query_one("#doc-input", Input).value = "/c"
            await pilot.pause()
            assert palette.option_count == 2          # /copy and /clear
    _run(scenario())


def test_choosing_from_the_menu_runs_the_command(monkeypatch):
    runs = []

    def counting(cfg=None):
        runs.append(cfg)
        return FAKE_CHECKS
    monkeypatch.setattr(diagnostics, "run_checks", counting)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            inp = screen.query_one("#doc-input", Input)
            palette = screen.query_one("#doc-palette", OptionList)

            inp.value = "/ref"
            await pilot.pause()
            await pilot.press("down")                 # step into the menu
            await pilot.pause()
            await pilot.press("enter")                # choose /refresh
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert len(runs) == 2                     # it really re-ran
            assert palette.display is False           # menu closed after use
            assert inp.value == ""                    # and the line was cleared
    _run(scenario())


def test_escape_closes_the_menu_before_leaving_the_screen(monkeypatch):
    """Otherwise opening the menu is a one-way trip without a mouse."""
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            palette = screen.query_one("#doc-palette", OptionList)
            screen.query_one("#doc-input", Input).value = "/"
            await pilot.pause()
            assert palette.display is True

            await pilot.press("escape")               # first Esc: close menu
            await pilot.pause()
            assert palette.display is False
            assert app.screen is screen               # still on the screen

            await pilot.press("escape")               # second Esc: leave
            await pilot.pause()
            assert app.screen is not screen
    _run(scenario())
