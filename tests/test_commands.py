# tests/test_commands.py
"""Tests for OVAT's Textual command-palette entries."""
import asyncio
import os

import pytest

pytest.importorskip("textual")

from textual.widgets import RichLog

from ovat.cli.commands import OvatCommands
from ovat.cli.tui import OvatTUI


def _run(coro):
    asyncio.run(coro)


async def _entries(app):
    """(title, help, callback) for every palette entry OVAT contributes."""
    provider = OvatCommands(app.screen)
    return provider._entries


def test_ovat_commands_are_added_not_substituted():
    """Ctrl-P must still offer Textual's own system commands as well."""
    from textual.app import App
    assert OvatCommands in OvatTUI.COMMANDS
    assert App.COMMANDS <= OvatTUI.COMMANDS      # the built-ins survived


def test_palette_offers_a_screenshot_and_every_theme():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            titles = [title for title, _, _ in await _entries(app)]
            assert "Save screenshot" in titles
            for theme in app.available_themes:
                assert f"Theme: {theme}" in titles
    _run(scenario())


def test_each_theme_entry_applies_its_own_theme():
    """The late-binding trap: a closure over the loop variable would make
    every entry set the LAST theme in the list."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            entries = {title: cb for title, _, cb in await _entries(app)}
            for name in ("nord", "dracula", "gruvbox"):
                entries[f"Theme: {name}"]()
                await pilot.pause()
                assert app.theme == name
    _run(scenario())


def test_screenshot_writes_an_svg_and_reports_where(tmp_path):
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            app._cwd = str(tmp_path)
            app.action_save_ovat_screenshot()
            await pilot.pause()
            files = list(tmp_path.glob("*.svg"))
            assert len(files) == 1                       # a real file landed
            # encoding="utf-8" is load-bearing, not decoration. Textual writes
            # the SVG as UTF-8 and the screenshot contains OVAT's box-drawing
            # wordmark, so reading it back with the PLATFORM default decoded as
            # cp1252 on Windows and died on byte 0x90 mid-glyph. The same
            # assumption ui.enable_windows_utf8() exists to fix, made in a test.
            assert files[0].read_text(encoding="utf-8").lstrip().startswith("<")
            log = "\n".join(str(ln) for ln
                            in app.query_one("#output", RichLog).lines)
            assert "screenshot saved" in log             # and it said where
    _run(scenario())


def test_a_failing_screenshot_reports_instead_of_raising(monkeypatch):
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            def boom(*a, **k):
                raise OSError("disk full")
            monkeypatch.setattr(app, "save_screenshot", boom)
            app.action_save_ovat_screenshot()            # must not raise
            await pilot.pause()
            log = "\n".join(str(ln) for ln
                            in app.query_one("#output", RichLog).lines)
            assert "could not save the screenshot" in log
            assert "disk full" in log
    _run(scenario())


def test_palette_actions_report_safely_from_a_screen_without_the_log(
        monkeypatch, tmp_path):
    """#output only exists on the launcher, but the palette works everywhere."""
    from ovat.cli.doctor_screen import DoctorScreen
    from ovat.cli import diagnostics
    from ovat.cli.diagnostics import Check

    # monkeypatch, never a bare assignment: a plain `diagnostics.run_checks =`
    # here leaked the fake into every later test in the session, and
    # test_doctor's bad-config case stopped failing as it should.
    monkeypatch.setattr(diagnostics, "run_checks",
                        lambda cfg=None: [Check("x", "ok", "y")])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            app._cwd = str(tmp_path)
            app.push_screen(DoctorScreen())
            await app.workers.wait_for_complete()
            await pilot.pause()
            app.action_save_ovat_screenshot()            # falls back to notify
            await pilot.pause()
            assert len(list(tmp_path.glob("*.svg"))) == 1
    _run(scenario())


def test_ovat_palette_is_registered_as_the_default_theme():
    """Widgets are written against $primary/$success, so the brand has to BE
    a theme; otherwise picking Dracula recoloured Textual's chrome and left
    OVAT's widgets hardcoded Intel blue on top of it."""
    from ovat.cli.theme import OVAT_THEME
    from ovat.cli import ui

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            assert app.theme == "ovat"
            theme = app.get_theme("ovat")
            assert theme.primary == ui.BLUE          # derived from ui.PALETTE
            assert theme.success == ui.GREEN
            assert theme.error == ui.RED
    _run(scenario())


def test_switching_theme_from_the_palette_still_works():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            entries = {title: cb for title, _, cb in await _entries(app)}
            entries["Theme: dracula"]()
            await pilot.pause()
            assert app.theme == "dracula"            # the chat widgets follow it
    _run(scenario())
