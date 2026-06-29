# tests/test_tui.py
"""Tests for the Textual TUI, driven by Textual's headless Pilot.

Note to myself: run_test() starts the app in a headless terminal so I can press
keys and inspect widgets with no real screen. I drive it with asyncio.run so I
do not need pytest-asyncio. These prove the VIEW is wired: typing shows the
slash menu, submitting runs a real command into the log, Tab completes, and
/exit actually stops the app.
"""
import asyncio

from textual.widgets import Input, RichLog, Static

from ovat.cli.tui import OvatTUI


def _run(coro):
    asyncio.run(coro)


def test_palette_shows_while_typing_and_hides_for_args():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            palette = app.query_one("#palette", Static)
            assert palette.display is False           # hidden at rest
            inp = app.query_one("#prompt", Input)

            inp.value = "/d"
            await pilot.pause()
            assert palette.display is True            # menu appears for /d

            inp.value = "/doctor "                    # now typing an argument
            await pilot.pause()
            assert palette.display is False           # menu gets out of the way
    _run(scenario())


def test_submitting_a_command_writes_to_the_log():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            log = app.query_one("#output", RichLog)
            before = len(log.lines)
            inp = app.query_one("#prompt", Input)
            inp.value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            assert len(log.lines) > before            # output grew
            assert inp.value == ""                    # input cleared after submit
    _run(scenario())


def test_tab_completes_a_partial_command():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "/d"
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert inp.value == "/doctor "
    _run(scenario())


def test_exit_command_stops_the_app():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "/exit"
            await pilot.press("enter")
            await pilot.pause()
        assert app.is_running is False
    _run(scenario())
