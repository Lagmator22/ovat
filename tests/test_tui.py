# tests/test_tui.py
"""Tests for the Textual TUI, driven by Textual's headless Pilot.

Note to myself: run_test() starts the app in a headless terminal so I can press
keys and inspect widgets with no real screen, driven by asyncio.run so I do not
need pytest-asyncio. These prove the view is wired: the slash dropdown appears
and fills the input, a real command streams into the log, and /exit quits.
"""
import asyncio

from textual.widgets import Input, OptionList, RichLog

from ovat.cli.tui import OvatTUI


def _run(coro):
    asyncio.run(coro)


def test_slash_opens_a_dropdown_of_templates():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            palette = app.query_one("#palette", OptionList)
            assert palette.display is False
            inp = app.query_one("#prompt", Input)
            inp.value = "/d"
            await pilot.pause()
            assert palette.display is True
            assert palette.option_count >= 1          # at least /doctor
    _run(scenario())


def test_tab_fills_the_command_template():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "/doc"
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert inp.value == "ovat doctor "         # template inserted, ready to edit
    _run(scenario())


def test_running_a_real_command_streams_into_the_log():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            log = app.query_one("#output", RichLog)
            before = len(log.lines)
            inp = app.query_one("#prompt", Input)
            inp.value = "echo OVAT_PILOT_123"
            await pilot.press("enter")
            await app.workers.wait_for_complete()       # let the worker finish
            await pilot.pause()
            assert len(log.lines) > before              # output landed in the log
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
