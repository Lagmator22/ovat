# tests/test_commands.py
"""Tests for OVAT's command-palette entries.

Two mechanisms, per Textual's docs: App.get_system_commands for app-wide
entries, and a Screen.COMMANDS provider for entries that only apply while a
particular screen is up.
"""
import asyncio

import pytest

pytest.importorskip("textual")

from ovat.cli import diagnostics
from ovat.cli.diagnostics import Check
from ovat.cli.tui import OvatTUI


def _run(coro):
    asyncio.run(coro)


def _titles(app):
    return [c.title for c in app.get_system_commands(app.screen)]


def _fake_chat(monkeypatch):
    from ovat.cli import chat_screen
    from ovat.config.workflow import WorkflowConfig

    class R:
        def retrieve(self, q, top_k=5):
            return []

    class L:
        def chat(self, m, tools=None, on_token=None):
            return {"finish_reason": "stop", "content": "A", "tool_calls": None}

    monkeypatch.setattr(chat_screen, "_build_components",
                        lambda c, m, max_tokens=256:
                        (WorkflowConfig(model={"name": "m"}), R(), L()))


def test_textual_builtins_are_kept_not_replaced():
    """Theme, Screenshot, Quit and Keys are Textual's own.

    An earlier version of this shipped an OVAT "Save screenshot" and 21
    "Theme: name" entries, which put two palette entries in front of the user
    for one job and gave the worse one first.
    """
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            titles = _titles(app)
            for builtin in ("Theme", "Quit", "Screenshot", "Keys"):
                assert builtin in titles
            assert titles.count("Theme") == 1
            assert titles.count("Screenshot") == 1
            assert not any(t.startswith("Theme: ") for t in titles)
    _run(scenario())


def test_ovat_adds_its_own_app_wide_entries():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            titles = _titles(app)
            assert "Doctor" in titles
            assert "Chat" in titles
    _run(scenario())


def test_the_doctor_entry_opens_the_doctor_screen(monkeypatch):
    from ovat.cli.doctor_screen import DoctorScreen
    monkeypatch.setattr(diagnostics, "run_checks",
                        lambda cfg=None: [Check("Python", "ok", "3.11")])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            command = next(c for c in app.get_system_commands(app.screen)
                           if c.title == "Doctor")
            command.callback()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, DoctorScreen)
    _run(scenario())


# Screen-scoped commands

def test_chat_commands_are_scoped_to_the_chat_screen(monkeypatch, tmp_path):
    """"Copy the last answer" means nothing on the launcher, so it must not
    be offered there."""
    from ovat.cli.chat_screen import ChatCommands, ChatScreen
    _fake_chat(monkeypatch)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            assert ChatCommands not in OvatTUI.COMMANDS      # not app-wide
            screen = ChatScreen("w.yml", "m", cwd=str(tmp_path))
            app.push_screen(screen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ChatCommands in ChatScreen.COMMANDS       # scoped to here

            titles = [t for t, _, _ in ChatCommands(screen).commands()]
            assert "Copy the last answer" in titles
            assert "Show reasoning" in titles
    _run(scenario())


def test_the_reasoning_entry_describes_what_it_will_do(monkeypatch, tmp_path):
    """Show while hidden, Hide while shown: the palette never offers to do
    the thing that already happened."""
    from ovat.cli.chat_screen import ChatCommands, ChatScreen
    _fake_chat(monkeypatch)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = ChatScreen("w.yml", "m", cwd=str(tmp_path))
            app.push_screen(screen)
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert "Show reasoning" in [t for t, _, _
                                        in ChatCommands(screen).commands()]
            screen._show_thinking = True
            assert "Hide reasoning" in [t for t, _, _
                                        in ChatCommands(screen).commands()]
    _run(scenario())


def test_doctor_commands_are_scoped_to_the_doctor_screen(monkeypatch):
    from ovat.cli.doctor_screen import DoctorCommands, DoctorScreen
    monkeypatch.setattr(diagnostics, "run_checks",
                        lambda cfg=None: [Check("Python", "ok", "3.11")])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = DoctorScreen()
            app.push_screen(screen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert DoctorCommands in DoctorScreen.COMMANDS
            titles = [t for t, _, _ in DoctorCommands(screen).commands()]
            assert titles == ["Re-run the checks", "Copy the report"]
    _run(scenario())


# Discoverability: a palette nobody knows about is no feature at all

def test_the_footer_advertises_the_palette_on_every_screen(monkeypatch):
    from textual.widgets import Footer
    from ovat.cli.doctor_screen import DoctorScreen
    monkeypatch.setattr(diagnostics, "run_checks",
                        lambda cfg=None: [Check("Python", "ok", "3.11")])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            assert app.query(Footer)                     # the launcher
            app.push_screen(DoctorScreen())
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.screen.query(Footer)              # and the doctor screen
    _run(scenario())


def test_the_palette_binding_is_shown_not_hidden():
    """show=True is what puts it in the Footer. show=False would leave the
    palette existing with nothing on screen ever mentioning it."""
    palette = [b for b in OvatTUI.BINDINGS
               if getattr(b, "action", None) == "command_palette"]
    assert palette, "no binding for the command palette"
    assert all(b.show for b in palette)


def test_the_welcome_text_names_the_shortcut():
    from textual.widgets import RichLog

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            log = "\n".join(str(line) for line
                            in app.query_one("#output", RichLog).lines)
            assert "Ctrl-P" in log
    _run(scenario())
