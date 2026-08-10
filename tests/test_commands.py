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
from textual.widgets import Input

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
            await pilot.pause()                    # let on_mount fire
            for _ in range(60):
                await app.workers.wait_for_complete()
                await pilot.pause()
                if screen.query("#chat-view") and not any(
                        w.is_running for w in app.workers):
                    break
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
            await pilot.pause()                    # let on_mount fire
            for _ in range(60):
                await app.workers.wait_for_complete()
                await pilot.pause()
                if screen.query("#chat-view") and not any(
                        w.is_running for w in app.workers):
                    break
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
            assert titles == ["Re-run the checks", "Copy the report",
                              "Sort by severity"]
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


def test_the_palette_binding_is_textuals_own_not_a_duplicate():
    """Declaring our own ctrl+p listed "^p Palette" TWICE in the footer, once
    at each end, because Textual already binds it via COMMAND_PALETTE_BINDING
    and shows it."""
    ours = [b for b in OvatTUI.BINDINGS
            if getattr(b, "action", None) == "command_palette"]
    assert ours == [], "the palette binding is Textual's; do not re-declare it"
    assert OvatTUI.COMMAND_PALETTE_BINDING == "ctrl+p"


def test_the_welcome_text_names_the_shortcut():
    from textual.widgets import RichLog

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            log = "\n".join(str(line) for line
                            in app.query_one("#output", RichLog).lines)
            assert "Ctrl-P" in log
    _run(scenario())


def test_a_reset_theme_command_appears_only_once_you_have_left_ovat():
    """Several built-in themes fight OVAT's greens and blues, and finding
    "ovat" again in an alphabetical list of 22 is a chore."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            assert app.theme == "ovat"
            assert "Reset theme" not in _titles(app)   # nothing to reset yet

            app.theme = "dracula"
            await pilot.pause()
            assert "Reset theme" in _titles(app)

            command = next(c for c in app.get_system_commands(app.screen)
                           if c.title == "Reset theme")
            command.callback()
            await pilot.pause()
            assert app.theme == "ovat"
    _run(scenario())


def test_every_screen_follows_the_theme_not_just_the_chat():
    """The chat was converted to theme variables; the launcher and doctor
    screen were not, so picking Dracula recoloured one screen and left the
    other two Intel blue. Every screen's chrome is a theme variable now."""
    from ovat.cli.doctor_screen import DoctorScreen

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            launcher = app.query_one("#output")
            ovat_blue = launcher.styles.border[0][1]

            app.theme = "dracula"
            await pilot.pause()
            await pilot.pause()
            dracula = launcher.styles.border[0][1]
            assert dracula != ovat_blue          # the launcher moved

            app.push_screen(DoctorScreen())
            await app.workers.wait_for_complete()
            await pilot.pause()
            # The results TABLE is the doctor screen's main element and the
            # one carrying $primary; the message strip below it is $secondary.
            doctor = app.screen.query_one("#doc-table")
            assert doctor.styles.border[0][1] == dracula   # and agrees
    _run(scenario())


def test_no_hardcoded_hex_survives_in_any_screen_css():
    """A literal hex in CSS is a colour a theme cannot reach."""
    import re
    from ovat.cli import chat_screen, doctor_screen, tui

    for module in (tui.OvatTUI, doctor_screen.DoctorScreen,
                   chat_screen.ChatScreen, chat_screen.SessionPicker):
        css = getattr(module, "CSS", "") or getattr(module, "DEFAULT_CSS", "")
        leftover = re.findall(r"#[0-9A-Fa-f]{6}", css)
        assert not leftover, f"{module.__name__} pins {leftover}"


def test_a_no_theme_entry_hands_colour_back_to_the_terminal():
    """ansi-dark maps onto the sixteen colours your terminal already
    defines, so OVAT stops arguing with a scheme you already chose."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            assert "No theme" in _titles(app)
            command = next(c for c in app.get_system_commands(app.screen)
                           if c.title == "No theme")
            command.callback()
            await pilot.pause()
            assert app.theme == "ansi-dark"
            # And once you are there it stops offering itself, but does offer
            # the way back.
            assert "No theme" not in _titles(app)
            assert "Reset theme" in _titles(app)
    _run(scenario())


def test_child_commands_are_told_a_width_that_fits_the_scrollbar():
    """A rich Panel drawn to the full width lost its right edge and bottom
    corner behind Textual's overlaying scrollbar, and the box came apart."""
    from textual.widgets import RichLog

    captured = {}

    async def scenario():
        app = OvatTUI()
        async with app.run_test(size=(100, 24)) as pilot:
            log = app.query_one("#output", RichLog)

            def fake_run(cmd, cols):
                captured["cols"] = cols
            app._run_command = fake_run
            app.query_one("#prompt", Input).value = "echo hi"
            await pilot.press("enter")
            await pilot.pause()

            assert captured["cols"] == (log.content_size.width
                                        - log.scrollbar_size_vertical)
            assert captured["cols"] < log.content_size.width   # room reserved
    _run(scenario())
