# tests/test_tui.py
"""Tests for the Textual TUI, driven by Textual's headless Pilot.

Note to myself: run_test() starts the app in a headless terminal so I can press
keys and inspect widgets with no real screen, driven by asyncio.run so I do not
need pytest-asyncio. These prove the view is wired: the slash dropdown appears
and fills the input, a real command streams into the log, and /exit quits.
"""
import asyncio

import pytest

# The TUI is an optional extra (pip install 'ovat[tui]'). Without it these
# tests skip cleanly; the isolation contract lives in test_tui_isolation.py.
pytest.importorskip("textual")

from textual.widgets import Input, OptionList, RichLog
from textual.app import App, ComposeResult

from ovat.cli import tui as tui_module
from ovat.cli.animated_gif import AnimatedGif
from ovat.cli.tui import OvatTUI
from tests.conftest import py_command


def _run(coro):
    asyncio.run(coro)


def test_animated_gif_plays_once_and_preserves_frame_durations(tmp_path):
    """The startup GIF caches its frames then leaves its final frame static."""
    from PIL import Image

    source = tmp_path / "animation.gif"
    Image.new("RGB", (4, 4), "red").save(
        source,
        save_all=True,
        append_images=[Image.new("RGB", (4, 4), "blue")],
        duration=[20, 80],
        loop=0,
    )

    class GifApp(App):
        def compose(self) -> ComposeResult:
            yield AnimatedGif(source, columns=4, id="animation")

    async def scenario():
        app = GifApp()
        async with app.run_test() as pilot:
            animation = app.query_one("#animation", AnimatedGif)
            animation.pause()
            assert animation.frame_count == 2
            assert [frame.duration for frame in animation._frames] == [0.02, 0.08]
            first = animation.render()
            animation.play()
            await pilot.pause(0.04)
            animation.pause()
            assert animation.frame_index == 1
            assert animation.render() is not first
            animation.play()
            await pilot.pause(0.1)
            assert animation.frame_index == 1
            assert animation.is_playing is False
            animation.restart()
            assert animation.is_playing is True
    _run(scenario())


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
            # /index still INSERTS text; /doctor is now a UI action that opens
            # the doctor screen, so it is not the example to use here.
            inp.value = "/ind"
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert inp.value == "ovat index  "         # template inserted, ready to edit
    _run(scenario())


def test_tab_on_an_action_template_runs_it_instead_of_inserting_text():
    """/doctor is an action, so completing it opens the screen, not text."""
    from ovat.cli.doctor_screen import DoctorScreen

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "/doc"
            await pilot.pause()
            await pilot.press("tab")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, DoctorScreen)
            assert inp.value == "/doc"                 # nothing was inserted
    _run(scenario())


def test_run_template_puts_the_cursor_in_the_config_gap():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "/run"
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert inp.value == 'ovat run  --input ""'
            # The next keystrokes are the config path, so the cursor must sit
            # in the gap after "ovat run ", not at the end of the line.
            assert inp.cursor_position == len("ovat run ")
    _run(scenario())


def test_ctrl_c_cancels_the_child_but_keeps_the_app_alive():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = py_command("import time;time.sleep(30)")
            await pilot.press("enter")
            for _ in range(100):
                if app._proc is not None:
                    break
                await pilot.pause(0.05)
            proc = app._proc
            await pilot.press("ctrl+c")            # busy -> cancel, not quit
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert proc.poll() is not None         # child gone
            assert app.is_running is True          # app survived
    _run(scenario())


def test_one_ctrl_c_warns_and_the_second_quits():
    """Ctrl-C is the reflex for "interrupt", so one press must not end the app."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.is_running is True              # still here
            # A toast, not a log line: see the cross-screen test below.
            assert any("again to quit" in str(n.message)
                       for n in app._notifications)

            await pilot.press("ctrl+c")
            await pilot.pause()
        assert app.is_running is False


def test_a_stale_first_ctrl_c_does_not_count_as_half_a_quit():
    """Two presses minutes apart are two interrupts, not one quit."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+c")
            await pilot.pause()
            # Age the first press past the confirmation window.
            app._last_quit_press -= tui_module.QUIT_CONFIRM_S + 1
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.is_running is True              # warned again, not quit
    _run(scenario())


def test_placeholder_shows_running_state_and_restores():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            # sleep, not echo: an echo can FINISH before the assertion runs,
            # and then we would see the already-restored placeholder.
            inp.value = py_command("import time;time.sleep(30)")
            await pilot.press("enter")
            assert "running" in inp.placeholder    # set synchronously on submit
            app._cancel_running()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "type / for shortcuts" in inp.placeholder
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


def test_cd_persists_the_working_directory():
    async def scenario():
        import os
        app = OvatTUI()
        async with app.run_test() as pilot:
            start = app._cwd
            inp = app.query_one("#prompt", Input)
            inp.value = "cd .."
            await pilot.press("enter")
            await pilot.pause()
            assert app._cwd == os.path.normpath(os.path.join(start, ".."))
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


# The busy gate (no double-spawn, no orphaned processes)

def test_second_submit_while_busy_is_refused_not_double_spawned():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = py_command("import time;time.sleep(5)")
            await pilot.press("enter")            # gate closes on this thread,
            first_proc_gate = app._busy           # synchronously
            inp.value = "echo should_be_refused"
            await pilot.press("enter")
            await pilot.pause()
            log_text = "\n".join(str(ln) for ln in app.query_one("#output", RichLog).lines)
            assert first_proc_gate is True
            assert "still running" in log_text    # the refusal, not a 2nd spawn
            app._cancel_running()                 # clean up the sleeper
            await app.workers.wait_for_complete()
    _run(scenario())


def test_gate_reopens_after_a_command_finishes():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "echo one"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._busy is False             # gate reopened
            inp.value = "echo two"                # and a second command runs fine
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            log_text = "\n".join(str(ln) for ln in app.query_one("#output", RichLog).lines)
            assert "two" in log_text
    _run(scenario())


def test_escape_terminates_the_child_process():
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = py_command("import time;time.sleep(30)")
            await pilot.press("enter")
            # Wait until the worker has actually spawned the process.
            for _ in range(100):
                if app._proc is not None:
                    break
                await pilot.pause(0.05)
            proc = app._proc
            assert proc is not None
            await pilot.press("escape")           # the cancel path
            await app.workers.wait_for_complete() # worker sees EOF + exit
            await pilot.pause()
            assert proc.poll() is not None        # the child is really dead
            assert app._busy is False             # and the gate reopened
    _run(scenario())


def test_absolute_path_command_runs_instead_of_being_read_as_a_shortcut():
    """"/usr/bin/env foo" is a command, not a typo'd shortcut.

    The TUI promises that anything you type runs as a real command, but every
    line starting with "/" was matched against the slash templates and, when
    it did not match, refused. On macOS/Linux that swallowed EVERY
    absolute-path command. One slash still means a shortcut; more than one
    means a path.
    """
    import sys as _sys

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = f"{_sys.executable} -c \"print('OVAT_ABS_PATH_OK')\""
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            log_text = "\n".join(str(ln) for ln
                                 in app.query_one("#output", RichLog).lines)
            assert "OVAT_ABS_PATH_OK" in log_text      # it really ran
            assert "Unknown shortcut" not in log_text
    _run(scenario())


def test_a_mistyped_slash_shortcut_still_gets_a_hint():
    """The other half of the same rule: "/doctr" must not be run as a command."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "/doctr"
            await pilot.press("enter")
            await pilot.pause()
            log_text = "\n".join(str(ln) for ln
                                 in app.query_one("#output", RichLog).lines)
            assert "Unknown shortcut" in log_text
            assert app._proc is None                   # nothing was spawned
    _run(scenario())


# The recursion guard (a TUI inside the TUI)

def test_run_tui_refuses_without_a_real_terminal(capsys):
    # pytest captures stdout, so isatty() is False here; exactly the situation
    # inside a TUI subprocess or a pipe. run_tui must print a hint and return
    # instead of starting a full-screen app into the void.
    from ovat.cli.tui import run_tui
    run_tui()
    assert "needs a real terminal" in capsys.readouterr().out


def test_bare_ovat_inside_the_tui_env_prints_a_hint_not_a_tui(monkeypatch):
    from typer.testing import CliRunner
    from ovat.cli.main import app as cli_app
    monkeypatch.setenv("OVAT_TUI", "1")           # what shell.venv_env stamps
    result = CliRunner().invoke(cli_app, [])
    assert result.exit_code == 0
    assert "already inside" in result.output


def test_the_quit_warning_is_visible_from_a_pushed_screen(monkeypatch):
    """It used to be written somewhere the user could not see.

    App.query_one searches the whole DOM, so writing the warning to the
    launcher's #output SUCCEEDED from the chat screen and put it on a screen
    that was not in front: the first Ctrl-C looked like nothing happened and
    the second quit, losing the conversation. A toast follows the user.
    """
    from ovat.cli import chat_screen
    from ovat.cli.chat_screen import ChatScreen
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

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            app.push_screen(ChatScreen("w.yml", "m", cwd="/tmp"))
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.is_running is True
            assert any("again to quit" in str(n.message)
                       for n in app._notifications)
    _run(scenario())


def test_opening_a_screen_you_are_already_on_does_not_stack_it(monkeypatch):
    """The palette works everywhere, so "Chat" from a chat used to push a
    SECOND ChatScreen: another ~30s model load, and Esc twice to get out."""
    from ovat.cli import chat_screen, diagnostics
    from ovat.cli.chat_screen import ChatScreen
    from ovat.cli.diagnostics import Check
    from ovat.cli.doctor_screen import DoctorScreen
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
    monkeypatch.setattr(diagnostics, "run_checks",
                        lambda cfg=None: [Check("Python", "ok", "3.11")])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            app.push_screen(ChatScreen("w.yml", "m", cwd="/tmp"))
            await app.workers.wait_for_complete()
            await pilot.pause()
            depth = len(app.screen_stack)

            app._open_chat("")                       # already here
            await pilot.pause()
            assert len(app.screen_stack) == depth

            app.push_screen(DoctorScreen())
            await app.workers.wait_for_complete()
            await pilot.pause()
            depth = len(app.screen_stack)
            app._open_doctor("")                     # already here: re-runs
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(app.screen_stack) == depth
    _run(scenario())


def test_the_launcher_recalls_commands_with_up_and_down():
    """It is a shell front-end, so Up/Down must do what the shell does."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "echo one"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            inp.value = "echo two"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            inp.value = "unfinished"
            await pilot.press("up")
            await pilot.pause()
            assert inp.value == "echo two"
            await pilot.press("up")
            await pilot.pause()
            assert inp.value == "echo one"
            await pilot.press("down")
            await pilot.press("down")
            await pilot.pause()
            assert inp.value == "unfinished"
    _run(scenario())
