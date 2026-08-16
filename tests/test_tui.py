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

from textual.containers import Horizontal
from textual.widgets import Input, OptionList, RichLog, Static

from ovat.cli import tui as tui_module
from ovat.cli.tui import OvatTUI, _StartupIntelAnimation, _secondary_wordmark
from tests.conftest import py_command


def _run(coro):
    asyncio.run(coro)


def test_launcher_has_isolated_three_panel_startup_header():
    """The launcher owns the startup mark; no other screen or command does."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test():
            assert app.query_one("#masthead", Horizontal)
            assert app.query_one("#brand-panel", Static)
            assert app.query_one("#updates-panel", Static)
            assert app.query_one("#intel-animation", _StartupIntelAnimation)
            assert "/doctor" not in app.query_one("#updates-panel", Static).render().plain
    _run(scenario())


def test_secondary_startup_labels_reuse_the_real_ovat_wordmark_face():
    """Sub-brands and Updates use ANSI Shadow rather than a generic block font."""
    label = _secondary_wordmark("Updates", color="#8F5CFF")
    assert label.plain.count("\n") == 6
    assert "█" in label.plain


def test_startup_mark_plays_gif_then_three_cached_phases_and_settles(tmp_path):
    """GIF frames preserve timing before Gemini -> clipboard -> static ChatGPT."""
    from PIL import Image

    intro = tmp_path / "intro.gif"
    first = Image.new("RGB", (12, 8), "#001e5a")
    first.save(intro, save_all=True, append_images=[Image.new("RGB", (12, 8), "#0071c5")],
               duration=[20, 80], loop=0)
    sources = []
    for index, colour in enumerate(("#001e5a", "#0071c5", "#ffffff"), start=1):
        source = tmp_path / f"phase-{index}.png"
        Image.new("RGB", (12, 8), colour).save(source)
        sources.append(source)

    class StartupApp(tui_module.App):
        def compose(self):
            yield _StartupIntelAnimation(sources, intro_source=intro, columns=4,
                                         phase_seconds=0.03, id="startup")

    async def scenario():
        app = StartupApp()
        async with app.run_test():
            startup = app.query_one("#startup", _StartupIntelAnimation)
            assert startup.frame_count == 5
            assert [frame.duration for frame in startup._frames] == [0.02, 0.08, 0.03, 0.03, None]
            assert startup.frame_index == 0
            assert startup.is_animating is True
            for expected_index in range(1, 5):
                startup._cancel_timer()
                startup._advance()
                assert startup.frame_index == expected_index
            assert startup.is_animating is False
            final = startup.render()
            assert startup.render() is final
            assert not any(0x2800 <= ord(character) <= 0x28FF
                           for character in final.plain)
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
            # Model what actually failed on Windows CI, rather than hoping
            # to hit a timing window.
            #
            # Assigning `value` sends the cursor to the end -- Input does that
            # in its value watcher, synchronously. The correction used to be
            # scheduled with call_after_refresh, and a scheduled callback only
            # runs once a refresh happens. On a loaded runner that refresh did
            # not arrive while the test was looking, so the cursor was found
            # still at the end of the line: 20, not 9. Pausing longer cannot
            # fix work that has not been scheduled to run.
            #
            # Neutering call_after_refresh reproduces that exactly, and leaves
            # the synchronous placement as the only thing that can pass it.
            app.call_after_refresh = lambda *a, **kw: None
            await pilot.press("tab")
            await pilot.pause()

            expected = len("ovat run ")
            assert inp.value == 'ovat run  --input ""'
            # The next keystrokes are the config path, so the cursor must sit
            # in the gap after "ovat run ", not at the end of the line.
            assert inp.cursor_position == expected, (
                "cursor placement depended on a refresh that never came")
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


def test_search_paths_reads_the_field_and_shrugs_off_a_bad_config(tmp_path):
    """Auto-detection happens before ChatScreen exists, so /chat has to open
    the workflow itself just to learn where models may live.

    A config that will not load must come back empty rather than raise: that
    error is ChatScreen's to report properly, and turning it into "no model on
    this machine" would send the user hunting for the wrong thing.
    """
    from ovat.cli.tui import _search_paths

    good = tmp_path / "good.yml"
    good.write_text("model:\n  name: m\nmodel_search_paths:\n"
                    "  - /somewhere/odd\n", encoding="utf-8")
    assert _search_paths(str(good)) == ["/somewhere/odd"]

    bare = tmp_path / "bare.yml"
    bare.write_text("model:\n  name: m\n", encoding="utf-8")
    assert _search_paths(str(bare)) == []

    broken = tmp_path / "broken.yml"
    broken.write_text("model:\n  name: m\n  nonsense_key: 1\n", encoding="utf-8")
    assert _search_paths(str(broken)) == []
    assert _search_paths(str(tmp_path / "does-not-exist.yml")) == []


def test_tui_chat_autodetect_honours_model_search_paths(tmp_path, monkeypatch):
    """The TUI ignored the setting the whole feature was about.

    `model_search_paths` moved OVAT_MODELS into workflow.yml, and `ovat chat`
    passes cfg.model_search_paths into pick_chat_llm. /chat called
    pick_chat_llm() with no arguments, so a model that lived ONLY in a
    configured folder was found by the CLI and reported as
    "No local text LLM found" by the TUI, against the same config.

    Its fix line then sent the user to OVAT_MODELS -- the variable the config
    was introduced to replace.
    """
    from ovat.cli import chat_screen
    from ovat.core import model_scout

    config = tmp_path / "w.yml"
    config.write_text("model:\n  name: m\nmodel_search_paths:\n"
                      "  - /odd/models/root\n", encoding="utf-8")

    # No remembered model path: a real .ovat/chat_prefs.json in the developer's
    # cwd would satisfy `if not model` and skip auto-detection entirely, so
    # this would describe that machine rather than the code.
    monkeypatch.setattr(chat_screen, "load_prefs", lambda cwd: {})

    seen = {}

    def fake_pick(extra_roots=None):
        seen["pick"] = extra_roots
        return None, []                      # force the not-found branch

    def fake_searched(extra_roots=None):
        seen["searched"] = extra_roots
        return list(extra_roots or []) + ["/conventional/models"]

    monkeypatch.setattr(model_scout, "pick_chat_llm", fake_pick)
    monkeypatch.setattr(model_scout, "searched_roots", fake_searched)
    monkeypatch.setattr(model_scout, "find_models",
                        lambda kind=None, extra_roots=None: [])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            app._open_chat(str(config))
            await pilot.pause()
            log_text = "\n".join(str(ln) for ln
                                 in app.query_one("#output", RichLog).lines)

        # The configured root reached BOTH the search and the "scanned" list.
        assert seen["pick"] == ["/odd/models/root"]
        assert seen["searched"] == ["/odd/models/root"]
        assert "/odd/models/root" in log_text
        # and the fix names the setting, not the variable it replaced.
        assert "model_search_paths" in log_text
        assert "OVAT_MODELS" not in log_text
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


# The masthead hierarchy: three type sizes, largest to smallest

def _rows(text) -> list:
    return text.plain.rstrip("\n").split("\n")


def _styles(text) -> set:
    return {str(span.style) for span in text.spans}


def test_the_masthead_ranks_its_marks_by_size():
    """A terminal has ONE glyph size, so hierarchy comes from the FACE.

    OVAT is the six-row solid-with-shadow face; everything beneath it is the
    same solid family at three rows. The credit is NOT ranked below the
    attribution by weight: an outline face was tried there and read as
    broken rather than as lighter, because at three rows there is not enough
    resolution for an outline to look deliberate. Colour separates them.
    """
    from ovat.cli import tui

    ovat = len(tui._ansi_shadow_lines("OVAT"))
    openvino = len(_rows(tui._attribution_wordmark("OpenVINO")))
    credit = len(_rows(tui._credit_wordmark("powered by")))
    assert ovat > openvino, "the attribution must be smaller than the wordmark"
    assert credit == openvino, "the sub-marks should be one consistent size"


def test_every_sub_mark_is_drawn_in_the_same_solid_face():
    """The bug this guards: the credit was set in an outline face and looked
    like the solid glyphs had failed to render, next to two marks that had
    not."""
    from ovat.cli import tui

    def glyphs(text):
        return {c for r in _rows(text) for c in r if c != " "}

    credit = glyphs(tui._credit_wordmark("powered by"))
    attribution = glyphs(tui._attribution_wordmark("OpenVINO"))
    assert credit <= attribution | credit, "unexpected glyph set"
    # Both draw with block characters, not box-drawing rules.
    assert credit & set("\u2588\u2580\u2584\u2591")
    assert attribution & set("\u2588\u2580\u2584\u2591")


def test_every_masthead_mark_is_shaded_not_flat():
    """Each mark carries a gradient, like the OVAT wordmark, rather than one
    flat colour: that is what gives the block glyphs their depth."""
    from ovat.cli import tui

    for mark in (tui._credit_wordmark("powered by"),
                 tui._attribution_wordmark("OpenVINO"),
                 tui._startup_updates()):
        assert len(_styles(mark)) > 1, f"{mark.plain[:12]!r} is a flat colour"


def test_the_updates_panel_actually_lists_updates():
    """It used to be a heading over nothing at all."""
    from ovat.cli import tui

    panel = tui._startup_updates().plain
    assert tui.TUI_UPDATES, "no updates are defined"
    for line in tui.TUI_UPDATES:
        assert line in panel
    assert panel.count("\u203a") == len(tui.TUI_UPDATES), "one bullet per line"


def test_the_updates_panel_fits_its_column():
    """#updates-panel is 1fr between two fixed panels; at 120 columns that is
    about 47, and an over-long line would wrap into the bullet below it."""
    from ovat.cli import tui

    rows = _rows(tui._startup_updates())
    assert len(rows) <= 16
    assert max(len(r) for r in rows) <= 60


def test_the_attribution_marks_are_darker_than_the_wordmark():
    """They sit directly under OVAT, so they have to read as subordinate."""
    from ovat.cli import tui

    def brightness(hex_colour: str) -> int:
        return sum(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))

    attribution = [s for s in _styles(tui._attribution_wordmark("OVMS"))]
    assert attribution, "the mark carries no colour at all"
    darkest_wordmark = "#%02X%02X%02X" % tui._BLUE_RGB
    for style in attribution:
        shade = style.split()[-1]
        assert brightness(shade) < brightness(darkest_wordmark), (
            f"{shade} is not darker than the wordmark's darkest {darkest_wordmark}")


def test_the_credit_is_a_shaded_purple_mark():
    """It sits between two blue marks, so purple separates it from both.
    With every sub-mark in one face, colour is the only thing distinguishing
    the credit from the names it introduces."""
    from ovat.cli import tui

    credit = tui._credit_wordmark("powered by")
    assert len(_rows(credit)) > 1, "the credit should be a real FIGlet mark"
    for style in _styles(credit):
        shade = style.split()[-1]
        r, g, b = (int(shade[i:i + 2], 16) for i in (1, 3, 5))
        assert b > g and r > g, f"{shade} is not in the purple family"


def test_updates_is_green_throughout():
    """Green is the panel's identity: heading and bullets alike."""
    from ovat.cli import tui

    for style in _styles(tui._startup_updates()):
        shade = style.split()[-1]
        r, g, b = (int(shade[i:i + 2], 16) for i in (1, 3, 5))
        assert g > r and g > b, f"{shade} is not a green"


def test_the_banner_fits_inside_its_panel():
    """#brand-panel is a fixed 42 columns and the masthead is 17 rows. A mark
    wider than the panel does not clip, it WRAPS, which shears the glyphs."""
    from ovat.cli import tui

    banner = _rows(tui._banner())
    # 17 rows of masthead, less the round border top and bottom. This is an
    # EXACT fit, which is why it is asserted: one extra row and the stack is
    # clipped, and raising the masthead to make room hangs the app at 80x24.
    assert len(banner) <= 15, f"{len(banner)} rows will not fit in the panel"
    # #brand-panel is a fixed 42 columns, less its right border and padding.
    # A mark wider than that WRAPS rather than clipping, which shears glyphs.
    assert max(len(r) for r in banner) <= 39, "wider than the brand panel"


def test_a_missing_font_costs_a_size_not_the_masthead():
    """pyfiglet ships different font sets depending on packaging, so an
    unknown font name must degrade, never raise."""
    from ovat.cli import tui

    assert tui._figlet_lines("OVMS", font="no-such-font-exists") == []
    # And the caller still produces something renderable.
    assert tui._attribution_wordmark("OVMS").plain.strip()


# Selecting text and copying it out

def _drag(screen, x1, y1, x2, y2):
    """Drive a mouse drag through the SCREEN's own event routing.

    Not post_message: selection is handled in Screen._forward_event, which
    posting straight to the screen bypasses entirely. A probe that posts
    events sees no selection and wrongly concludes the feature is broken.
    """
    from textual.events import MouseDown, MouseMove, MouseUp

    def event(cls, x, y):
        return cls(None, x, y, 0, 0, 1, False, False, False,
                   screen_x=x, screen_y=y)

    screen._forward_event(event(MouseDown, x1, y1))
    screen._forward_event(event(MouseMove, x2, y2))
    screen._forward_event(event(MouseUp, x2, y2))


def test_dragging_over_the_log_yields_text_that_can_be_copied():
    """RichLog is a LINE-API widget, so Textual's generic get_selection()
    returns None for it and the clipboard came back empty. Highlighting
    worked, copying did not, which reads as "selection is broken"."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test(size=(100, 34)) as pilot:
            log = app.query_one("#output")
            log.write("SELECT THIS LINE OF TEXT PLEASE")
            await pilot.pause()
            region = log.region
            _drag(app.screen, region.x + 2, region.y + 1,
                  region.x + 28, region.y + 2)
            await pilot.pause()
            assert app.screen.get_selected_text(), "nothing was selected"
    _run(scenario())


def test_ctrl_c_copies_a_selection_instead_of_arming_a_quit():
    """The app's ctrl+c binding is priority=True and SHADOWS Textual's own
    "ctrl+c,super+c -> screen.copy_text" on every screen. Without an explicit
    branch, selecting text and pressing Ctrl-C armed a quit, which is the
    opposite of what every terminal does."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test(size=(100, 34)) as pilot:
            log = app.query_one("#output")
            log.write("COPY ME PLEASE")
            await pilot.pause()
            region = log.region
            _drag(app.screen, region.x + 2, region.y + 1,
                  region.x + 28, region.y + 2)
            await pilot.pause()

            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.clipboard, "ctrl+c did not copy the selection"
            assert app.is_running is True, "it quit instead of copying"
            # Cleared, so the NEXT ctrl+c means cancel-or-quit again and the
            # user is not stuck copying the same stale selection forever.
            assert not app.screen.get_selected_text()

            await pilot.press("ctrl+c")
            await pilot.pause()
            assert any("again to quit" in str(n.message)
                       for n in app._notifications)
    _run(scenario())


def test_ctrl_c_with_no_selection_still_arms_the_quit():
    """The copy branch must not swallow the plain case."""
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.is_running is True
            assert any("again to quit" in str(n.message)
                       for n in app._notifications)
    _run(scenario())


def test_no_widget_carries_a_hover_tooltip():
    """Tooltips were tried across the TUI and had to come out.

    Textual renders one as an unstyled block floating over whatever is
    behind it, anchored to the pointer, and wraps its text mid-phrase. On a
    full-screen app every tooltip therefore covers content the user is
    reading. The placeholders already say what matters.
    """
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            for widget in app.screen.query("*"):
                assert widget.tooltip is None, (
                    f"{widget!r} grew a tooltip again")
    _run(scenario())


# The telemetry screen: live numbers inside the TUI

def test_slash_telemetry_opens_the_page():
    from ovat.cli.telemetry_screen import TelemetryScreen

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "/telemetry"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, TelemetryScreen)
            app.screen.collector.stop()
    _run(scenario())


def test_the_page_shows_sources_that_cannot_run_here_with_the_reason():
    """On a Mac the Intel row must say so rather than draw a flat line at
    zero, which is indistinguishable from an idle NPU."""
    from textual.widgets import DataTable
    from ovat.cli.telemetry_screen import TelemetryScreen

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = TelemetryScreen()
            app.push_screen(screen)
            await pilot.pause()
            table = screen.query_one("#tel-table", DataTable)
            rows = "\n".join(
                " ".join(str(cell) for cell in table.get_row_at(i))
                for i in range(table.row_count))
            assert "process" in rows and "intel" in rows
            # Every source is accounted for, live or not.
            assert table.row_count == len(screen.collector.sources)
            screen.collector.stop()
    _run(scenario())


def test_sampling_does_not_run_on_the_ui_thread():
    """The collector owns its own daemon thread, so a stalled hardware
    collector cannot freeze the interface."""
    from ovat.cli.telemetry_screen import TelemetryScreen

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = TelemetryScreen()
            app.push_screen(screen)
            await pilot.pause()
            assert screen.collector._thread is not None
            assert screen.collector._thread.daemon
            screen.collector.stop()
    _run(scenario())


def test_leaving_the_page_stops_the_collector():
    """A hardware source launches a subprocess; it must not outlive the page
    that started it."""
    from ovat.cli.telemetry_screen import TelemetryScreen

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = TelemetryScreen()
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert screen.collector._thread is None
    _run(scenario())




# The telemetry page's tabs

def test_the_telemetry_page_has_four_switchable_tabs():
    """One page rather than four screens: these are four VIEWS of the same
    machine, and separate screens would drop the live buffer every time you
    looked at a different one."""
    from textual.widgets import TabbedContent
    from ovat.cli.telemetry_screen import TelemetryScreen

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = TelemetryScreen()
            app.push_screen(screen)
            await pilot.pause()
            tabs = screen.query_one("#tel-tabs", TabbedContent)
            assert tabs.tab_count == 4
            screen.collector.stop()
    _run(scenario())


def test_no_agent_row_remains_on_the_telemetry_page():
    """It could only ever read n/a: an agent lives in the chat screen's OVMS
    engine, only the native loop fills last_trace, and a permanently
    unavailable row teaches the reader to ignore the whole table."""
    from ovat.cli.telemetry_screen import TelemetryScreen

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = TelemetryScreen()
            app.push_screen(screen)
            await pilot.pause()
            names = [s.name for s in screen.collector.sources]
            assert "agent" not in names
            screen.collector.stop()
    _run(scenario())


def test_the_plano_tab_carries_the_commands_to_run_it():
    """A tab that only says a thing exists is worse than no tab. This one has
    to be runnable straight off the screen."""
    from textual.widgets import Static
    from ovat.cli.telemetry_screen import TelemetryScreen

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = TelemetryScreen()
            app.push_screen(screen)
            await pilot.pause()
            widget = screen.query_one("#tel-plano-help", Static)
            text = str(getattr(widget, "renderable", None)
                       or getattr(widget, "_content", "")
                       or widget.render())
            assert "planoai up" in text
            assert "base_url_path_prefix" in text     # the spike answer
            screen.collector.stop()
    _run(scenario())


def test_the_tui_boots_lists_its_commands_and_exits():
    """The one end-to-end smoke test the TUI never had.

    Every other TUI test drives one widget or one screen. Nothing asserted
    that the app BOOTS, shows the commands the README documents, and shuts
    down again -- the launcher was verified only by a human watching it, and
    an AI PC verification sweep could not run it at all because it takes over
    the terminal's alternate screen.

    Textual's headless Pilot is exactly the missing piece: no real terminal,
    so this runs anywhere, including in CI.
    """
    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            # The commands the README promises are really on offer. Read
            # from the shell's own template registry, which is what the slash
            # menu renders, so this cannot drift from what the user sees.
            from ovat.cli import shell
            offered = set(shell.TEMPLATES_BY_NAME)
            for documented in ("/chat", "/doctor", "/init", "/index", "/run"):
                assert documented in offered, f"{documented} is not offered"
            # And it comes down cleanly rather than hanging on exit.
            app.exit()
        assert app.return_code in (0, None)
    _run(scenario())


def test_the_app_boots_in_a_default_80x24_terminal():
    """The masthead height is load-bearing, and nothing said so out loud.

    #masthead is 17 rows because an 18-row one leaves the output pane no
    space at 80x24 and the app HANGS on startup instead of laying out. 24
    rows is a default terminal, so that is not an exotic case.

    Every test in this file already runs at 80x24 -- Textual's run_test
    defaults to size=(80, 24) -- so the constraint has in fact been covered
    all along, by accident. This states it: if someone passes a larger size
    here, or Textual changes that default, the guard silently disappears and
    the next person to try 18 rows finds out from a hang rather than a test.

    What it checks is the VALUE plus a successful layout, not the hang
    itself: headless rendering does not reproduce the hang (set the height
    to 18 and this fails on the assertion, not by hanging). Pinning the
    measured-good number is the guard; the boot proves the pane still fits.
    """
    async def scenario():
        app = OvatTUI()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert app.size.width == 80 and app.size.height == 24
            masthead = app.query_one("#masthead", Horizontal)
            assert masthead.styles.height.value == 17, (
                "raising this to 18 hangs the app at 80x24")
            # It laid out and the prompt is reachable, which is the thing a
            # hang would prevent.
            assert app.query_one("#prompt", Input).region.height > 0
    _run(scenario())


def test_pixels_works_on_pillow_without_get_flattened_data():
    """get_flattened_data() arrived in Pillow 12.1.0; pillow has no floor here.

    Calling the new name unconditionally raises AttributeError on any older
    resolve, and the loader only guards (OSError, UnidentifiedImageError) --
    so it would escape as a crash at TUI launch rather than a missing
    animation. Back the fallback out and this fails.
    """
    from ovat.cli.tui import _pixels

    class OldPillowImage:
        """Pillow < 12.1: getdata() only."""
        def getdata(self):
            return [(1, 2, 3), (4, 5, 6)]

    assert list(_pixels(OldPillowImage())) == [(1, 2, 3), (4, 5, 6)]


def test_pixels_prefers_the_new_name_when_it_exists():
    """getdata() is deprecated in favour of it, so new Pillow must not warn."""
    from ovat.cli.tui import _pixels

    class NewPillowImage:
        def get_flattened_data(self):
            return [("new",)]

        def getdata(self):
            raise AssertionError("used the deprecated name")

    assert list(_pixels(NewPillowImage())) == [("new",)]


def test_a_real_pillow_image_still_reads():
    """The shim must not break the path that actually runs today."""
    from PIL import Image

    from ovat.cli.tui import _pixels

    image = Image.new("RGB", (2, 1), (10, 20, 30))
    assert list(_pixels(image)) == [(10, 20, 30), (10, 20, 30)]
