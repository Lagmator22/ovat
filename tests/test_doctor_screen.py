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
    """Push the doctor screen and wait until its checks worker has finished.

    The pause BEFORE wait_for_complete is what makes this reliable. The worker
    is started from on_mount, so on a slow runner it may not exist yet at the
    moment we ask to wait for it -- wait_for_complete then returns instantly
    with nothing to wait for, and the test reads an empty log. That failed on
    the Windows runner while passing on every developer machine, which makes
    it the same environment-sensitive class as everything else this release
    fixed: a test measuring the scheduler rather than the code.

    The loop afterwards is belt and braces: workers can enqueue further work,
    so we re-check a bounded number of times rather than trusting one pass.
    """
    screen = DoctorScreen(config_path=config_path)
    app.push_screen(screen)
    await pilot.pause()                        # let on_mount start the worker
    for _ in range(50):                        # bounded: never hangs the suite
        await app.workers.wait_for_complete()
        await pilot.pause()
        if not any(worker.is_running for worker in app.workers):
            break
    return screen


def _log_text(screen) -> str:
    from ovat.cli.widgets import SelectableRichLog
    log = screen.query_one("#doc-log", SelectableRichLog)
    return "\n".join(str(line) for line in log.lines)


def _table(screen):
    from textual.widgets import DataTable
    return screen.query_one("#doc-table", DataTable)


def _table_text(screen) -> str:
    """Every cell in the results table, flattened.

    The checks used to be rendered into the log; they are a DataTable now, so
    assertions about RESULTS look here and assertions about MESSAGES (copied,
    unknown command, errors) still look at the log.
    """
    table = _table(screen)
    rows = [table.get_row_at(i) for i in range(table.row_count)]
    return "\n".join("  ".join(str(cell) for cell in row) for row in rows)


def _statuses(screen) -> list:
    """The status column, top to bottom, as bare words."""
    table = _table(screen)
    return [str(table.get_row_at(i)[1]).split()[-1]
            for i in range(table.row_count)]


def test_a_bracket_in_a_check_detail_does_not_break_the_render(monkeypatch):
    """Table cells must be Text, not str.

    A DataTable renders a str cell as markup, so "model=Q[/b]8B" would raise
    MarkupError mid-render. This is the one place ui.esc() cannot reach,
    because the value never passes through the CLI console.
    """
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            text = _table_text(screen)
            assert "Q[/b]8B" in text            # rendered literally
            assert "Python" in text
            # The counts moved to the table's border, where they describe the
            # thing they are counting.
            assert "1 fail" in str(_table(screen).border_title)
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
            assert palette.option_count == 5          # every command

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


# What the table adds over a rendered block of text

def test_the_worst_checks_are_listed_first(monkeypatch):
    """The point of running doctor is finding what is broken. A fail sitting
    below nine oks is a fail you scroll past."""
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            statuses = _statuses(screen)
            severity = {"fail": 0, "warn": 1, "ok": 2}
            ranks = [severity[s] for s in statuses]
            assert ranks == sorted(ranks), f"not worst-first: {statuses}"
    _run(scenario())


def test_sorting_alphabetically_would_put_ok_between_fail_and_warn(monkeypatch):
    """Why the rows are rebuilt rather than run through DataTable.sort().

    The status cell renders as a glyph plus a word, so sorting on the cell
    would be alphabetical: fail, ok, warn. That puts the state you do not
    care about between the two you do.
    """
    assert sorted(["fail", "warn", "ok"]) == ["fail", "ok", "warn"]


def test_f6_toggles_back_to_the_order_the_checks_ran_in(monkeypatch):
    """Severity order is the useful default, but the original order is how
    doctor explains itself: environment first, then workflow."""
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            sorted_names = [str(_table(screen).get_row_at(i)[0])
                            for i in range(_table(screen).row_count)]
            await pilot.press("f6")
            await pilot.pause()
            original = [str(_table(screen).get_row_at(i)[0])
                        for i in range(_table(screen).row_count)]
            assert original == [c.name for c in FAKE_CHECKS]
            assert original != sorted_names, (
                "the fixture is already in severity order, so this proves "
                "nothing; give it a check that is out of order")
    _run(scenario())


def test_selecting_a_row_copies_just_that_check(monkeypatch):
    """When one check fails, that ONE line is what gets pasted into an issue.
    Copying all fifteen and asking the reader to find it is worse."""
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            table = _table(screen)
            table.focus()
            table.move_cursor(row=0)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert app.clipboard, "selecting a row copied nothing"
            first_row = "  ".join(str(c) for c in table.get_row_at(0))
            assert app.clipboard == first_row
            # One check, not the whole report.
            assert len(app.clipboard.splitlines()) == 1
            assert "copied:" in _log_text(screen)
    _run(scenario())


def test_slash_copy_still_takes_the_whole_report(monkeypatch):
    """Per-row copy is an addition, not a replacement."""
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            screen.query_one("#doc-input", Input).value = "/copy"
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.clipboard.splitlines()) > len(FAKE_CHECKS) - 1
            for check in FAKE_CHECKS:
                assert check.name in app.clipboard
    _run(scenario())


def test_the_message_strip_stays_hidden_until_there_is_a_message(monkeypatch):
    """A clean run should be the table and nothing else, not the table plus
    an empty box taking five rows off a short terminal."""
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            from ovat.cli.widgets import SelectableRichLog
            log = screen.query_one("#doc-log", SelectableRichLog)
            assert log.display is False

            screen.query_one("#doc-input", Input).value = "/nonsense"
            await pilot.press("enter")
            await pilot.pause()
            assert log.display is True
    _run(scenario())


# check_action: the footer must not advertise keys that cannot fire

def test_f6_is_greyed_until_there_are_checks_to_sort(monkeypatch):
    """Greyed, not hidden. A key that vanishes and reappears makes the footer
    twitch and leaves the user unsure the key exists at all."""
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            screen._checks = []
            assert screen.check_action("toggle_sort", ()) is None
            screen._checks = list(FAKE_CHECKS)
            assert screen.check_action("toggle_sort", ()) is True
    _run(scenario())


def test_f5_is_greyed_while_the_checks_are_already_running(monkeypatch):
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            assert screen.check_action("refresh_checks", ()) is True
            screen._busy = True
            assert screen.check_action("refresh_checks", ()) is None
    _run(scenario())


def test_escape_is_never_gated(monkeypatch):
    """Whatever else is true, the user must be able to leave."""
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            screen._busy = True
            screen._checks = []
            assert screen.check_action("back", ()) is True
    _run(scenario())


def test_the_footer_is_actually_refreshed_when_busy_flips(monkeypatch):
    """check_action is only consulted when the bindings are refreshed. Without
    the explicit call the footer keeps showing F5 as live for the whole run,
    which is the bug this feature exists to prevent."""
    monkeypatch.setattr(diagnostics, "run_checks", lambda cfg=None: FAKE_CHECKS)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _open_doctor(app, pilot)
            calls = []
            screen.refresh_bindings = lambda: calls.append(1)
            screen.action_refresh_checks()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert calls, "refresh_bindings was never called"
    _run(scenario())
