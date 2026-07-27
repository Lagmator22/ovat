# tests/test_editing.py
"""Tests for the input-line conveniences: history recall and clipboard paste.

Both are plain logic with no textual import, so they are tested directly
rather than through a headless screen.
"""
import sys

from ovat.cli.editing import InputHistory, read_clipboard


# History: Up and Down, the way a shell does it

def test_up_walks_back_through_what_you_submitted():
    history = InputHistory()
    for line in ("first", "second", "third"):
        history.add(line)
    assert history.previous("") == "third"
    assert history.previous("") == "second"
    assert history.previous("") == "first"


def test_the_oldest_entry_is_a_floor_not_a_wrap():
    """Walking off the top should stay put, not jump to the newest."""
    history = InputHistory()
    history.add("only")
    assert history.previous("") == "only"
    assert history.previous("") == "only"


def test_down_returns_the_half_typed_line_you_left_behind():
    """The part that is infuriating to get wrong.

    Type half a question, press Up to check what you asked before, press
    Down again: a shell gives you your unfinished line back.
    """
    history = InputHistory()
    history.add("what is openvino?")
    assert history.previous("half typed qu") == "what is openvino?"
    assert history.next() == "half typed qu"
    assert history.next() is None          # past the end, nothing more


def test_down_does_nothing_when_you_never_went_up():
    assert InputHistory().next() is None
    history = InputHistory()
    history.add("a")
    assert history.next() is None


def test_submitting_again_stops_browsing():
    history = InputHistory()
    history.add("a")
    history.add("b")
    history.previous("")                   # start browsing
    history.add("c")                       # a new submission resets it
    assert history.next() is None
    assert history.previous("") == "c"


def test_an_immediate_repeat_is_not_stored_twice():
    """Otherwise asking the same thing twice needs two presses to get past."""
    history = InputHistory()
    history.add("same")
    history.add("same")
    assert history.entries == ["same"]


def test_blank_lines_are_not_recorded():
    history = InputHistory()
    history.add("   ")
    history.add("")
    assert history.entries == []
    assert history.previous("") is None


def test_history_is_capped_and_keeps_the_newest():
    history = InputHistory(limit=3)
    for line in ("a", "b", "c", "d"):
        history.add(line)
    assert history.entries == ["b", "c", "d"]


# Clipboard: reading it is the half terminals do not give us

def test_read_clipboard_returns_the_text_without_a_trailing_newline():
    class Result:
        returncode = 0
        stdout = "pasted text\r\n"
    assert read_clipboard(_run=lambda *a, **k: Result()) == "pasted text"


def test_read_clipboard_is_empty_rather_than_raising():
    """A missing xclip or a locked clipboard means the paste does nothing.

    It must never take the TUI down, which is the whole reason this is not
    just a subprocess call at the call site.
    """
    def explode(*a, **k):
        raise FileNotFoundError("xclip not installed")
    assert read_clipboard(_run=explode) == ""

    class Failed:
        returncode = 1
        stdout = ""
    assert read_clipboard(_run=lambda *a, **k: Failed()) == ""


def test_the_paste_command_suits_the_platform(monkeypatch):
    from ovat.cli import editing
    for platform, expected in (("win32", "Get-Clipboard"),
                               ("darwin", "pbpaste"),
                               ("linux", "xclip")):
        monkeypatch.setattr(editing.sys, "platform", platform)
        assert any(expected in part for part in editing._paste_command())


# The paste widget: Textual's own Input cannot reach the system clipboard

def test_paste_input_prefers_the_system_clipboard(monkeypatch):
    """Input.action_paste reads app.clipboard, Textual's INTERNAL buffer,
    which only holds what the app itself copied. Text copied from a browser
    or the terminal's own selection could not be pasted in at all: the key
    was bound, the action ran, and nothing happened."""
    import pytest
    pytest.importorskip("textual")
    from ovat.cli import widgets

    monkeypatch.setattr(widgets, "read_clipboard", lambda: "FROM THE OS")
    pasted = []

    class Probe(widgets.PasteInput):
        selection = (0, 0)

        def replace(self, text, start, end):
            pasted.append(text)

        @property
        def app(self):
            class A:
                clipboard = "internal"
            return A()

    Probe().action_paste()
    assert pasted == ["FROM THE OS"]


def test_paste_input_falls_back_to_textuals_own_clipboard(monkeypatch):
    """So pasting back what /copy just put there works even with no pbpaste."""
    import pytest
    pytest.importorskip("textual")
    from ovat.cli import widgets

    monkeypatch.setattr(widgets, "read_clipboard", lambda: "")
    pasted = []

    class Probe(widgets.PasteInput):
        selection = (0, 0)

        def replace(self, text, start, end):
            pasted.append(text)

        @property
        def app(self):
            class A:
                clipboard = "internal"
            return A()

    Probe().action_paste()
    assert pasted == ["internal"]
