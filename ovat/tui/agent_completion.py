# ovat/tui/agent_completion.py
"""OVAT's additions to the copied `/` and `@` completion popup.

The copied popup only listens to the keyboard. Here the mouse drives it too,
by sending the same keys the person could have pressed: the wheel is Up and
Down, a click is Down/Up to the row and then Tab. That way each completion
controller (slash, path) keeps its own rules and nothing is reimplemented.
"""
from __future__ import annotations

from textual import events

from ovat.tui.vibe.cli.autocompletion.base import CompletionResult


def press(container, key: str) -> CompletionResult:
    """Send `key` to whichever completion is on screen, as the text area
    would. IGNORED when the container or its input is not ready."""
    widget = container.input_widget
    if widget is None:
        return CompletionResult.IGNORED
    return container._completion_manager.on_key(
        events.Key(key, None), widget.get_full_text(),
        widget._get_full_cursor_offset())


def scroll(container, delta: int) -> None:
    """Wheel: move the highlight one row."""
    press(container, "down" if delta > 0 else "up")


def pick(container, delta: int) -> None:
    """Click: move the highlight `delta` rows, then choose it."""
    for _ in range(abs(delta)):
        press(container, "down" if delta > 0 else "up")
    press(container, "tab")
    container.focus_input()
