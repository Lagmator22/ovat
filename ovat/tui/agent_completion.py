# ovat/tui/agent_completion.py
"""OVAT's additions to the copied `/` and `@` completion popup.

`@` also offers the workflow's subagents, ahead of the files: `@name task`
is how a person sends a task to one (ovat/tui/server/turns.py).

The copied popup only listens to the keyboard. Here the mouse drives it too,
by sending the same keys the person could have pressed: the wheel is Up and
Down, a click is Down/Up to the row and then Tab. That way each completion
controller (slash, path) keeps its own rules and nothing is reimplemented.
"""
from __future__ import annotations

from textual import events

from ovat.tui.vibe.cli.autocompletion.base import (CompletionEntry,
                                                  CompletionResult)
from ovat.tui.vibe.cli.autocompletion.path_completion import (
    PathCompletionController)


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


class SubagentCompletionController(PathCompletionController):
    """The copied `@` completion, with the subagent names first. `agents` is
    a function returning (name, description) pairs; it is asked every time,
    because /workflow can change the list."""

    def __init__(self, base: PathCompletionController, agents) -> None:
        # Same completer and view as the controller it replaces.
        super().__init__(base._completer, base._view)
        self._agents = agents

    def _compute_completions(self, text: str, cursor_index: int,
                             generation: int) -> list[CompletionEntry]:
        files = super()._compute_completions(text, cursor_index, generation)
        fragment = text[:cursor_index].rpartition("@")[2].lower()
        named = [(name, note) for name, note in self._agents()
                 if fragment in name.lower()]
        # names that start with what was typed come before ones that only
        # contain it
        named.sort(key=lambda pair: not pair[0].lower().startswith(fragment))
        return [CompletionEntry(f"@{name}", note) for name, note in named
                ] + files


def offer_subagents(container, agents) -> None:
    """Swap the container's file completion for the one above."""
    controllers = container._completion_manager._controllers
    for i, controller in enumerate(controllers):
        if type(controller) is PathCompletionController:
            controllers[i] = SubagentCompletionController(controller, agents)
