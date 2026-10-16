from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from textual import events
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from ovat.tui.vibe.cli.autocompletion.base import CompletionEntry

COMPLETION_POPUP_MAX_HEIGHT = 12
COMPLETION_POPUP_PADDING_X = 1
SELECTED_CLASS = "completion-selected"
NO_DESCRIPTION_CLASS = "completion-no-description"


class _CompletionItem(Static):
    pass


class _CompletionRow(Horizontal):
    pass


class CompletionPopup(VerticalScroll):
    # OVAT: the wheel and a click used to do nothing here. They only say what
    # was asked; ovat/tui/agent_completion.py moves the highlight or picks.
    class Scrolled(Message):
        def __init__(self, delta: int) -> None:
            self.delta = delta
            super().__init__()

    class Picked(Message):
        def __init__(self, delta: int) -> None:
            # steps from the highlighted row to the clicked one
            self.delta = delta
            super().__init__()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="completion-popup", **kwargs)
        self.styles.display = "none"
        self.styles.max_height = COMPLETION_POPUP_MAX_HEIGHT
        self.styles.padding = (0, COMPLETION_POPUP_PADDING_X)
        self.can_focus = False
        self._suggestions: list[CompletionEntry] = []
        self._selected = 0  # OVAT

    def update_suggestions(
        self, suggestions: list[CompletionEntry], selected: int
    ) -> None:
        if not suggestions:
            self.hide()
            return

        if suggestions != self._suggestions:
            rows = self._rebuild(suggestions)
        else:
            rows = list(self.query(_CompletionRow))
        self._select(rows, selected)
        self.styles.display = "block"

    def _rebuild(self, suggestions: list[CompletionEntry]) -> list[_CompletionRow]:
        self.remove_children()
        self._suggestions = suggestions
        has_descriptions = any(entry.description for entry in suggestions)
        self.set_class(not has_descriptions, NO_DESCRIPTION_CLASS)
        command_width = max(
            cell_len(self._display_label(entry.label)) for entry in suggestions
        )
        rows: list[_CompletionRow] = []
        for entry in suggestions:
            command = _CompletionItem(
                self._display_label(entry.label), classes="completion-command"
            )
            if has_descriptions:
                command.styles.width = command_width
            description_cell = _CompletionItem(
                entry.description, classes="completion-description"
            )
            rows.append(_CompletionRow(command, description_cell))
        self.mount_all(rows)
        return rows

    def _select(self, rows: list[_CompletionRow], selected: int) -> None:
        self._selected = selected  # OVAT
        for idx, row in enumerate(rows):
            row.set_class(idx == selected, SELECTED_CLASS)
        if 0 <= selected < len(rows):
            rows[selected].scroll_visible(animate=False)

    # OVAT: wheel moves the highlight, a click chooses that row.
    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.stop()
        event.prevent_default()
        self.post_message(self.Scrolled(1))

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.stop()
        event.prevent_default()
        self.post_message(self.Scrolled(-1))

    def on_click(self, event: events.Click) -> None:
        widget = event.widget
        while widget is not None and not isinstance(widget, _CompletionRow):
            widget = widget.parent
        rows = list(self.query(_CompletionRow))
        if widget is None or widget not in rows:
            return
        event.stop()
        self.post_message(self.Picked(rows.index(widget) - self._selected))

    def hide(self) -> None:
        self.remove_children()
        self._suggestions = []
        self.styles.display = "none"

    @property
    def content_text(self) -> str:
        return "\n".join(str(child.render()) for child in self.query(_CompletionItem))

    @staticmethod
    def _display_label(label: str) -> str:
        if label.startswith("@"):
            return label[1:]
        return label
