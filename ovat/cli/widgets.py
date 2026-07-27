# ovat/cli/widgets.py
"""Small widget subclasses shared by the TUI screens."""
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Input, RichLog, TextArea

from ovat.cli.editing import read_clipboard


class PasteInput(Input):
    """An Input whose Ctrl-V reaches the SYSTEM clipboard.

    Textual's own Input.action_paste reads app.clipboard, which is Textual's
    internal buffer and only ever holds what the app itself copied with
    copy_to_clipboard. Text copied from a browser, an editor, or the
    terminal's own selection was therefore impossible to paste in: the key
    was bound, the action ran, and nothing happened, which is worse than not
    being bound at all.

    The system clipboard wins when it has anything; app.clipboard is the
    fallback, so pasting straight back what /copy just put there still works
    even on a machine with no pbpaste/xclip.
    """

    def action_paste(self) -> None:
        text = read_clipboard() or self.app.clipboard
        if not text:
            return
        start, end = self.selection
        self.replace(text, start, end)


class ChatInput(TextArea):
    """A multi-line prompt box. Enter sends, Shift-Enter starts a new line.

    The single-line Input could not compose or edit a long prompt, which is
    the one thing you most want to do before spending thirty seconds of CPU
    generating an answer to it.

    Two compatibility shims keep the rest of the screen unchanged: `value`
    maps to TextArea's `text`, and `cursor_position` maps to a character
    offset. Everything already written against an Input (history recall, the
    slash menu, tests) therefore keeps working, which is the difference
    between a widget swap and a rewrite.
    """

    BINDINGS = [
        # priority=True: TextArea consumes Enter as a newline otherwise, and
        # a chat box that cannot send on Enter is not a chat box.
        Binding("enter", "submit", "Send", priority=True, show=False),
        Binding("shift+enter", "newline", "New line", show=False),
        Binding("ctrl+v", "paste_clipboard", "Paste", priority=True,
                show=False),
    ]

    class Submitted(Message):
        """Posted when Enter is pressed, mirroring Input.Submitted."""

        def __init__(self, chat_input: "ChatInput", value: str) -> None:
            super().__init__()
            self.chat_input = chat_input
            self.value = value

        @property
        def control(self) -> "ChatInput":
            return self.chat_input

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, new_value: str) -> None:
        self.load_text(new_value)

    @property
    def cursor_position(self) -> int:
        row, column = self.cursor_location
        lines = self.text.split("\n")
        return sum(len(line) + 1 for line in lines[:row]) + column

    @cursor_position.setter
    def cursor_position(self, offset: int) -> None:
        row, remaining = 0, offset
        for line in self.text.split("\n"):
            if remaining <= len(line):
                break
            remaining -= len(line) + 1
            row += 1
        self.cursor_location = (row, remaining)

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self, self.text))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_paste_clipboard(self) -> None:
        """Same system-clipboard read as PasteInput; see its docstring."""
        text = read_clipboard() or self.app.clipboard
        if text:
            self.insert(text)


class SelectableRichLog(RichLog):
    """A RichLog whose selected text can actually be copied out.

    Textual's generic Widget.get_selection() renders the widget and extracts
    from the result, but only when that result is a Text or Content. RichLog
    is a LINE-API widget: it renders to Strips, so the generic path returns
    None and there is nothing to copy. The effect was that dragging across
    the log highlighted it, ctrl+c/cmd+c fired, and the clipboard stayed
    empty, which reads as "selection is broken" rather than "this widget
    cannot extract".

    RichLog already keeps every rendered line in self.lines, and a Strip
    knows its own text, so joining them gives exactly the document the
    selection offsets are measured against.
    """

    def get_selection(self, selection):
        text = "\n".join(strip.text for strip in self.lines)
        return selection.extract(text), "\n"
