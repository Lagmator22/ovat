# ovat/tui/widgets/input.py
# Look adapted from Mistral Vibe (https://github.com/mistralai/mistral-vibe,
# vibe/cli/textual_ui/widgets/chat_input/ and app.tcss #input-box,
# #completion-popup), Copyright Mistral AI, Apache License 2.0. Changed for
# OVAT: built on OptionList for mouse support, OVAT colours, OVAT commands.
"""The prompt: multi-line input, history, `!` bash mode, `/` completion.

Keys:
  Enter            send (or accept the highlighted completion)
  Ctrl-J           new line (Shift-Enter too, where the terminal reports it)
  Up / Down        history, but only from the first / last line, so moving
                   inside a multi-line draft still works
  Tab              accept the highlighted completion
  Esc              close the completion list (the app handles the rest)

The prompt glyph turns into a purple "!" when the text starts with "!", so
it is obvious before Enter that this goes to the shell, not the model.
History is saved to ~/.ovat/agent_history.jsonl (one JSON string per line,
so multi-line entries survive), separate from the launcher's.
"""
import json
from pathlib import Path

from rich.text import Text
from textual import events
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

from ovat.agentic import commands
from ovat.cli import ui
from ovat.cli.editing import InputHistory

HISTORY_FILE = Path.home() / ".ovat" / "agent_history.jsonl"
_HISTORY_KEEP = 500


def load_history(path: Path = HISTORY_FILE) -> InputHistory:
    history = InputHistory(limit=_HISTORY_KEEP)
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                history.add(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return history


def save_history_entry(text: str, path: Path = HISTORY_FILE) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(text) + "\n")
    except OSError:
        pass                     # history is a convenience, never an error


class CompletionPopup(OptionList, can_focus=False):
    """Every slash command matching what is typed, in a box (Mistral Vibe's
    shape): the mouse hovers and clicks rows, the wheel scrolls, and a
    scrollbar shows there is more. It never takes the focus, so typing
    carries on in the input while it is open."""

    DEFAULT_CSS = """
    CompletionPopup { height: auto; max-height: 12; width: 100%;
                      display: none; border: solid $foreground 35%;
                      background: $background; padding: 0 1;
                      scrollbar-size-vertical: 1;
                      scrollbar-color: $foreground 40%; }
    CompletionPopup:focus { border: solid $foreground 35%; }
    CompletionPopup.shown { display: block; }
    CompletionPopup > .option-list--option-highlighted {
        background: transparent; }
    CompletionPopup:focus > .option-list--option-highlighted {
        background: transparent; }
    CompletionPopup > .option-list--option-hover {
        background: $foreground 8%; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.items: list[commands.Command] = []

    @staticmethod
    def _row(command: commands.Command, width: int,
             selected: bool = False) -> Text:
        """Vibe's row: the command bold (reversed when selected), then its
        description, muted (italic when selected)."""
        label = f"/{command.name}" + (f" {command.usage}" if command.usage
                                      else "")
        row = Text(no_wrap=True, overflow="ellipsis")
        row.append(label, style="bold reverse" if selected else "bold")
        row.append(" " * max(2, width - len(label)))
        row.append(command.help, style="italic" if selected else ui.DIM)
        return row

    def _width(self) -> int:
        return max(len(c.name) + len(c.usage) for c in commands.COMMANDS) + 4

    def _restyle(self, old: int | None, new: int | None) -> None:
        for index, selected in ((old, False), (new, True)):
            if index is not None and 0 <= index < len(self.items):
                self.replace_option_prompt_at_index(
                    index, self._row(self.items[index], self._width(),
                                     selected))

    def show_for(self, text: str) -> None:
        wanted = (text.startswith("/") and " " not in text
                  and "\n" not in text)
        self.items = commands.complete(text, limit=None) if wanted else []
        self.clear_options()
        if self.items:
            width = self._width()
            self.add_options([Option(self._row(c, width, i == 0), id=c.name)
                              for i, c in enumerate(self.items)])
            self.highlighted = 0
        self.set_class(bool(self.items), "shown")

    def move(self, delta: int) -> None:
        if self.items:
            old = self.highlighted
            self.highlighted = ((old or 0) + delta) % len(self.items)
            self._restyle(old, self.highlighted)

    @property
    def shown(self) -> bool:
        return bool(self.items)

    def selected(self) -> commands.Command | None:
        if not self.items or self.highlighted is None:
            return None
        return self.items[self.highlighted]

    def hide(self) -> None:
        self.items = []
        self.clear_options()
        self.remove_class("shown")


class InputArea(TextArea):
    """TextArea with chat keys. Posts InputBox.Submitted via its box."""

    def __init__(self, box: "InputBox", **kwargs) -> None:
        super().__init__(soft_wrap=True, show_line_numbers=False,
                         highlight_cursor_line=False, compact=True, **kwargs)
        self._box = box

    async def _on_key(self, event: events.Key) -> None:
        popup = self._box.popup
        key = event.key
        if key == "backspace" and not self.text and self._box.bash:
            event.prevent_default()
            event.stop()
            self._box.set_bash(False)          # back to a normal prompt
            return
        if key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if key == "enter":
            event.prevent_default()
            event.stop()
            if popup.shown:
                self._box.accept_completion(run=True)
            else:
                self._box.submit()
            return
        if key == "tab" and popup.shown:
            event.prevent_default()
            event.stop()
            self._box.accept_completion(run=False)
            return
        if key == "escape" and popup.shown:
            event.prevent_default()
            event.stop()
            popup.hide()
            return
        if key == "up":
            if popup.shown:
                event.prevent_default(); event.stop(); popup.move(-1)
                return
            if self.cursor_at_first_line:
                event.prevent_default(); event.stop(); self._box.history(-1)
                return
        if key == "down":
            if popup.shown:
                event.prevent_default(); event.stop(); popup.move(1)
                return
            if self.cursor_at_last_line:
                event.prevent_default(); event.stop(); self._box.history(1)
                return
        await super()._on_key(event)


class InputBox(Vertical):
    """Popup + prompt glyph + text, between two thin rules."""

    DEFAULT_CSS = """
    InputBox { height: auto; width: 100%; }
    InputBox #input-row { height: auto; border: none;
                          border-top: solid $foreground 35%;
                          border-bottom: solid $foreground 35%;
                          border-title-align: right;
                          border-title-color: $text-muted; }
    InputBox #input-row.bash { border-top: solid $secondary;
                               border-bottom: solid $secondary;
                               border-title-color: $secondary; }
    InputBox #input-row.border-warning { border-top: solid $warning;
                                         border-bottom: solid $warning;
                                         border-title-color: $warning; }
    InputBox #glyph { width: auto; padding: 0 1 0 0; color: $accent;
                      text-style: bold; }
    InputBox #input-row.bash #glyph { color: $secondary; }
    InputBox InputArea { width: 1fr; height: auto; min-height: 3;
                         max-height: 50vh; background: transparent;
                         border: none; padding: 0;
                         scrollbar-visibility: hidden; }
    InputBox InputArea:focus { border: none; }
    """

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, history: InputHistory | None = None,
                 history_file: Path | None = HISTORY_FILE, **kwargs) -> None:
        super().__init__(**kwargs)
        self._history_file = history_file
        self._history = history if history is not None else (
            load_history(history_file) if history_file else InputHistory())
        self.popup = CompletionPopup()
        # Bash mode: the typed "!" is taken out of the text and shown as
        # the purple prompt glyph instead, so it appears once.
        self.bash = False
        self.area = InputArea(self, placeholder="Ask anything, /command, "
                                                "or !shell command")

    def compose(self):
        yield self.popup
        with Horizontal(id="input-row"):
            yield Static(">", id="glyph")
            yield self.area

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = self.area.text
        if text.startswith("!") and not self.bash:
            self.set_bash(True)
            self.area.text = text[1:]          # fires Changed again
            self.area.move_cursor(self.area.document.end)
            return
        self.popup.show_for("" if self.bash else text)

    def on_option_list_option_selected(
            self, event: OptionList.OptionSelected) -> None:
        """A click on a completion row: same as Enter on it."""
        event.stop()
        self.popup.highlighted = event.option_index
        self.accept_completion(run=True)
        self.area.focus()

    def set_bash(self, on: bool) -> None:
        self.bash = on
        row = self.query_one("#input-row")
        row.set_class(on, "bash")
        self.query_one("#glyph", Static).update("!" if on else ">")
        row.border_title = ("bash mode  ·  backspace leaves" if on
                            else getattr(self, "_title", ""))

    def set_text(self, text: str) -> None:
        if self.bash and not text.startswith("!"):
            self.set_bash(False)
        self.area.text = text
        self.area.move_cursor(self.area.document.end)

    def submit(self) -> None:
        text = ("!" + self.area.text) if self.bash else self.area.text
        if not text.strip() or text.strip() == "!":
            return
        if self.bash:
            self.set_bash(False)
        self._history.add(text)
        if self._history_file:
            save_history_entry(text.strip(), self._history_file)
        self.area.text = ""
        self.popup.hide()
        self.post_message(self.Submitted(text.strip()))

    def accept_completion(self, run: bool) -> None:
        command = self.popup.selected()
        if command is None:
            return
        self.popup.hide()
        if run and not command.usage:
            self.area.text = f"/{command.name}"
            self.submit()
            return
        self.set_text(f"/{command.name} ")

    def history(self, direction: int) -> None:
        value = (self._history.previous(self.area.text) if direction < 0
                 else self._history.next())
        if value is not None:
            self.set_text(value)

    def set_warning(self, on: bool) -> None:
        """Vibe's warning border: every tool is approved (yolo)."""
        self.query_one("#input-row").set_class(on, "border-warning")

    def set_title(self, title: str) -> None:
        title = f" {title} " if title else ""
        self._title = title
        row = self.query_one("#input-row")
        if not row.has_class("bash"):
            row.border_title = title
