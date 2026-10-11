# ovat/tui/widgets/input.py
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
from textual.widgets import Static, TextArea

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


class CompletionPopup(Static):
    """Slash commands matching what is typed, with what each does."""

    DEFAULT_CSS = """
    CompletionPopup { height: auto; width: 100%; display: none;
                      padding: 0 1; }
    CompletionPopup.shown { display: block; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.items: list[commands.Command] = []
        self.index = 0

    def show_for(self, text: str) -> None:
        wanted = (text.startswith("/") and " " not in text
                  and "\n" not in text)
        self.items = commands.complete(text) if wanted else []
        self.index = 0
        self.set_class(bool(self.items), "shown")
        self._draw()

    def move(self, delta: int) -> None:
        if self.items:
            self.index = (self.index + delta) % len(self.items)
            self._draw()

    @property
    def shown(self) -> bool:
        return bool(self.items)

    def selected(self) -> commands.Command | None:
        return self.items[self.index] if self.items else None

    def hide(self) -> None:
        self.items = []
        self.remove_class("shown")

    def _draw(self) -> None:
        out = Text()
        width = max((len(c.name) + len(c.usage) for c in self.items),
                    default=0) + 3
        for i, c in enumerate(self.items):
            if i:
                out.append("\n")
            label = f"/{c.name}" + (f" {c.usage}" if c.usage else "")
            selected = i == self.index
            out.append(label.ljust(width),
                       style=f"bold reverse {ui.PURPLE}" if selected
                       else f"bold {ui.PURPLE}")
            out.append("  " + c.help, style="italic" if selected else ui.DIM)
        self.update(out)


class InputArea(TextArea):
    """TextArea with chat keys. Posts InputBox.Submitted via its box."""

    def __init__(self, box: "InputBox", **kwargs) -> None:
        super().__init__(soft_wrap=True, show_line_numbers=False,
                         highlight_cursor_line=False, compact=True, **kwargs)
        self._box = box

    async def _on_key(self, event: events.Key) -> None:
        popup = self._box.popup
        key = event.key
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
    InputBox #input-row { height: auto; border-top: solid $foreground 30%;
                          border-bottom: solid $foreground 30%;
                          border-title-align: right;
                          border-title-color: $text-muted; }
    InputBox #input-row.bash { border-top: solid $secondary;
                               border-bottom: solid $secondary;
                               border-title-color: $secondary; }
    InputBox #glyph { width: 2; color: $accent; text-style: bold; }
    InputBox #input-row.bash #glyph { color: $secondary; }
    InputBox InputArea { width: 1fr; height: auto; min-height: 1;
                         max-height: 12; background: transparent;
                         border: none; padding: 0; }
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
        self.area = InputArea(self, placeholder="Ask anything, /command, "
                                                "or !shell command")

    def compose(self):
        yield self.popup
        with Horizontal(id="input-row"):
            yield Static("›", id="glyph")
            yield self.area

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = self.area.text
        bash = text.startswith("!")
        row = self.query_one("#input-row")
        row.set_class(bash, "bash")
        self.query_one("#glyph", Static).update("!" if bash else "›")
        row.border_title = "bash mode" if bash else ""
        self.popup.show_for(text)

    def set_text(self, text: str) -> None:
        self.area.text = text
        self.area.move_cursor(self.area.document.end)

    def submit(self) -> None:
        text = self.area.text
        if not text.strip():
            return
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

    def set_title(self, title: str) -> None:
        row = self.query_one("#input-row")
        if not row.has_class("bash"):
            row.border_title = title
