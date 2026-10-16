# ovat/tui/widgets/messages.py
# Adapted from Mistral Vibe (https://github.com/mistralai/mistral-vibe,
# vibe/cli/textual_ui/widgets/messages.py and app.tcss), Copyright Mistral AI,
# Apache License 2.0. Changed for OVAT: no Vibe history/attachment/loop
# state, OVAT colours, a turn footer, Text instead of markup everywhere.
"""What the transcript is made of: questions, answers, notices, shell runs.

Every string OVAT did not write (the question, the answer, a path, an
exception, shell output) goes in as Text or Markdown, never as console
markup: "C:\\x\\[/y]" in markup is a crash, as Text it is characters.
"""
import re

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.geometry import Size
from textual.widgets import Markdown, Static

from ovat.cli import ui

PROMPT_GLYPH = ">"
PREVIEW_LINES = 20
_MAX_SHELL_LINES = 2000
_PULSE = ("■", "■", "■", "□", "□")


class NoMarkupStatic(Static):
    def __init__(self, content="", **kwargs) -> None:
        super().__init__(content, markup=False, **kwargs)


class ExpandingBorder(NoMarkupStatic):
    """A one-column left rule, ⎢ down to ⎣, as tall as its neighbour."""

    def get_content_width(self, container: Size, viewport: Size) -> int:
        return 1

    def render(self) -> str:
        height = max(self.size.height, 1)
        return "\n".join(["⎢"] * (height - 1) + ["⎣"])

    def on_resize(self) -> None:
        self.refresh()


class ExpandingSeparator(NoMarkupStatic):
    """A full-width rule under a question."""

    def render(self) -> str:
        return "─" * max(self.size.width, 1)

    def on_resize(self) -> None:
        self.refresh()


class UserMessage(Static):
    DEFAULT_CSS = """
    UserMessage { width: 100%; height: auto; margin-top: 1; }
    UserMessage .user-message-container { width: 100%; height: auto; }
    UserMessage .user-message-prompt { width: auto; height: auto;
                                       color: $accent; text-style: bold; }
    UserMessage .user-message-content { width: 1fr; height: auto;
                                        text-style: bold; }
    UserMessage .user-message-separator { width: 100%; height: 1;
                                          color: $text-muted; }
    UserMessage.slash .user-message-content { color: $secondary; }
    """

    def __init__(self, text: str, slash: bool = False) -> None:
        super().__init__(classes="slash" if slash else "")
        self.text = text

    def compose(self):
        with Horizontal(classes="user-message-container"):
            yield NoMarkupStatic(f"{PROMPT_GLYPH} ",
                                 classes="user-message-prompt")
            yield NoMarkupStatic(self.text, classes="user-message-content")
        yield ExpandingSeparator(classes="user-message-separator")


class AssistantMessage(Vertical):
    """The answer as Markdown, with a dim footer of what it cost."""

    DEFAULT_CSS = """
    AssistantMessage { width: 100%; height: auto; margin-top: 1;
                       padding: 0 2 0 2; }
    AssistantMessage Markdown { width: 100%; height: auto; padding: 0;
                                margin: 0; background: transparent; }
    AssistantMessage .footer { height: auto; color: $text-muted; }
    AssistantMessage.failed Markdown { color: $error; }
    """

    def __init__(self, markdown: str, footer: Text | None = None,
                 failed: bool = False) -> None:
        super().__init__(classes="failed" if failed else "")
        self.markdown = markdown
        self.footer = Static(footer or "", classes="footer")

    def compose(self):
        yield Markdown(self.markdown)
        yield self.footer


class _BorderedLine(Static):
    """Shared shape of command output, interrupt, warning and error lines."""

    KIND = "user-command"
    DEFAULT_CSS = """
    _BorderedLine { width: 100%; height: auto; margin-top: 1; }
    _BorderedLine .line-container { width: 100%; height: auto; }
    _BorderedLine .line-border { width: auto; height: 100%;
                                 padding: 0 1 0 2; color: $text-muted; }
    _BorderedLine .line-content { width: 1fr; height: auto;
                                  padding-right: 2; }
    """

    def __init__(self, content: str | Text) -> None:
        super().__init__(classes=f"{self.KIND}-message")
        self.content = content if isinstance(content, Text) else Text(content)

    def compose(self):
        with Horizontal(classes="line-container"):
            yield ExpandingBorder(classes="line-border")
            yield Static(self.content, classes=f"line-content "
                                               f"{self.KIND}-content")


class CommandOutput(_BorderedLine):
    """What a slash command printed (/help, /tools, /agents ...)."""


class InterruptMessage(_BorderedLine):
    KIND = "interrupt"
    DEFAULT_CSS = """
    InterruptMessage .interrupt-content { color: $warning; }
    """

    def __init__(self) -> None:
        super().__init__("Interrupted · What should OVAT do instead?")


class WarningMessage(_BorderedLine):
    KIND = "warning"
    DEFAULT_CSS = """
    WarningMessage .warning-content { color: $warning; }
    """


class ErrorMessage(_BorderedLine):
    KIND = "error"
    DEFAULT_CSS = """
    ErrorMessage .error-content { color: $error; text-style: bold; }
    """


def notice(text: str | Text, level: str = "info") -> _BorderedLine:
    """The right line for a session Notice."""
    return {"warn": WarningMessage, "error": ErrorMessage}.get(
        level, CommandOutput)(text)


class WhatsNewMessage(Static):
    DEFAULT_CSS = """
    WhatsNewMessage { width: 100%; height: auto; margin-top: 1;
                      border-left: heavy $secondary; }
    WhatsNewMessage Markdown { padding: 0 0 0 1; margin: 0;
                               background: transparent; }
    """

    def __init__(self, markdown: str) -> None:
        super().__init__()
        self.markdown = markdown

    def compose(self):
        yield Markdown(self.markdown)


class BashOutputMessage(Static):
    """`!cmd`: a pulsing prompt while it runs, `$` and its colour after;
    the first 20 lines show, the rest fold behind "+N lines"."""

    DEFAULT_CSS = """
    BashOutputMessage { width: 100%; height: auto; margin-top: 1; }
    BashOutputMessage .bash-command-line { width: 100%; height: auto; }
    BashOutputMessage .bash-prompt { width: auto; height: auto; }
    BashOutputMessage.bash-success .bash-prompt { color: $success; }
    BashOutputMessage.bash-error .bash-prompt { color: $error; }
    BashOutputMessage.bash-interrupted .bash-prompt { color: $warning; }
    BashOutputMessage .bash-command { width: 1fr; height: auto; }
    BashOutputMessage .bash-output-container { width: 100%; height: auto;
                                               display: none; }
    BashOutputMessage.has-output .bash-output-container { display: block; }
    BashOutputMessage .bash-output-border { width: auto; height: 100%;
                                            padding: 0 1 0 2;
                                            color: $text-muted; }
    BashOutputMessage .bash-output-body { width: 1fr; height: auto;
                                          padding-right: 2; }
    BashOutputMessage .bash-more { height: auto; color: $text-muted; }
    BashOutputMessage .bash-time { height: auto; color: $text-muted; }
    """

    def __init__(self, command: str) -> None:
        super().__init__(classes="bash-pending")
        self.command = command
        self._lines: list[str] = []
        self._dropped = 0
        self._open = False
        self._step = 0
        self._timer = None
        self._prompt = NoMarkupStatic(f"{_PULSE[0]} ", classes="bash-prompt")
        self._output = Static("", classes="bash-output")
        self._more = NoMarkupStatic("", classes="bash-more")
        self._more.display = False
        self._time = NoMarkupStatic("", classes="bash-time")

    def compose(self):
        with Horizontal(classes="bash-command-line"):
            yield self._prompt
            yield NoMarkupStatic(self.command, classes="bash-command")
        with Horizontal(classes="bash-output-container"):
            yield ExpandingBorder(classes="bash-output-border")
            with Vertical(classes="bash-output-body"):
                yield self._output
                yield self._more
                yield self._time

    def on_mount(self) -> None:
        # A fast command can finish before this mounts; then there is
        # nothing to pulse, and starting the timer would put the square
        # back over the "$".
        if self.has_class("bash-pending"):
            self._timer = self.set_interval(0.1, self._pulse)

    def _pulse(self) -> None:
        self._step += 1
        self._prompt.update(f"{_PULSE[self._step % len(_PULSE)]} ",
                            layout=False)

    def add_line(self, line: str) -> None:
        self._lines.append(line)
        if len(self._lines) > _MAX_SHELL_LINES:
            self._lines.pop(0)
            self._dropped += 1
        self.add_class("has-output")
        self._redraw()

    def _redraw(self) -> None:
        shown = self._lines if self._open else self._lines[:PREVIEW_LINES]
        head = (f"[{self._dropped} earlier lines not kept]\n"
                if self._dropped and self._open else "")
        # from_ansi: the command's own colours survive, nothing is markup.
        self._output.update(Text.from_ansi(head + "\n".join(shown)))
        hidden = len(self._lines) - PREVIEW_LINES
        self._more.display = hidden > 0
        self._more.update("show less" if self._open else f"+{hidden} lines")

    def on_click(self) -> None:
        if len(self._lines) > PREVIEW_LINES:
            self._open = not self._open
            self._redraw()

    def finish(self, code: int | None, elapsed: float) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.remove_class("bash-pending")
        if code == 0:
            self.add_class("bash-success")
        elif code is None or code < 0:
            self.add_class("bash-interrupted")
        else:
            self.add_class("bash-error")
        self._prompt.update("$ ")
        what = ("done" if code == 0 else "stopped" if code is None or code < 0
                else f"exit {code}")
        self._time.update(f"{what} in {elapsed:.1f}s")
        self.add_class("has-output")

    @property
    def output_text(self) -> str:
        return "\n".join(self._lines)


def turn_footer(elapsed: float, totals: dict, sources: list,
                failed: bool, cancelled: bool) -> Text:
    """"✓ 6.3s · 2 tools · 1,812 → 96 tokens · 31 tok/s · leave.md"."""
    out = Text()
    if cancelled:
        out.append("■ stopped", style=ui.YELLOW)
    elif failed:
        out.append("✕ failed", style=ui.RED)
    else:
        out.append("✓", style=ui.GREEN)
    out.append(f" {elapsed:.1f}s")
    calls = totals.get("tool_calls")
    if calls:
        out.append(f" · {calls} tool{'s' if calls != 1 else ''}")
    prompt, completion = (totals.get("prompt_tokens"),
                          totals.get("completion_tokens"))
    if prompt is not None and completion is not None:
        out.append(f" · {prompt:,} → {completion:,} tokens")
        latency = totals.get("latency_s") or elapsed
        if completion and latency:
            out.append(f" · {completion / latency:.0f} tok/s", style=ui.CYAN)
    if sources:
        names = []
        for src in sources:
            name = re.split(r"[\\/]", str(src))[-1]
            if name and name not in names:
                names.append(name)
        out.append(" · ")
        out.append(", ".join(names[:4]), style=ui.PURPLE)
        if len(names) > 4:
            out.append(f" +{len(names) - 4}")
    return out
