# ovat/tui/widgets/messages.py
"""What the transcript is made of: questions, answers, notices, shell runs.

Every piece of text that OVAT did not write itself (the question, the
answer, a path, an exception, shell output) goes in as a rich Text object
or Markdown, never as console markup: a path like "C:\\x\\[/y]" in markup
is a crash or a styling accident, as a Text it is just characters.
"""
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.widgets import Markdown, Static

from ovat.cli import ui

PROMPT_GLYPH = "›"
_MAX_SHELL_LINES = 2000


class UserMessage(Horizontal):
    DEFAULT_CSS = """
    UserMessage { height: auto; width: 100%; margin-top: 1; }
    UserMessage .glyph { width: 2; color: $accent; text-style: bold; }
    UserMessage .text { width: 1fr; text-style: bold; }
    """

    def __init__(self, text: str, glyph: str = PROMPT_GLYPH,
                 colour: str | None = None) -> None:
        super().__init__()
        self._text, self._glyph, self._colour = text, glyph, colour

    def compose(self):
        glyph = Static(Text(self._glyph), classes="glyph")
        if self._colour:
            glyph.styles.color = self._colour
        yield glyph
        yield Static(Text(self._text), classes="text")


class AssistantMessage(Vertical):
    """The answer, as Markdown, with a dim footer of what it cost."""

    DEFAULT_CSS = """
    AssistantMessage { height: auto; width: 100%; margin-top: 1;
                       padding-left: 2; }
    AssistantMessage Markdown { margin: 0; padding: 0;
                                background: transparent; }
    AssistantMessage .footer { color: $text-muted; height: auto; }
    AssistantMessage.failed { border-left: heavy $error; padding-left: 1; }
    """

    def __init__(self, markdown: str, footer: Text | None = None,
                 failed: bool = False) -> None:
        super().__init__(classes="failed" if failed else "")
        self._markdown, self._footer = markdown, footer

    def compose(self):
        yield Markdown(self._markdown)
        if self._footer is not None:
            yield Static(self._footer, classes="footer")


class Notice(Static):
    """A line from OVAT itself: info, warn or error, with a coloured bar."""

    DEFAULT_CSS = """
    Notice { height: auto; width: 100%; margin-top: 1; padding: 0 1;
             border-left: heavy $accent; color: $text-muted; }
    Notice.warn { border-left: heavy $warning; color: $warning; }
    Notice.error { border-left: heavy $error; color: $error; }
    """

    def __init__(self, text: str | Text, level: str = "info") -> None:
        body = text if isinstance(text, Text) else Text(text)
        super().__init__(body, classes=level if level != "info" else "")


class ShellBlock(Vertical):
    """`!cmd`: the command, its output as it arrives, and how it ended."""

    DEFAULT_CSS = """
    ShellBlock { height: auto; width: 100%; margin-top: 1; }
    ShellBlock .cmd { height: 1; color: $secondary; text-style: bold; }
    ShellBlock .out { height: auto; padding-left: 2; color: $foreground; }
    ShellBlock .end { height: 1; padding-left: 2; color: $text-muted; }
    """

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command
        self._lines: list[str] = []
        self._dropped = 0
        # Made here so output arriving before the mount finishes is kept.
        self._out = Static("", classes="out")
        self._end = Static(Text("running…", style=ui.DIM), classes="end")

    def compose(self):
        yield Static(Text(f"! {self.command}"), classes="cmd")
        yield self._out
        yield self._end

    def add_line(self, line: str) -> None:
        self._lines.append(line)
        if len(self._lines) > _MAX_SHELL_LINES:
            self._lines.pop(0)
            self._dropped += 1
        head = (f"[{self._dropped} earlier lines not shown]\n"
                if self._dropped else "")
        # from_ansi: a command's own colours survive, and nothing it prints
        # is read as markup.
        self._out.update(Text.from_ansi(head + "\n".join(self._lines)))

    def finish(self, code: int | None, elapsed: float) -> None:
        end = Text()
        if code == 0:
            end.append("✓ ", style=ui.GREEN)
            end.append(f"done in {elapsed:.1f}s", style=ui.DIM)
        else:
            end.append("✕ ", style=ui.RED)
            what = "stopped" if code is None or code < 0 else f"exit {code}"
            end.append(f"{what} after {elapsed:.1f}s", style=ui.DIM)
        self._end.update(end)


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
    out.append(f"  {elapsed:.1f}s", style=ui.DIM)
    calls = totals.get("tool_calls")
    if calls:
        out.append(f"  ·  {calls} tool{'s' if calls != 1 else ''}",
                   style=ui.DIM)
    prompt, completion = (totals.get("prompt_tokens"),
                          totals.get("completion_tokens"))
    if prompt is not None and completion is not None:
        out.append(f"  ·  {prompt:,} → {completion:,} tokens", style=ui.DIM)
        latency = totals.get("latency_s") or elapsed
        if completion and latency:
            out.append(f"  ·  {completion / latency:.0f} tok/s",
                       style=f"{ui.CYAN}")
    if sources:
        import re
        names = []
        for src in sources:
            name = re.split(r"[\\/]", str(src))[-1]
            if name and name not in names:
                names.append(name)
        out.append("  ·  ", style=ui.DIM)
        out.append(", ".join(names[:4]), style=ui.PURPLE)
        if len(names) > 4:
            out.append(f" +{len(names) - 4}", style=ui.DIM)
    return out
