# ovat/tui/widgets/intro.py
"""The startup intro: the Intel GIF, full screen and centred, then the chat.

Plays assets/intel.gif and the three Intel boards once, as large as the
terminal allows, through the launcher's own renderer
(cli/tui._StartupIntelAnimation). Any key skips it. The agent is built in
the background meanwhile, so the intro costs no time. Afterwards the
banner shows the settled mark as a still image (StaticIntelMark).
"""
from rich.text import Text
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Static

from ovat.cli import ui

# Height / width of the artwork per terminal column, in rows: the GIF crop
# is 400x260 and a cell is two pixels wide by four (braille) tall, so about
# 0.33; the boards (92x60) give the same.
_ROWS_PER_COLUMN = 0.33
_HOLD_LAST_FRAME_S = 0.8


def mark_size(width: int, height: int) -> tuple[int, int]:
    """(columns, rows) for the biggest mark that fits with room for a
    title line under it."""
    columns = max(20, min(width - 4, 96))
    while columns > 20 and round(columns * _ROWS_PER_COLUMN) > height - 4:
        columns -= 2
    return columns, round(columns * _ROWS_PER_COLUMN)


def _animation_class():
    from ovat.cli.tui import _StartupIntelAnimation
    return _StartupIntelAnimation


def static_mark(columns: int, **kwargs):
    """The settled Intel mark, no animation: its last frame only."""
    base = _animation_class()

    class StaticIntelMark(base):
        def on_mount(self) -> None:
            super().on_mount()
            self._cancel_timer()
            if self.frame_count:
                self._frame_index = self.frame_count - 1
                self.refresh()

    return StaticIntelMark(columns=columns, **kwargs)


class IntroScreen(Screen):
    DEFAULT_CSS = """
    IntroScreen { align: center middle; background: $background; }
    IntroScreen #intro { width: auto; height: auto; align: center middle; }
    IntroScreen #intro-title { width: 100%; height: 1; content-align: center middle;
                               margin-top: 1; }
    IntroScreen #intro-skip { width: 100%; height: 1; content-align: center middle;
                              color: $text-muted; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._done_at: float | None = None
        self._leaving = False

    def compose(self):
        columns, rows = mark_size(self.app.size.width, self.app.size.height)
        mark = _animation_class()(columns=columns, id="intro-mark")
        mark.styles.width = columns
        mark.styles.height = rows
        with Vertical(id="intro"):
            yield mark
            title = Text()
            title.append("OVAT", style=f"bold {ui.CYAN}")
            title.append(" agent", style=f"bold {ui.PURPLE}")
            yield Static(title, id="intro-title")
            yield Static("any key to skip", id="intro-skip")

    def on_mount(self) -> None:
        self.set_interval(0.1, self._check)

    def _check(self) -> None:
        mark = self.query_one("#intro-mark")
        if not mark.frame_count:
            # Decoding failed or is still running; never trap the user.
            if self._done_at is None:
                self._done_at = 0.0
            self._done_at += 0.1
            if self._done_at > 3.0:
                self._close()
            return
        if not mark.is_animating:
            self._done_at = (self._done_at or 0.0) + 0.1
            if self._done_at >= _HOLD_LAST_FRAME_S:
                self._close()

    def on_key(self, event) -> None:
        event.stop()
        self._close()

    def on_click(self) -> None:
        self._close()

    def _close(self) -> None:
        if not self._leaving:
            self._leaving = True
            self.dismiss(None)
