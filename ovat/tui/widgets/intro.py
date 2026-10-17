# ovat/tui/widgets/intro.py
"""The startup intro: the Intel GIF, full screen and centred, then the chat.

An overlay ON the main screen, not a screen of its own. As a separate
screen it sat on top while the agent finished building, the app's queries
then searched the intro instead of the chat, and the "agent ready" report
was dropped: the banner said "Loading the agent" forever (AI PC, 2026-10-11).

Frames are converted on a worker thread, so the title shows at once rather
than a black screen while 116 GIF frames are rasterised, and every second
GIF frame is played, which doubles the speed. Any key or click skips.
"""
import threading

from rich.text import Text
from textual.containers import Center, Vertical
from textual.message import Message
from textual.widgets import Static

from ovat.cli import ui
from ovat.tui.widgets.loading import wave_text

# Rows per column for the artwork (400x260 crop, 2x4 pixels per cell).
_ROWS_PER_COLUMN = 0.33
_MAX_COLUMNS = 84
_HOLD_LAST_FRAME_S = 0.6
_GIVE_UP_S = 10.0              # never trap anyone behind a slow decode
_BOARD_SECONDS = 0.35          # the three still boards after the GIF
_BOARDS = 3


def mark_size(width: int, height: int) -> tuple[int, int]:
    """(columns, rows) for the biggest mark that fits with a title under it."""
    columns = max(20, min(width - 4, _MAX_COLUMNS))
    while columns > 20 and round(columns * _ROWS_PER_COLUMN) > height - 5:
        columns -= 2
    return columns, round(columns * _ROWS_PER_COLUMN)


def _fast_mark_class():
    from ovat.cli.tui import _StartupIntelAnimation

    class FastIntelMark(_StartupIntelAnimation):
        """The launcher's renderer, decoded off the UI thread, 2x speed."""

        def on_mount(self) -> None:
            threading.Thread(target=self._decode_in_background,
                             daemon=True, name="ovat-intro").start()

        def _decode_in_background(self) -> None:
            try:
                frames = self._decode_frames()
            except Exception:
                frames = ()
            gif, boards = frames[:-_BOARDS], frames[-_BOARDS:]
            frames = tuple(gif[::2]) + tuple(boards)
            try:
                self.app.call_from_thread(self._play, frames)
            except Exception:
                pass                    # the app closed meanwhile

        def _play(self, frames) -> None:
            self._frames = frames
            self._frame_index = 0
            self.refresh()
            self._schedule_advance()

    return FastIntelMark


class IntroOverlay(Vertical, can_focus=True):
    DEFAULT_CSS = """
    IntroOverlay { width: 100%; height: 100%; align: center middle;
                   background: $background; }
    IntroOverlay Center { height: auto; }
    IntroOverlay #intro-title { width: 100%; height: 1;
                                content-align: center middle; margin-top: 1; }
    IntroOverlay #intro-skip { width: 100%; height: 1;
                               content-align: center middle; color: $text-muted; }
    """

    class Finished(Message):
        """The intro is over (played out, skipped, or gave up)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._waited = 0.0
        self._held = 0.0
        self._over = False
        self._step = 0

    def compose(self):
        columns, rows = mark_size(self.app.size.width, self.app.size.height)
        mark = _fast_mark_class()(columns=columns, phase_seconds=_BOARD_SECONDS,
                                  id="intro-mark")
        mark.styles.width = columns
        mark.styles.height = rows
        with Center():
            yield mark
        yield Static(self._title(), id="intro-title")
        yield Static("any key to skip", id="intro-skip")

    def _title(self) -> Text:
        title = Text()
        title.append("OVAT", style=f"bold {ui.CYAN}")
        title.append(" agent", style=f"bold {ui.PURPLE}")
        return title

    def on_mount(self) -> None:
        self.focus()
        self.set_interval(0.1, self._check)

    def _check(self) -> None:
        mark = self.query_one("#intro-mark")
        if not mark.frame_count:
            # Still decoding: keep something moving, never a black screen.
            self._step += 1
            self.query_one("#intro-skip", Static).update(
                wave_text("starting…", self._step))
            self._waited += 0.1
            if self._waited >= _GIVE_UP_S:
                self.finish()
            return
        if not mark.is_animating:
            self._held += 0.1
            if self._held >= _HOLD_LAST_FRAME_S:
                self.finish()

    def on_key(self, event) -> None:
        event.stop()
        event.prevent_default()
        self.finish()

    def on_click(self) -> None:
        self.finish()

    def finish(self) -> None:
        if not self._over:
            self._over = True
            self.post_message(self.Finished())
