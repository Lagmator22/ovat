# ovat/tui/widgets/loading.py
"""The line above the input while the agent works: spinner, status, time.

Two animations, both cheap:
- a "snake" walking a 2x4 braille cell (one character wide);
- a colour wave running across the status text, Intel blue to cyan to
  purple and back, one letter per tick.
Every frame is the same width as the last, so updates skip the relayout
(`layout=False`), which keeps a long transcript from re-measuring 10 times
a second. Idea from Mistral Vibe's loading widget; colours and words ours.
"""
import random
import time

from rich.text import Text
from textual.containers import Horizontal
from textual.widgets import Static

from ovat.cli import ui

# Blue -> cyan -> purple: the wave's stops.
WAVE = (ui.BLUE, "#1E90E6", ui.CYAN, "#7AA2FF", ui.PURPLE)

# A path around the braille cell's 8 dots (bit values, Unicode order).
_SNAKE_PATH = (0x01, 0x08, 0x10, 0x20, 0x80, 0x40, 0x04, 0x02)
_SNAKE_LENGTH = 3

# Now and then the generic word is swapped for one of these.
EASTER_EGGS = (
    "Waking the NPU", "Warming the KV cache", "Quantizing to INT4",
    "Compiling for the GPU", "Counting tokens on Lunar Lake",
    "Asking OpenVINO nicely", "Folding tensors", "Consulting the iGPU",
)
_EGG_CHANCE = 0.12


def snake_frame(step: int) -> str:
    """One braille character: three lit dots chasing each other round."""
    mask = 0
    for i in range(_SNAKE_LENGTH):
        mask |= _SNAKE_PATH[(step - i) % len(_SNAKE_PATH)]
    return chr(0x2800 + mask)


def wave_text(text: str, step: int) -> Text:
    """`text` with the colour wave at position `step`."""
    out = Text()
    span = len(WAVE)
    for i, ch in enumerate(text):
        # A smooth triangle wave over the stops, moving right each tick.
        pos = (step - i) % (2 * span - 2)
        index = pos if pos < span else 2 * span - 2 - pos
        out.append(ch, style=f"bold {WAVE[index]}")
    return out


def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m{secs:02d}s"


class LoadingLine(Horizontal):
    """Spinner + waving status + "(12s esc to interrupt)"."""

    DEFAULT_CSS = """
    LoadingLine { height: 1; width: 100%; }
    LoadingLine #spin { width: 2; }
    LoadingLine #status { width: auto; }
    LoadingLine #hint { width: auto; color: $text-muted; padding-left: 1; }
    """

    def __init__(self, status: str = "Thinking", **kwargs) -> None:
        super().__init__(**kwargs)
        self._status = self._maybe_egg(status)
        self._step = 0
        self._started = time.monotonic()
        self._paused_at: float | None = None
        self._paused_total = 0.0
        self._last_hint = ""
        self._spin = Static(snake_frame(0), id="spin")
        self._status_widget = Static(wave_text(self._status + "…", 0),
                                     id="status")
        self._hint = Static("", id="hint")

    def compose(self):
        yield self._spin
        yield self._status_widget
        yield self._hint

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(0.1, self._tick)

    @staticmethod
    def _maybe_egg(status: str) -> str:
        if status == "Thinking" and random.random() < _EGG_CHANCE:
            return random.choice(EASTER_EGGS)
        return status

    def set_status(self, status: str) -> None:
        """Show what is happening now ("Calling search_docs")."""
        if status == self._status:
            return
        self._status = status
        self._status_widget.update(
            wave_text(status + "…", self._step))   # width changed: relayout

    def pause(self) -> None:
        """Stop the clock while a person is being asked something."""
        if self._paused_at is None:
            self._paused_at = time.monotonic()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._paused_total += time.monotonic() - self._paused_at
            self._paused_at = None

    def elapsed(self) -> float:
        paused = self._paused_total + (
            time.monotonic() - self._paused_at if self._paused_at else 0.0)
        return time.monotonic() - self._started - paused

    def _tick(self) -> None:
        self._step += 1
        colour = WAVE[self._step % len(WAVE)]
        self._spin.update(
            Text(snake_frame(self._step), style=f"bold {colour}"),
            layout=False)
        self._status_widget.update(
            wave_text(self._status + "…", self._step), layout=False)
        hint = f"({format_elapsed(self.elapsed())}  esc to interrupt)"
        if hint != self._last_hint:
            self._hint.update(hint, layout=len(hint) != len(self._last_hint))
            self._last_hint = hint
