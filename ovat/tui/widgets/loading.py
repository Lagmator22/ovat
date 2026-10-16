# ovat/tui/widgets/loading.py
# SnakeSpinner is copied from Mistral Vibe
# (https://github.com/mistralai/mistral-vibe, vibe/cli/textual_ui/widgets/
# spinner.py), Copyright Mistral AI, Apache License 2.0. Changed for OVAT:
# no Spinner base class; the rest of the file is OVAT's.
"""The line above the input while the agent works: spinner, status, time.

Two animations, both cheap:
- a "snake" walking a 2x4 braille cell (one character wide);
- a colour wave running across the status text, Intel blue to cyan to
  purple and back, one letter per tick.
Every frame is the same width as the last, so updates skip the relayout
(`layout=False`), which keeps a long transcript from re-measuring 10 times
a second.
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


class SnakeSpinner:
    MAP_WIDTH = 4
    MAP_HEIGHT = 4
    SNAKE_LENGTH = 3

    def __init__(self) -> None:
        self._positions: list[complex] = [1, 0, 1j]

    @property
    def current_direction(self) -> complex:
        return self._positions[0] - self._positions[1]

    def _is_in_bounds(self, position: complex) -> bool:
        return (
            0 <= position.real < self.MAP_WIDTH
            and 0 <= position.imag < self.MAP_HEIGHT
        )

    def _get_direction(self) -> complex:
        if (
            len(set(z.real for z in self._positions)) > 1
            and len(set(z.imag for z in self._positions)) > 1
            and self._is_in_bounds(self._positions[0] + self.current_direction)
        ):
            return self.current_direction
        valid_directions = []
        for rotation in [1, 1j, -1j]:
            offset = rotation * self.current_direction
            new_position = self._positions[0] + offset
            if self._is_in_bounds(new_position) and new_position not in self._positions:
                valid_directions.append(offset)
        return random.choice(valid_directions)

    def _next_positions(self) -> list[complex]:
        if len(self._positions) > self.SNAKE_LENGTH:
            return self._positions[: self.SNAKE_LENGTH]
        head_position = self._positions[0]
        direction = self._get_direction()
        if self.current_direction != direction:
            return [head_position + direction] + self._positions
        return [head_position + direction] + self._positions[:-1]

    def current_frame(self) -> str:
        from ovat.tui.widgets.braille import render_braille
        return render_braille(self._positions, self.MAP_WIDTH, self.MAP_HEIGHT)

    def next_frame(self) -> str:
        self._positions = self._next_positions()
        return self.current_frame()

    def reset(self) -> None:
        self._positions = [1, 0, 1j]


def snake_frame(step: int) -> str:
    """One braille character: three lit dots chasing each other round
    (kept for the banner's cheap one-cell spinner)."""
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
    LoadingLine #spin { width: 3; }
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
        self._snake = SnakeSpinner()
        self._spin = Static(self._snake.next_frame(), id="spin")
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
            Text(self._snake.next_frame(), style=f"bold {colour}"),
            layout=False)
        self._status_widget.update(
            wave_text(self._status + "…", self._step), layout=False)
        hint = f"({format_elapsed(self.elapsed())} Esc/Ctrl+C to interrupt)"
        if hint != self._last_hint:
            self._hint.update(hint, layout=len(hint) != len(self._last_hint))
            self._last_hint = hint
