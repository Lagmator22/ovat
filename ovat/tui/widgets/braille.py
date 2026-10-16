# ovat/tui/widgets/braille.py
# render_braille() is copied from Mistral Vibe
# (https://github.com/mistralai/mistral-vibe,
# vibe/cli/textual_ui/widgets/braille_renderer.py), Copyright Mistral AI,
# Apache License 2.0, unchanged. ChipMark below is OVAT's own.
"""Dot art in braille characters, and OVAT's animated banner mark.

The mark is a chip whose core fills column by column like an NPU warming
up, then rests lit; it plays at start and again now and then while idle.
"""
import random

from textual.widgets import Static

from collections.abc import Iterable
import math

# for more details on braille characters encoding, see: https://en.wikipedia.org/wiki/Braille_Patterns

_BRAILLE_DOT_COUNT = 8


def _braille_dot_index(x: int, y: int) -> int:
    """returns the number associated with a dot in a braille character
    x ∈ {0, 1}, y ∈ {0, 1, 2, 3}
     -x->
    | 1 4
    y 2 5
    | 3 6
    V 7 8
    """
    if y < 3:  # noqa: PLR2004
        return y + 1 + 3 * x
    return 7 + x


def _braille_char_from_dot_indices(indices: list[int]) -> str:
    if any(n < 1 or n > _BRAILLE_DOT_COUNT for n in indices):
        raise ValueError(f"Invalid braille dot indices: {indices}")
    return chr(0x2800 + sum(2 ** (d - 1) for d in indices)) if indices else " "


def render_braille(dot_coords: Iterable[complex], width: int, height: int) -> str:
    """this function receives a list of dot coordinantes, a width and a height,
    and returns a string representing these dots with braille characters.

    Origin is (0,0) and is located at the top left:
    0----x---->
    |
    y
    |
    V
    """
    dots_matrix: list[list[list[int]]] = [
        [[] for _ in range(math.ceil(width / 2))] for _ in range(math.ceil(height / 4))
    ]  # the list of dots for each character in the final str

    for coord in dot_coords:
        x = int(coord.real // 2)
        y = int(coord.imag // 4)
        sub_x = int(coord.real) % 2
        sub_y = int(coord.imag) % 4
        dots_matrix[y][x].append(_braille_dot_index(sub_x, sub_y))

    braille_chars = [
        [_braille_char_from_dot_indices(char_dots) for char_dots in row]
        for row in dots_matrix
    ]

    return "\n".join("".join(row) for row in braille_chars)


# ---- the OVAT chip mark ---------------------------------------------------

WIDTH, HEIGHT = 22, 12
_FRAME_S = 0.09
_PAUSE_MIN_S, _PAUSE_MAX_S = 6.0, 18.0


def _outline() -> set[complex]:
    dots: set[complex] = set()
    for x in range(4, 18):
        dots |= {complex(x, 1), complex(x, 10)}
    for y in range(1, 11):
        dots |= {complex(4, y), complex(17, y)}
    for y in (3, 5, 8):                      # side pins
        dots |= {complex(x, y) for x in (1, 2, 3)}
        dots |= {complex(x, y) for x in (18, 19, 20)}
    for x in (7, 11, 14):                    # top and bottom pins
        dots |= {complex(x, 0), complex(x, 11)}
    return dots


_CORE_XS = range(7, 15)
_CORE_YS = range(4, 8)


def _core_columns(upto: int) -> set[complex]:
    return {complex(x, y) for x in _CORE_XS[:upto] for y in _CORE_YS}


def frames() -> list[str]:
    """Warm-up: the core fills left to right, flashes once, rests lit."""
    outline = _outline()
    out = []
    for n in range(len(_CORE_XS) + 1):
        out.append(render_braille(outline | _core_columns(n), WIDTH, HEIGHT))
    full = outline | _core_columns(len(_CORE_XS))
    out.append(render_braille(outline, WIDTH, HEIGHT))          # flash off
    out.append(render_braille(full, WIDTH, HEIGHT))             # and on
    return out


class ChipMark(Static):
    """The animated mark. Fixed 11x3 cells, so frames never relayout."""

    DEFAULT_CSS = """
    ChipMark { width: 12; height: 3; color: $accent; }
    """

    def __init__(self, animate: bool = True, **kwargs) -> None:
        self._frames = frames()
        super().__init__(self._frames[0 if animate else -1], **kwargs)
        self._play = animate
        self._index = 0
        self._timer = None

    def on_mount(self) -> None:
        if self._play:
            self._timer = self.set_interval(_FRAME_S, self._next)

    def _next(self) -> None:
        self._index += 1
        if self._index >= len(self._frames):
            self._stop_timer()
            self._timer = self.set_timer(
                random.uniform(_PAUSE_MIN_S, _PAUSE_MAX_S), self._again)
            return
        self.update(self._frames[self._index], layout=False)

    def _again(self) -> None:
        self._index = 0
        self.update(self._frames[0], layout=False)
        self._timer = self.set_interval(_FRAME_S, self._next)

    def freeze(self) -> None:
        """Rest lit (the app calls this while the agent is working)."""
        self._stop_timer()
        self.update(self._frames[-1], layout=False)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
