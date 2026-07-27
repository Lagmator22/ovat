"""A small, self-contained animated-GIF widget for the optional Textual TUI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, ImageSequence, UnidentifiedImageError
from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual.app import RenderResult
from textual.timer import Timer
from textual.widget import Widget


_DEFAULT_SOURCE: Final = Path(__file__).resolve().parent.parent / "assets" / "intel.gif"
_TRANSPARENT_BACKGROUND: Final = (11, 15, 20, 255)


@dataclass(frozen=True)
class _Frame:
    """One already-rendered GIF frame and the time it should remain visible."""

    renderable: Text
    duration: float


class AnimatedGif(Widget):
    """Render a GIF as terminal half-blocks, driven by Textual one-shot timers.

    Pillow decodes and converts every frame once when the widget mounts.  The
    timer only swaps cached Rich ``Text`` renderables, keeping playback fully
    independent of command execution and the rest of the TUI.
    """

    DEFAULT_CSS = """
    AnimatedGif {
        width: 42;
        height: 16;
        content-align: center middle;
    }
    """

    def __init__(self, source: str | Path = _DEFAULT_SOURCE, *, columns: int = 40,
                 name: str | None = None, id: str | None = None,
                 classes: str | None = None, disabled: bool = False) -> None:
        """Create the widget.

        Args:
            source: GIF file to decode. Defaults to OVAT's bundled Intel asset.
            columns: Terminal-cell width used to rasterize the animation.
        """
        if columns < 2:
            raise ValueError("columns must be at least 2")
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._source = Path(source)
        self._columns = columns
        self._frames: list[_Frame] = []
        self._frame_index = 0
        self._timer: Timer | None = None
        self._playing = False

    @property
    def frame_count(self) -> int:
        """Number of decoded frames (zero before mount or after a load error)."""
        return len(self._frames)

    @property
    def frame_index(self) -> int:
        """Index of the frame currently rendered by the widget."""
        return self._frame_index

    @property
    def is_playing(self) -> bool:
        """Whether the animation is currently advancing."""
        return self._playing

    def on_mount(self) -> None:
        """Decode once, then let a widget-owned Textual timer run playback."""
        try:
            self._frames = self._decode_frames()
        except (OSError, UnidentifiedImageError) as error:
            self._frames = []
            self._render_error = Text(f"Could not load animation: {error}", style="red")
            self.refresh()
            return

        if not self._frames:
            self._render_error = Text("Animation contains no frames", style="red")
            self.refresh()
            return

        self._render_error: Text | None = None
        self._frame_index = 0
        self._playing = True
        self.refresh()
        self._schedule_next_frame()

    def on_unmount(self) -> None:
        """Release the timer explicitly when this widget leaves the DOM."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def play(self) -> None:
        """Resume playback without changing the displayed frame."""
        if not self._frames or self._playing:
            return
        self._playing = True
        if self._timer is None:
            self._schedule_next_frame()
        else:
            self._timer.resume()

    def pause(self) -> None:
        """Pause playback, preserving the current frame and remaining timer."""
        if self._timer is not None:
            self._timer.pause()
        self._playing = False

    def stop(self) -> None:
        """Stop playback and return to the first frame."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._playing = False
        if self._frames and self._frame_index != 0:
            self._frame_index = 0
            self.refresh()

    def restart(self) -> None:
        """Start the animation again from its first frame."""
        self.stop()
        self.play()

    def render(self) -> RenderResult:
        """Return the current cached Rich renderable; decoding never happens here."""
        if self._frames:
            return self._frames[self._frame_index].renderable
        return getattr(self, "_render_error", Text(""))

    def _schedule_next_frame(self) -> None:
        """Set one timer using the active frame's original GIF duration."""
        if not self._playing or not self._frames:
            return
        duration = self._frames[self._frame_index].duration
        self._timer = self.set_timer(duration, self._advance_frame)

    def _advance_frame(self) -> None:
        """Move once, repaint once, then schedule the next original duration."""
        self._timer = None
        if not self._playing or not self._frames:
            return
        next_index = (self._frame_index + 1) % len(self._frames)
        if next_index != self._frame_index:
            self._frame_index = next_index
            self.refresh()
        self._schedule_next_frame()

    def _decode_frames(self) -> list[_Frame]:
        """Read every Pillow frame and convert it to a cached Rich ``Text``."""
        frames: list[_Frame] = []
        with Image.open(self._source) as image:
            default_duration = image.info.get("duration", 100)
            for frame in ImageSequence.Iterator(image):
                duration_ms = frame.info.get("duration", default_duration)
                # A zero duration is legal GIF data but cannot be a Textual
                # timer interval. One millisecond preserves its intent best.
                duration = max(float(duration_ms) / 1000, 0.001)
                frames.append(_Frame(self._to_renderable(frame), duration))
        return frames

    def _to_renderable(self, frame: Image.Image) -> Text:
        """Rasterize a Pillow frame into Rich upper-half block characters."""
        target_height = max(2, round(self._columns * frame.height / frame.width))
        if target_height % 2:
            target_height += 1
        image = frame.convert("RGBA").resize(
            (self._columns, target_height), Image.Resampling.LANCZOS
        )
        background = Image.new("RGBA", image.size, _TRANSPARENT_BACKGROUND)
        background.alpha_composite(image)
        pixels = list(background.convert("RGB").get_flattened_data())

        text = Text(no_wrap=True)
        for row in range(0, target_height, 2):
            upper = row * self._columns
            lower = upper + self._columns
            for column in range(self._columns):
                foreground = Color.from_rgb(*pixels[upper + column])
                background_color = Color.from_rgb(*pixels[lower + column])
                text.append("▀", Style(color=foreground, bgcolor=background_color))
            if row + 2 < target_height:
                text.append("\n")
        return text
