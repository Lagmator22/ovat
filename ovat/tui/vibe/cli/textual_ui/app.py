# ovat/tui/vibe/cli/textual_ui/app.py
# ChatScroll is copied from Mistral Vibe (vibe/cli/textual_ui/app.py),
# Copyright Mistral AI, Apache License 2.0, unchanged. The rest of Vibe's
# app.py (its App class and agent wiring) is NOT copied: OVAT's app is
# ovat/tui/app.py. This module exists so the copied widgets' imports of
# ChatScroll resolve.
from __future__ import annotations

from collections.abc import Callable

from textual.containers import VerticalScroll
from textual.dom import NoScreen
from textual.events import MouseScrollDown, MouseScrollUp

WHEEL_SCROLL_DURATION = 0.1


class ChatScroll(VerticalScroll):
    """Optimized scroll container that skips cascading style recalculations."""

    @property
    def is_at_bottom(self) -> bool:
        return self.scroll_target_y >= self.max_scroll_y

    _reanchor_pending: bool = False
    _scrolling_down: bool = False

    @property
    def _is_selecting(self) -> bool:
        try:
            return self.screen._selecting
        except NoScreen:
            return False

    def anchor(self, anchor: bool = True) -> None:
        if anchor and self._is_selecting:
            return
        super().anchor(anchor)

    def preserve_scroll_position(self) -> None:
        super().release_anchor()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        if self._is_selecting and new_value < old_value:
            self._anchor_released = True
        super().watch_scroll_y(old_value, new_value)
        self._scrolling_down = new_value >= old_value

    def release_anchor(self) -> None:
        super().release_anchor()
        # Textual's MRO dispatch calls Widget._on_mouse_scroll_down AFTER
        # our override, so any re-anchor we do gets immediately undone.
        # Defer the re-check until all handlers for this event have finished.
        if not self._reanchor_pending:
            self._reanchor_pending = True
            self.call_later(self._maybe_reanchor)

    def _maybe_reanchor(self) -> None:
        self._reanchor_pending = False
        if (
            self._anchored
            and self._anchor_released
            and self.is_at_bottom
            and self._scrolling_down
        ):
            self.anchor()

    def update_node_styles(self, animate: bool = True) -> None:
        pass

    def _on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self._smooth_wheel_scroll(event, self._scroll_down_for_pointer)

    def _on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self._smooth_wheel_scroll(event, self._scroll_up_for_pointer)

    def _smooth_wheel_scroll(
        self, event: MouseScrollDown | MouseScrollUp, scroller: Callable[..., bool]
    ) -> None:
        # Leave ctrl/shift (horizontal) wheel to the base Widget handler, which
        # Textual's MRO dispatch still reaches because we don't prevent_default.
        if event.ctrl or event.shift or not self.allow_vertical_scroll:
            return
        # Cover the same per-notch distance as the default handler, but render it
        # as a smooth linear glide so motion passes through each line one by one
        # instead of jumping the full sensitivity in a single frame. prevent_default
        # breaks the MRO loop so the base handler's instant jump never runs on top.
        event.prevent_default()
        if scroller(animate=True, duration=WHEEL_SCROLL_DURATION, easing="linear"):
            event.stop()

