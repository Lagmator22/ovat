# ovat/cli/theme.py
"""OVAT's brand palette, expressed as a Textual theme.

Why this exists: the TUI had two colour systems that could not see each
other. Widgets hardcoded hex from ui.PALETTE, while Textual's own CSS
variables ($primary, $success, $surface) came from whatever theme was
active. Switching to Dracula from the command palette therefore recoloured
Textual's chrome and left OVAT's widgets Intel blue on top of it.

Registering the palette AS a theme fixes both directions at once. OVAT's
widgets can now be written against $primary and $success like any other
Textual widget, and they still come out in brand colours because the
default theme IS the brand. Pick another theme and they follow it, because
the variables are the theme's, not ours.

TUI-only, and imported only from tui.py, because textual is an optional
extra: ui.py cannot import this or `ovat --help` would need textual on a
CLI-only install (tests/test_tui_isolation.py enforces that).
"""
from textual.theme import Theme

from ovat.cli import ui

# The two darks the TUI already used for its panels. Kept here rather than in
# ui.PALETTE because they are Textual surface colours, not brand identity.
_BACKGROUND = "#0b0e14"
_SURFACE = "#0d1117"

OVAT_THEME = Theme(
    name="ovat",
    primary=ui.BLUE,        # Intel energy blue: borders, headers, the user
    secondary=ui.PURPLE,    # the accent half of the scheme: menus, input
    accent=ui.CYAN,
    success=ui.GREEN,       # a passing check, and the assistant's frame
    warning=ui.YELLOW,
    error=ui.RED,
    background=_BACKGROUND,
    surface=_SURFACE,
    panel=_SURFACE,
    dark=True,
)
