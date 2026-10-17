from __future__ import annotations

from ovat.tui.vibe._compat import StrEnum

# Value persisted for the unpinned "default" model state. Mirrors the schema's
# ``UNPINNED_ACTIVE_MODEL`` sentinel; kept UI-side so the textual layer stays
# free of ``vibe.core`` imports (see tests/cli/textual_ui/test_app_server_boundary).
UNPINNED_ACTIVE_MODEL = ""


# OVAT edit: Intel blue -> cyan -> purple in place of Vibe's orange ramp.
# Names kept so the copied code is otherwise unchanged.
class MistralColors(StrEnum):
    RED = "#8F5CFF"
    ORANGE_DARK = "#7AA2FF"
    ORANGE = "#00C7FD"
    ORANGE_LIGHT = "#1E90E6"
    YELLOW = "#0068B5"
