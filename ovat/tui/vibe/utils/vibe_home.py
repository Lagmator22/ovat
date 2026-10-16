from __future__ import annotations

import os
from pathlib import Path

# OVAT edit: the TUI's own files (cache, caches of the update notifier) live in
# ~/.ovat/tui, selected by OVAT_TUI_HOME, never in Vibe's ~/.vibe / VIBE_HOME,
# so an installed Mistral Vibe and OVAT cannot read or overwrite each other.
_DEFAULT_VIBE_HOME = Path.home() / ".ovat" / "tui"


def get_vibe_home() -> Path:
    if vibe_home := os.getenv("OVAT_TUI_HOME"):
        return Path(vibe_home).expanduser().resolve()
    return _DEFAULT_VIBE_HOME
