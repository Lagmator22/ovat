from __future__ import annotations

# OVAT edit: Vibe's NarratorManager (Mistral's text to speech) is not
# copied; OVAT passes the copied app a narrator that does nothing.
from ovat.tui.vibe.cli.narrator_manager.narrator_manager_port import (
    NarratorManagerListener,
    NarratorManagerPort,
    NarratorState,
)

__all__ = ["NarratorManagerListener", "NarratorManagerPort", "NarratorState"]
