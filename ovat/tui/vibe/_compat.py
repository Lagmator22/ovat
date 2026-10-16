# ovat/tui/vibe/_compat.py (OVAT's, not part of the copied code)
"""Python 3.10 and 3.11 stand-ins for the 3.11 / 3.12 names the copied code
uses. OVAT supports Python >= 3.10; the copied code was written for 3.12."""
import enum

try:
    from enum import StrEnum
except ImportError:                      # Python 3.10
    class StrEnum(str, enum.Enum):
        """str-valued Enum; auto() gives the lower-cased member name."""

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

        def __str__(self) -> str:
            return str(self.value)

try:
    from datetime import UTC
except ImportError:                      # Python 3.10
    from datetime import timezone
    UTC = timezone.utc

from typing_extensions import Self, assert_never, override  # noqa: E402,F401
