from __future__ import annotations

import os
from pathlib import Path

# Strip the "For further information visit https://errors.pydantic.dev/..." line
# from ValidationError string output. pydantic-core caches this setting on the
# first error render, so it must be set before any validation error is stringified.
os.environ.setdefault("PYDANTIC_ERRORS_INCLUDE_URL", "0")

VIBE_ROOT = Path(__file__).parent
# OVAT edit: the version shown and compared by the update check is OVAT's own
# (Vibe: "2.26.1").
try:
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("ovat")
except PackageNotFoundError:  # not installed, e.g. run from a plain checkout
    __version__ = "0"
