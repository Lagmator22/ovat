# ovat/tui/vibe/_compat.py (OVAT's, not part of the copied code)
"""Python 3.10 and 3.11 stand-ins for the 3.11 / 3.12 names the copied code
uses, and tiny replacements for two small third-party helpers.

OVAT supports Python >= 3.10; the copied code was written for 3.12. Every
name the copied code needs from a newer Python lives here, so a copied file
changes one import line and nothing else.
"""
from __future__ import annotations

import asyncio
import enum
import json
import math
import re
from datetime import datetime
from typing import Any, BinaryIO

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

try:
    import tomllib
except ImportError:                      # Python 3.10: the `tomli` package
    import tomli as tomllib              # type: ignore[no-redef]  # noqa: F401

from typing_extensions import Self, assert_never, override  # noqa: E402,F401


def task_cancelling(task: asyncio.Task[Any]) -> int:
    """``Task.cancelling()`` (3.11+). On 3.10 there is no such count, so a
    task is reported as not being cancelled: the caller then treats a
    ``CancelledError`` from an awaited task as that task's own."""
    cancelling = getattr(task, "cancelling", None)
    return cancelling() if cancelling is not None else 0


_FRACTION = re.compile(r"(\.\d+)")


def fromisoformat(value: str) -> datetime:
    """``datetime.fromisoformat`` that also takes a trailing ``Z`` and any
    number of fractional-second digits, which 3.11+ accepts and 3.10 does not."""
    value = value.strip()
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    value = _FRACTION.sub(lambda m: m.group(1)[:7].ljust(7, "0"), value, count=1)
    return datetime.fromisoformat(value)


# ---- replaces `humanize` (naturaldelta, naturalsize) -----------------------

def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many.format(n)


def naturaldelta(seconds: float) -> str:
    """Replaces ``humanize.naturaldelta`` for a length of time in seconds:
    "7 days", "an hour", "a moment"."""
    s = abs(int(seconds))
    if s < 1:
        return "a moment"
    if s < 60:
        return _plural(s, "a second", "{} seconds")
    if s < 3600:
        m = s // 60
        return _plural(m, "a minute", "{} minutes")
    if s < 86400:
        h = s // 3600
        return _plural(h, "an hour", "{} hours")
    days = s // 86400
    if days < 30:
        return _plural(days, "a day", "{} days")
    if days < 365:
        months = round(days / 30.5)
        return _plural(months, "a month", "{} months")
    years = days // 365
    return _plural(years, "a year", "{} years")


def naturalsize(value: float, binary: bool = False) -> str:
    """Replaces ``humanize.naturalsize``: "1 Byte", "512 Bytes", "10.0 MiB"
    (``binary=True``) or "10.5 MB"."""
    base = 1024 if binary else 1000
    units = (("KiB", "MiB", "GiB", "TiB", "PiB") if binary
             else ("kB", "MB", "GB", "TB", "PB"))
    n = abs(value)
    if n == 1:
        return "1 Byte"
    if n < base:
        return f"{int(value)} Bytes"
    exp = min(int(math.log(n, base)), len(units))
    return f"{value / base ** exp:.1f} {units[exp - 1]}"


# ---- replaces `tomli_w` (dump) ---------------------------------------------

def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items())
        return "{" + inner + "}"
    raise TypeError(f"cannot write {type(value).__name__} as TOML")


def toml_dumps(data: dict[str, Any]) -> str:
    """Replaces ``tomli_w.dumps`` for the plain data the copied cache store
    writes: str, int, float, bool, datetime, lists, and tables of those."""
    lines: list[str] = []

    def table(prefix: tuple[str, ...], tbl: dict[str, Any]) -> None:
        plain = {k: v for k, v in tbl.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in tbl.items() if isinstance(v, dict)}
        if prefix and (plain or not nested):
            if lines:
                lines.append("")
            lines.append("[" + ".".join(_toml_key(p) for p in prefix) + "]")
        for k, v in plain.items():
            lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
        for k, v in nested.items():
            table(prefix + (k,), v)

    table((), data)
    return "\n".join(lines) + "\n"


def toml_dump(data: dict[str, Any], fp: BinaryIO) -> None:
    """Replaces ``tomli_w.dump``: write TOML to a binary file."""
    fp.write(toml_dumps(data).encode("utf-8"))
