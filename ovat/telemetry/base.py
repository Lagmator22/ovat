# ovat/telemetry/base.py
"""Telemetry contracts separating metric sources from output sinks.

``TelemetrySource`` defines where measurements originate (agent traces, process
RSS, hardware counters, OVMS logs), and ``TelemetrySink`` defines where
snapshots are written (JSON logs, live terminal buffers, fan-out combinators),
so N sources and M sinks compose independently via N+M implementations.
"""
from abc import ABC, abstractmethod


class TelemetrySource(ABC):
    """Something that can be asked for a snapshot of numbers."""

    #: Short stable id used as the key in a report and on the TUI page.
    name: str = "source"

    @abstractmethod
    def sample(self) -> dict:
        """Return the current numbers as a flat dict of name -> value.

        MUST NOT raise. A telemetry source that takes the run down with it has
        inverted its own purpose; a source that cannot read its hardware
        returns {} and says why through `unavailable`.
        """
        ...

    @property
    def unavailable(self) -> str | None:
        """Why this source cannot be used here, or None if it can.

        A sentence for a human: "Intel UT does not run on macOS" beats a
        silent empty sample, which is indistinguishable from zero load.
        """
        return None

    def start(self) -> None:
        """Begin collecting. Optional; polling sources need nothing here."""

    def stop(self) -> None:
        """Stop collecting and release anything held. Safe to call twice."""


class TelemetrySink(ABC):
    """Somewhere measurements are delivered."""

    name: str = "sink"

    @abstractmethod
    def record(self, sample: dict) -> None:
        """Accept one snapshot. MUST NOT raise, for the same reason as above:
        a failing exporter must not break the run it is measuring."""
        ...

    def close(self) -> None:
        """Flush and release. Safe to call twice, or never."""
