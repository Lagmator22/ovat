# ovat/telemetry/base.py
"""Layer 7 contracts: where measurements come FROM, and where they GO.

Telemetry has two independent axes, exactly like the LLM providers did, and
for exactly the same reason: getting it wrong once already cost this project a
real bug (four engines each holding their own copy of a setting).

  SOURCE  where a measurement comes from
          the agent's own run trace, this process's memory, Intel's hardware
          profiler, a plano gateway's OTel spans

  SINK    where it goes
          a JSON file, an OTLP collector, the live TUI page

Two sockets, not one, because the two vary independently: any source can feed
any sink. Wiring N sources to M sinks directly is N*M pieces of code that must
agree; through a contract it is N+M.

WHERE THIS PATTERN IS *NOT* WORTH IT, since that matters as much: an
abstraction with a single implementation is indirection you pay for and get
nothing back from. Both ABCs here already have three real implementations
between them, which is what earns them. The A2A server, by contrast, has one
protocol and should stay a plain class until a second appears.
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
