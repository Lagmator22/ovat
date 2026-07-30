# ovat/telemetry/collector.py
"""Ties sources to sinks and ticks them on a clock.

The whole point of the two contracts is that this file is tiny and neither
side knows about the other: N sources and M sinks wire up as N+M pieces
instead of N*M.
"""
import threading

from ovat.telemetry.base import TelemetrySink, TelemetrySource


class Collector:
    """Polls every source on an interval and hands each sample to the sink."""

    def __init__(self, sources: list, sink: TelemetrySink,
                 interval_s: float = 1.0):
        self.sources: list[TelemetrySource] = list(sources)
        self.sink = sink
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None

    @property
    def available(self) -> list:
        """The sources that can actually run here."""
        return [s for s in self.sources if s.unavailable is None]

    @property
    def unavailable(self) -> dict:
        """name -> the sentence explaining why, for the page to show.

        Shown rather than hidden: a missing source and an idle one look
        identical in a graph, and only one of them is worth acting on.
        """
        return {s.name: s.unavailable for s in self.sources
                if s.unavailable is not None}

    def sample_once(self) -> dict:
        """One merged snapshot from every available source.

        Keys are prefixed with the source name, so two sources reporting
        "rss_mb" cannot silently overwrite each other.
        """
        merged = {}
        for source in self.available:
            try:
                for key, value in (source.sample() or {}).items():
                    merged[f"{source.name}.{key}"] = value
            except Exception:
                # Contract says sample() must not raise; belt and braces, so
                # one broken source cannot end the collection thread.
                continue
        return merged

    def start(self) -> None:
        if self._thread is not None:
            return
        for source in self.available:
            try:
                source.start()
            except Exception:
                pass
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                self.sink.record(self.sample_once())
                self._stop.wait(self.interval_s)

        # daemon: a telemetry thread must never be the reason the CLI will
        # not exit.
        self._thread = threading.Thread(target=loop, daemon=True,
                                        name="ovat-telemetry")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        for source in self.sources:
            try:
                source.stop()
            except Exception:
                pass
